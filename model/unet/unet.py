# Copied and modified from https://github.com/archinetai/audio-diffusion-pytorch/blob/v0.0.94/audio_diffusion_pytorch/modules.py under MIT License
# License can be found in LICENSES/LICENSE_ADP.md

from inspect import isfunction
from math import floor, log, pi, log2
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from packaging import version

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, einsum
from torch.backends.cuda import sdp_kernel
from torch.nn import functional as F

from .utils import default, exists, closest_power_2, groupby
from .utils import TimePositionalEmbedding, STFT

from .cnn_blocks import Downsample1d, Upsample1d, ResnetBlock1d, Patcher, Unpatcher
from .attention_blocks import Transformer1d
from .unet_blocks import DownsampleBlock1d, UpsampleBlock1d, BottleneckBlock1d


class UNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[int],
        patch_size: int = 1,
        resnet_groups: int = 8,
        use_context_time: bool = True,
        kernel_multiplier_downsample: int = 2,
        use_nearest_upsample: bool = False,
        use_skip_scale: bool = True,
        use_snake: bool = False,
        out_channels: Optional[int] = None,
        context_features: Optional[int] = None,
        context_features_multiplier: int = 4,
        context_channels: Optional[Sequence[int]] = None,
        context_embedding_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)
        context_channels = list(default(context_channels, []))
        num_layers = len(multipliers) - 1
        use_context_features = exists(context_features)
        use_context_channels = len(context_channels) > 0
        context_mapping_features = None

        attention_kwargs, kwargs = groupby("attention_", kwargs, keep_prefix=True)

        self.num_layers = num_layers
        self.use_context_time = use_context_time
        self.use_context_features = use_context_features
        self.use_context_channels = use_context_channels

        self.context_features = context_features
        context_channels_pad_length = num_layers + 1 - len(context_channels)
        context_channels = context_channels + [0] * context_channels_pad_length
        self.context_channels = context_channels
        self.context_embedding_features = context_embedding_features

        if use_context_channels:
            has_context = [c > 0 for c in context_channels]
            self.has_context = has_context
            self.channels_ids = [sum(has_context[:i]) for i in range(len(has_context))]

        assert (
            len(factors) == num_layers
            and len(attentions) >= num_layers
            and len(num_blocks) == num_layers
        )

        if use_context_time or use_context_features:
            context_mapping_features = channels * context_features_multiplier

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )

        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        assert not kwargs, f"Unknown arguments: {', '.join(list(kwargs.keys()))}"

        self.to_in = Patcher(
            in_channels=in_channels + context_channels[0],
            out_channels=channels * multipliers[0],
            patch_size=patch_size,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    context_mapping_features=context_mapping_features,
                    context_channels=context_channels[i + 1],
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_skip=True,
                    use_snake=use_snake,
                    num_transformer_blocks=attentions[i],
                    **attention_kwargs,
                )
                for i in range(num_layers)
            ]
        )

        self.bottleneck = BottleneckBlock1d(
            channels=channels * multipliers[-1],
            context_mapping_features=context_mapping_features,
            context_embedding_features=context_embedding_features,
            num_groups=resnet_groups,
            num_transformer_blocks=attentions[-1],
            use_snake=use_snake,
            **attention_kwargs,
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i + 1],
                    out_channels=channels * multipliers[i],
                    context_mapping_features=context_mapping_features,
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i] + (1 if attentions[i] else 0),
                    factor=factors[i],
                    use_nearest=use_nearest_upsample,
                    num_groups=resnet_groups,
                    use_skip_scale=use_skip_scale,
                    use_pre_upsample=False,
                    use_skip=True,
                    use_snake=use_snake,
                    skip_channels=channels * multipliers[i + 1],
                    num_transformer_blocks=attentions[i],
                    **attention_kwargs,
                )
                for i in reversed(range(num_layers))
            ]
        )

        self.to_out = Unpatcher(
            in_channels=channels * multipliers[0],
            out_channels=out_channels,
            patch_size=patch_size,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

    def get_channels(
        self, channels_list: Optional[Sequence[Tensor]] = None, layer: int = 0
    ) -> Optional[Tensor]:
        """Gets context channels at `layer` and checks that shape is correct"""
        use_context_channels = self.use_context_channels and self.has_context[layer]
        if not use_context_channels:
            return None
        assert exists(channels_list), "Missing context"
        # Get channels index (skipping zero channel contexts)
        channels_id = self.channels_ids[layer]
        # Get channels
        channels = channels_list[channels_id]
        message = f"Missing context for layer {layer} at index {channels_id}"
        assert exists(channels), message
        # Check channels
        num_channels = self.context_channels[layer]
        message = f"Expected context with {num_channels} channels at idx {channels_id}"
        assert channels.shape[1] == num_channels, message

        return channels

    def get_mapping(
        self, time: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]
        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)
        return mapping

    def forward(
        self,
        x: Tensor,
        time: Optional[Tensor] = None,
        *,
        features: Optional[Tensor] = None,
        channels_list: Optional[Sequence[Tensor]] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False,
    ) -> Tensor:
        channels = self.get_channels(channels_list, layer=0)
        # Concat context channels at layer 0 if provided
        x = torch.cat([x, channels], dim=1) if exists(channels) else x
        # Compute mapping from time and features
        mapping = self.get_mapping(time, features)
        x = self.to_in(x, mapping)
        skips_list = [x]

        for i, downsample in enumerate(self.downsamples):
            channels = self.get_channels(channels_list, layer=i+1)
            x, skips = downsample(
                x, mapping=mapping, channels=channels, embedding=embedding, embedding_mask=embedding_mask, causal=causal
            )
            skips_list += [skips]

        x = self.bottleneck(x, mapping=mapping, embedding=embedding, embedding_mask=embedding_mask, causal=causal)

        for i, upsample in enumerate(self.upsamples):
            skips = skips_list.pop()
            x = upsample(x, skips=skips, mapping=mapping, embedding=embedding, embedding_mask=embedding_mask, causal=causal)

        x += skips_list.pop()
        x = self.to_out(x, mapping)

        return x


if __name__ == "__main__":
    unet = UNet1d(in_channels=128,
                  channels=128,
                  multipliers=[1, 2, 2, 2, 2],
                  factors=[2, 2, 2, 2],
                  num_blocks=[2, 2, 2, 2],
                  attentions=[0, 1, 1, 1],
                  attention_heads=8,
                  attention_features=64,
                  attention_multiplier=2,
                  context_embedding_features=768,
                  patch_size=16,
                  resnet_groups=8,
                  use_context_time=True,
                  kernel_multiplier_downsample=2,
                  use_nearest_upsample=True,
                  use_skip_scale=True,
                  use_snake=False,
                  out_channels=None,
                  context_features=None,
                  context_features_multiplier=4,
                  context_channels=None,
                  )

    x = torch.randn(2, 128, 256)
    time = torch.tensor([1, 2])
    embedding = torch.randn(2, 512, 768)
    y = unet(x, time=time, embedding=embedding)
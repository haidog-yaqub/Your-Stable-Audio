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

"""
Encoder/Decoder Components
"""


class DownsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_groups: int,
        num_layers: int,
        kernel_multiplier: int = 2,
        use_pre_downsample: bool = True,
        use_skip: bool = False,
        use_snake: bool = False,
        extract_channels: int = 0,
        context_channels: int = 0,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
    ):
        super().__init__()
        self.use_pre_downsample = use_pre_downsample
        self.use_skip = use_skip
        self.use_transformer = num_transformer_blocks > 0
        self.use_extract = extract_channels > 0
        self.use_context = context_channels > 0

        channels = out_channels if use_pre_downsample else in_channels

        self.downsample = Downsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            kernel_multiplier=kernel_multiplier,
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + context_channels if i == 0 else channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features,
                    use_snake=use_snake
                )
                for i in range(num_layers)
            ]
        )

        if self.use_transformer:
            assert (
                exists(attention_heads)
                and exists(attention_multiplier)
            )

            attention_features = default(attention_features, channels // attention_heads)

            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features
            )

        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
                use_snake=use_snake
            )

    def forward(
        self,
        x: Tensor,
        *,
        mapping: Optional[Tensor] = None,
        channels: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False
    ) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:

        if self.use_pre_downsample:
            x = self.downsample(x)

        if self.use_context and exists(channels):
            x = torch.cat([x, channels], dim=1)

        skips = []
        for block in self.blocks:
            x = block(x, mapping=mapping)
            skips += [x] if self.use_skip else []

        if self.use_transformer:
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)
            skips += [x] if self.use_skip else []

        if not self.use_pre_downsample:
            x = self.downsample(x)

        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted

        return (x, skips) if self.use_skip else x


class UpsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_layers: int,
        num_groups: int,
        use_nearest: bool = False,
        use_pre_upsample: bool = False,
        use_skip: bool = False,
        use_snake: bool = False,
        skip_channels: int = 0,
        use_skip_scale: bool = False,
        extract_channels: int = 0,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_extract = extract_channels > 0
        self.use_pre_upsample = use_pre_upsample
        self.use_transformer = num_transformer_blocks > 0
        self.use_skip = use_skip
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0

        channels = out_channels if use_pre_upsample else in_channels

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + skip_channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features,
                    use_snake=use_snake
                )
                for _ in range(num_layers)
            ]
        )

        if self.use_transformer:
            assert (
                exists(attention_heads)
                and exists(attention_multiplier)
            )

            attention_features = default(attention_features, channels // attention_heads)

            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
            )

        self.upsample = Upsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            use_nearest=use_nearest,
        )

        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
                use_snake=use_snake
            )

    def add_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(
        self,
        x: Tensor,
        *,
        skips: Optional[List[Tensor]] = None,
        mapping: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:

        if self.use_pre_upsample:
            x = self.upsample(x)

        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
            x = block(x, mapping=mapping)

        if self.use_transformer:
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)

        if not self.use_pre_upsample:
            x = self.upsample(x)

        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted

        return x


class BottleneckBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
        use_snake: bool = False,
    ):
        super().__init__()
        self.use_transformer = num_transformer_blocks > 0

        self.pre_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

        if self.use_transformer:
            assert (
                exists(attention_heads)
                and exists(attention_multiplier)
            )

            attention_features = default(attention_features, channels // attention_heads)

            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
            )

        self.post_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

    def forward(
        self,
        x: Tensor,
        *,
        mapping: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False
    ) -> Tensor:
        x = self.pre_block(x, mapping=mapping)
        if self.use_transformer:
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)
        x = self.post_block(x, mapping=mapping)
        return x


if __name__ == "__main__":
    model = UpsampleBlock1d(in_channels=128,
                            out_channels=128*2,
                            context_mapping_features=None,
                            context_embedding_features=None,
                            num_layers=2,
                            factor=2,
                            use_nearest=True,
                            num_groups=8,
                            use_skip_scale=True,
                            use_pre_upsample=False,
                            use_skip=True,
                            use_snake=False,
                            skip_channels=128,
                            num_transformer_blocks=2,
                            attention_heads=8,
                            attention_multiplier=1
                            )

    x = torch.rand(4, 128, 100)
    skip = [torch.rand(4, 128, 100), torch.rand(4, 128, 100)]
    y = model(x, skips=skip)
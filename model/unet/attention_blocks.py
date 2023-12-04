from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from packaging import version

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from torch import Tensor, einsum
from torch.backends.cuda import sdp_kernel
from torch.nn import functional as F

from .einops_exts import rearrange_many
from .utils import default, exists
from .cnn_blocks import Conv1d


"""
Attention Components
"""


def add_mask(sim: Tensor, mask: Tensor) -> Tensor:
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


def causal_mask(q: Tensor, k: Tensor) -> Tensor:
    b, i, j, device = q.shape[0], q.shape[-2], k.shape[-2], q.device
    mask = ~torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
    mask = repeat(mask, "n m -> b n m", b=b)
    return mask


class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features**-0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads
        out_features = default(out_features, features)

        self.to_out = nn.Linear(
            in_features=mid_features, out_features=out_features
        )

        self.use_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')

        if not self.use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # Use flash attention for A100 GPUs
            self.sdp_kernel_config = (True, False, False)
        else:
            # Don't use flash attention for other GPUs
            self.sdp_kernel_config = (False, True, True)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, is_causal: bool = False
    ) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        if not self.use_flash:
            if is_causal and not mask:
                # Mask out future tokens for causal attention
                mask = causal_mask(q, k)

            # Compute similarity matrix and add eventual mask
            sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale
            sim = add_mask(sim, mask) if exists(mask) else sim

            # Get attention matrix with softmax
            attn = sim.softmax(dim=-1, dtype=torch.float32)

            # Compute values
            out = einsum("... n m, ... m d -> ... n d", attn, v)
        else:
            with sdp_kernel(*self.sdp_kernel_config):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.context_features = context_features
        self.causal = causal
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            out_features=out_features,
        )

    def forward(
        self,
        x: Tensor, # [b, n, c]
        context: Optional[Tensor] = None, # [b, m, d]
        context_mask: Optional[Tensor] = None,  # [b, m], false is masked,
        causal: Optional[bool] = False,
    ) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)

        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        if exists(context_mask):
            # Mask out cross-attention for padding tokens
            mask = repeat(context_mask, "b m -> b m d", d=v.shape[-1])
            k, v = k * mask, v * mask

        # Compute and return attention
        return self.attention(q, k, v, is_causal=self.causal or causal)


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )

"""
Transformer Blocks
"""


class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None, context_mask: Optional[Tensor] = None, causal: Optional[bool] = False) -> Tensor:
        x = self.attention(x, causal=causal) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context, context_mask=context_mask) + x
        x = self.feed_forward(x) + x
        return x


"""
Transformers
"""


class Transformer1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.to_in = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True),
            Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
            Rearrange("b c t -> b t c"),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=channels,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    context_features=context_features,
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = nn.Sequential(
            Rearrange("b t c -> b c t"),
            Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None, context_mask: Optional[Tensor] = None, causal=False) -> Tensor:
        x = self.to_in(x)
        for block in self.blocks:
            x = block(x, context=context, context_mask=context_mask, causal=causal)
        x = self.to_out(x)
        return x


if __name__ == '__main__':
    model = Transformer1d(4, 256, 8, 64, 1)
    x = torch.rand(4, 256, 100)
    y = model(x)
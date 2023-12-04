from inspect import isfunction
from math import floor, log, pi, log2, ceil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor, einsum

from einops import rearrange, reduce, repeat
from .einops_exts import rearrange_many


"""
Utils
"""


class ConditionedSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(*modules)

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None):
        for module in self.module_list:
            x = module(x, mapping)
        return x


T = TypeVar("T")


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def exists(val: Optional[T]) -> T:
    return val is not None


def closest_power_2(x: float) -> int:
    exponent = log2(x)
    distance_fn = lambda z: abs(x - 2 ** z)  # noqa
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)


def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


"""
Time Embeddings
"""


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device, half_dim = x.device, self.dim // 2
        emb = torch.tensor(log(10000) / (half_dim - 1), device=device)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


"""
Audio Transforms
"""


class STFT(nn.Module):
    """Helper for torch stft and istft"""

    def __init__(
        self,
        num_fft: int = 1023,
        hop_length: int = 256,
        window_length: Optional[int] = None,
        length: Optional[int] = None,
        use_complex: bool = False,
    ):
        super().__init__()
        self.num_fft = num_fft
        self.hop_length = default(hop_length, floor(num_fft // 4))
        self.window_length = default(window_length, num_fft)
        self.length = length
        self.register_buffer("window", torch.hann_window(self.window_length))
        self.use_complex = use_complex

    def encode(self, wave: Tensor) -> Tuple[Tensor, Tensor]:
        b = wave.shape[0]
        wave = rearrange(wave, "b c t -> (b c) t")

        stft = torch.stft(
            wave,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            return_complex=True,
            normalized=True,
        )

        if self.use_complex:
            # Returns real and imaginary
            stft_a, stft_b = stft.real, stft.imag
        else:
            # Returns magnitude and phase matrices
            magnitude, phase = torch.abs(stft), torch.angle(stft)
            stft_a, stft_b = magnitude, phase

        return rearrange_many((stft_a, stft_b), "(b c) f l -> b c f l", b=b)

    def decode(self, stft_a: Tensor, stft_b: Tensor) -> Tensor:
        b, l = stft_a.shape[0], stft_a.shape[-1]  # noqa
        length = closest_power_2(l * self.hop_length)

        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> (b c) f l")

        if self.use_complex:
            real, imag = stft_a, stft_b
        else:
            magnitude, phase = stft_a, stft_b
            real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)

        stft = torch.stack([real, imag], dim=-1)

        wave = torch.istft(
            stft,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            length=default(self.length, length),
            normalized=True,
        )

        return rearrange(wave, "(b c) t -> b c t", b=b)

    def encode1d(
        self, wave: Tensor, stacked: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        stft_a, stft_b = self.encode(wave)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> b (c f) l")
        return torch.cat((stft_a, stft_b), dim=1) if stacked else (stft_a, stft_b)

    def decode1d(self, stft_pair: Tensor) -> Tensor:
        f = self.num_fft // 2 + 1
        stft_a, stft_b = stft_pair.chunk(chunks=2, dim=1)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b (c f) l -> b c f l", f=f)
        return self.decode(stft_a, stft_b)

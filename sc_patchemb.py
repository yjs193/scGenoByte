import torch.nn as nn
import logging
from typing import Callable, List, Optional, Tuple, Union
from itertools import repeat
import collections.abc

import torch
from torch import nn as nn
import torch.nn.functional as F
from enum import Enum

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            gene_size: Optional[int] = 224,
            patch_size: int = 100,
            embed_dim: int = 16,
            norm_layer: Optional[Callable] = None,
            strict_gene_size: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        if gene_size is not None:
            self.gene_size = gene_size
            self.num_patches = self.gene_size // self.patch_size
        else:
            self.gene_size = None
            self.num_patches = None

        self.strict_gene_size = strict_gene_size
        self.n_patch = gene_size // self.patch_size  # 16907/100
        self.proj = nn.Linear(patch_size, embed_dim)
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, N = x.shape  # (N, 25700)
        if self.gene_size is not None:
            if self.strict_gene_size:
                _assert(N == self.gene_size, f"Input height ({N}) doesn't match model ({self.gene_size}).")

            else:
                _assert(
                    N % self.patch_size == 0,
                    f"number of gene ({N}) should be divisible by patch size ({self.patch_size})."
                )
        x = x.view(B, self.n_patch, -1)  # (N, 25700)
        x = self.proj(x.float())  # (N, 257, 64)
        x = self.norm(x)
        return x

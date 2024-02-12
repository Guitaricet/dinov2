# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("DINO_USE_XFORMERS").lower() == "true"
if XFORMERS_ENABLED:
    from xformers.ops import memory_efficient_attention, unbind

# By default it will use the best available attention implementation,
# but if you want to ensure it uses Flash, use this environment variable
FORCE_FLASH = os.environ.get("DINO_FORCE_FLASH_ATTENTION").lower() == "true"
@contextmanager
def flash_attention(force=False):
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=not force, enable_mem_efficient=not force)
        yield
    finally:
        # Reset to default settings if needed
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn_drop_p = self.attn_drop if self.training else 0.0
        with flash_attention(force=FORCE_FLASH):
            attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=attn_drop_p)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_ENABLED:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

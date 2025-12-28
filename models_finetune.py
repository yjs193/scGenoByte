# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np

import timm.models.vision_transformer
from sc_patchemb import PatchEmbed
from util.my_flashattn import FlashAttentionBlock, FlashAttentionBlockTosa
from timm.models.vision_transformer import Block, Attention


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
        Vision Transformer with support for global average pooling
    """

    def __init__(self, num_pathways=0, **kwargs):  # <-- 【修改点 1】: 添加 num_pathways 参数
        super(VisionTransformer, self).__init__(**kwargs)
        img_size = kwargs['img_size']
        patch_size = kwargs['patch_size']
        embed_dim = kwargs['embed_dim']
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim) * .02)
        self.blocks = nn.ModuleList([
            FlashAttentionBlock(embed_dim, 8, 4, qkv_bias=True, attn_drop=0.1, norm_layer=nn.LayerNorm)
            for i in range(12)])

        # 【修改点 2】: 添加新的通路重构头
        # self.head 已经由父类 __init__ 定义 (用于分类)
        # 我们添加一个新的头用于通路重构
        # 假设 self.embed_dim 是可用的 (它在父类中被设置)
        if num_pathways > 0:
            self.pathway_head = nn.Linear(self.embed_dim, num_pathways)
        else:
            # 如果 num_pathways 为 0，则退化为原始模型
            self.pathway_head = nn.Identity()

        self.num_pathways = num_pathways

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        # 【修改点 3】: 修改 forward 方法以返回两个输出
        x_embed = self.forward_features(x)  # (B, embed_dim)

        # 任务 1: 细胞分类
        x_class = self.head(x_embed)

        # 任务 2: 通路重构
        if self.num_pathways > 0:
            x_pathway = self.pathway_head(x_embed)
        else:
            x_pathway = torch.tensor(0.0, device=x_embed.device)  # 占位符

        return x_class, x_pathway  # 返回两个头的输出



def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

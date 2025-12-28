from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from functools import partial
from timm.models.layers import DropPath, Mlp
import numpy as np
import torch
import torch.nn as nn


class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_weights = None  # 用于存储注意力权重

    def forward(self, x):
        # (N, 97, 64)
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        if torch.isnan(qkv).any():
            qkv = torch.nan_to_num(qkv)
        attn_output = flash_attn_qkvpacked_func(qkv, dropout_p=0.1).reshape(B, N, C)
        self.attn_weights = attn_output.mean(-1)
        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class FlashAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn_weights = None  # 用于存储注意力权重

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        self.attn_weights = self.attn.attn_weights  # 从 Attention 类中获取注意力权重
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FlashAttentionTosa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_weights = None  # 用于存储注意力权重

    def forward(self, x, return_attention=False):
        """
        FlashAttentionTosa前向传播。

        Args:
        - x: 输入张量，形状为 (B, N, C)，其中：
            B = 批量大小，
            N = 序列长度，
            C = 嵌入维度。
        - return_attention: 如果为True，返回 Q, K 和注意力权重。

        Returns:
        - x: 输出张量，形状为 (B, N, C)。
        - (可选) attn_output: 注意力结果（如果 `return_attention` 为 True）。
        - (可选) q, k: Q 和 K 投影结果（如果 `return_attention` 为 True）。
        """
        B, N, C = x.shape

        # 计算 QKV 投影
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        if torch.isnan(qkv).any():
            qkv = torch.nan_to_num(qkv)

        # 分离 Q, K, V
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # 形状: (B, N, num_heads, head_dim)

        # FlashAttention计算
        attn_output = flash_attn_qkvpacked_func(qkv, dropout_p=0.1).reshape(B, N, C)

        # 最后的投影
        x = self.proj(attn_output)
        x = self.proj_drop(x)

        if return_attention:
            # 返回输出、注意力权重、以及 Q/K
            return x, attn_output, q, k
        else:
            # 仅返回输出
            return x


class FlashAttentionBlockTosa(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttentionTosa(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn_weights = None  # 用于存储注意力权重

    def forward(self, x, return_attention=False):
        """
        FlashAttentionBlockTosa前向传播。

        Args:
        - x: 输入张量，形状为 (B, N, D)，其中：
            B = 批量大小，
            N = 序列长度，
            D = 嵌入维度。
        - return_attention: 如果为True，返回 Q, K 和注意力权重。

        Returns:
        - x: 输出张量，形状为 (B, N, D)。
        - (可选) attn_output: 注意力结果（如果 `return_attention` 为 True）。
        - (可选) q, k: Q 和 K 投影结果（如果 `return_attention` 为 True）。
        """
        if torch.isnan(x).any():
            print("original x has nan value")

        if return_attention:
            # 获取注意力权重及 Q/K 投影
            x_normed = self.norm1(x)
            x, attn_output, q, k = self.attn(x_normed, return_attention=True)
            x = x + self.drop_path(x)
            self.attn_weights = attn_output  # 保存注意力权重
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn_output, q, k  # 返回 x、注意力结果、Q、K
        else:
            # 标准前向传播
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

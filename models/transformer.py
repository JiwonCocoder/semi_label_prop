import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class Attention(nn.Module):
    def __init__(self, hidden_dim, qk_dim, v_dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or hidden_dim ** -0.5
        
        self.hidden_dim = hidden_dim
        self.q = nn.Linear(qk_dim, hidden_dim, bias=qkv_bias)
        self.k = nn.Linear(qk_dim, hidden_dim, bias=qkv_bias)
        self.v = nn.Linear(v_dim, hidden_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        B, N, C = query.shape
        _, N_key, _ = key.shape
        _, N_value, _ = value.shape

        q = self.q(query).reshape(B, N, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, N_key, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, N_value, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, v_dim, output_dim, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(
            dim, dim, v_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Linear(dim, output_dim)

    def forward(self, x, value):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), value))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return torch.softmax(self.proj(x), dim=2)
    
    def forward_cross(self, query, key, value):
        x = query + self.drop_path(self.attn(self.norm1(query), self.norm1(key), value))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return torch.softmax(self.proj(x), dim=2)


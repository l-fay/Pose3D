import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Pose3D(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, num_joints2=11, num_joints_out=7,
                 norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim3 = embed_dim_ratio * num_joints_out
        embed_dim2 = embed_dim_ratio * num_joints2
        embed_dim1 = embed_dim_ratio * num_joints
        embed_dim_head = embed_dim_ratio * num_joints
        out_dim = num_joints_out * 3
        out_dim2 = num_joints * 3
        self.embed_dim_ratio = embed_dim_ratio
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed3 = nn.Parameter(torch.zeros(1, num_joints_out, embed_dim_ratio))
        self.Spatial_pos_embed2 = nn.Parameter(torch.zeros(1, num_joints2, embed_dim_ratio))
        self.Spatial_pos_embed1 = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed3 = nn.Parameter(torch.zeros(1, num_frame, embed_dim3))
        self.Temporal_pos_embed2 = nn.Parameter(torch.zeros(1, num_frame, embed_dim2))
        self.Temporal_pos_embed1 = nn.Parameter(torch.zeros(1, num_frame, embed_dim1))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.part1con = nn.Conv2d(3, 2, kernel_size=1)
        self.part2con = nn.Conv2d(3, 2, kernel_size=1)
        self.part3con = nn.Conv2d(2, 1, kernel_size=1)
        self.part4con = nn.Conv2d(2, 1, kernel_size=1)
        self.part5con = nn.Conv2d(3, 2, kernel_size=1)
        self.part6con = nn.Conv2d(3, 2, kernel_size=1)
        self.part1conT = nn.ConvTranspose2d(2, 3, kernel_size=1)
        self.part2conT = nn.ConvTranspose2d(2, 3, kernel_size=1)
        self.part3conT = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part4conT = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part5conT = nn.ConvTranspose2d(2, 3, kernel_size=1)
        self.part6conT = nn.ConvTranspose2d(2, 3, kernel_size=1)
        self.part1con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part2con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part5con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part6con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part1conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part2conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part5conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part6conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.blocks1 = nn.ModuleList([
            Block(
                dim=embed_dim1, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.blocks2 = nn.ModuleList([
            Block(
                dim=embed_dim2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim3, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm3 = norm_layer(embed_dim3)
        self.Temporal_norm2 = norm_layer(embed_dim2)
        self.Temporal_norm1 = norm_layer(embed_dim1)
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim1),
            nn.Linear(embed_dim1, out_dim2),
        )
        self.hidden = nn.Linear(2, 1)

    def Spatial_forward_features1(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p  -> (b f) p  c', )
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed1
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def Spatial_forward_features2(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p  -> (b f) p  c', )
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed2
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def Spatial_forward_features3(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p  -> (b f) p  c', )
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed3
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features1(self, x):
        b = x.shape[0]
        x += self.Temporal_pos_embed1
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.Temporal_norm1(x)
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def forward_features2(self, x):
        b = x.shape[0]
        x += self.Temporal_pos_embed2
        x = self.pos_drop(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.Temporal_norm2(x)
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def forward_features3(self, x):
        b = x.shape[0]
        x += self.Temporal_pos_embed3
        x = self.pos_drop(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.Temporal_norm3(x)
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def skip(self, x1, x2, x3):
        x1 = rearrange(x1, 'b f (p e) -> b f p e', e=self.embed_dim_ratio)
        x2 = rearrange(x2, 'b f (p e) -> b f p e', e=self.embed_dim_ratio)
        x3 = rearrange(x3, 'b f (p e) -> b f p e', e=self.embed_dim_ratio)
        x3to2 = self.do7to11(x3)
        x2 = torch.stack((x2, x3to2))
        x2 = self.hidden(x2.permute(1, 2, 3, 4, 0))
        a, b, c, d, _ = x2.shape
        x2 = x2.view(a, b, c, d)
        x2to1 = self.do11to17(x2)
        x1 = torch.stack((x1, x2to1))
        x1 = self.hidden(x1.permute(1, 2, 3, 4, 0))
        a, b, c, d, _ = x1.shape
        x1 = x1.view(a, b, c, d)
        x = rearrange(x1, 'b f p e -> b f (p e)')
        return x

    def do17to11(self, x):
        part0 = x[:, 0:1, :, :]
        part1 = x[:, 1:4, :, :]
        part2 = x[:, 4:7, :, :]
        part3 = x[:, 7:9, :, :]
        part4 = x[:, 9:11, :, :]
        part5 = x[:, 11:14, :, :]
        part6 = x[:, 14:, :, :]
        part1 = self.part1con(part1)
        part2 = self.part2con(part2)
        part3 = self.part3con(part3)
        part4 = self.part4con(part4)
        part5 = self.part5con(part5)
        part6 = self.part6con(part6)
        x = torch.cat((part0, part1, part2, part3, part4, part5, part6), 1)
        return x

    def do11to7(self, x):
        part0 = x[:, 0:1, :, :]
        part1 = x[:, 1:3, :, :]
        part2 = x[:, 3:5, :, :]
        part3 = x[:, 5:6, :, :]
        part4 = x[:, 6:7, :, :]
        part5 = x[:, 7:9, :, :]
        part6 = x[:, 9:, :, :]
        part1 = self.part1con2(part1)
        part2 = self.part2con2(part2)
        part5 = self.part5con2(part5)
        part6 = self.part6con2(part6)
        x = torch.cat((part0, part1, part2, part3, part4, part5, part6), 1)
        return x

    def do7to11(self, x):
        x = x.permute(0, 2, 1, 3)
        part0 = x[:, 0:1, :, :]
        part1 = x[:, 1:2, :, :]
        part2 = x[:, 2:3, :, :]
        part3 = x[:, 3:4, :, :]
        part4 = x[:, 4:5, :, :]
        part5 = x[:, 5:6, :, :]
        part6 = x[:, 6:, :, :]
        part1 = self.part1conT2(part1)
        part2 = self.part2conT2(part2)
        part5 = self.part5conT2(part5)
        part6 = self.part6conT2(part6)
        x = torch.cat((part0, part1, part2, part3, part4, part5, part6), 1)
        x = x.permute(0, 2, 1, 3)
        return x

    def do11to17(self, x):
        x = x.permute(0, 2, 1, 3)
        part0 = x[:, 0:1, :, :]
        part1 = x[:, 1:3, :, :]
        part2 = x[:, 3:5, :, :]
        part3 = x[:, 5:6, :, :]
        part4 = x[:, 6:7, :, :]
        part5 = x[:, 7:9, :, :]
        part6 = x[:, 9:, :, :]
        part1 = self.part1conT(part1)
        part2 = self.part2conT(part2)
        part3 = self.part3conT(part3)
        part4 = self.part4conT(part4)
        part5 = self.part5conT(part5)
        part6 = self.part6conT(part6)
        x = torch.cat((part0, part1, part2, part3, part4, part5, part6), 1)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        x1 = x.permute(0, 2, 1, 3)
        x2 = self.do17to11(x1)
        x3 = self.do11to7(x2)
        x1 = x1.permute(0, 3, 2, 1)
        x2 = x2.permute(0, 3, 2, 1)
        x3 = x3.permute(0, 3, 2, 1)
        b1, _, _, p1 = x1.shape
        b2, _, _, p2 = x2.shape
        b3, _, _, p3 = x3.shape
        x1 = self.Spatial_forward_features1(x1)
        x2 = self.Spatial_forward_features2(x2)
        x3 = self.Spatial_forward_features3(x3)
        x1 = self.forward_features1(x1)
        x2 = self.forward_features2(x2)
        x3 = self.forward_features3(x3)
        x = self.skip(x1, x2, x3)
        x = self.head(x)
        x = x.view(b1, 1, p1, -1)
        return x

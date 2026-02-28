from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
# modified from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np


def posemb_sincos_1d(z, temperature=10000, dtype=torch.float32):
    # https://github.com/shawnazhao/Transformer-for-time-series-forecasting-/blob/main/utils.py#L14
    _, h, dim, device, dtype = *z.shape, z.device, z.dtype

    # Compute the positional encodings once in log space.
    pe = torch.zeros(h, dim)
    position = torch.arange(0, h).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(temperature) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe.type(dtype).to(device)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        pe = posemb_sincos_1d(x)  # b x h x dim
        x = x + pe
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

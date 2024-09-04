# https://github.com/lucidrains/vit-pytorch/tree/main

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype=dtype)

class RMSNorm(nn.Module):
    def __init__(self, heads, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(heads, 1, dim) / self.scale)

    def forward(self, x):
        normed = F.normalize(x, dim = -1, eps=self.eps)
        return normed * self.scale * self.weight

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # self.q_norm = RMSNorm(heads, dim_head)
        # self.k_norm = RMSNorm(heads, dim_head)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # q = self.q_norm(q)
        # k = self.k_norm(k)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = torch.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout = dropout),
                nn.LayerNorm(dim),
        ]))

    def forward(self, x):
        for attn, norm1, ff, norm2 in self.layers:
            skip = x
            x = norm1(attn(x) + x)
            x = norm2(ff(x) + x + skip)

        return x

class convBlock(nn.Module):
    def __init__(self, indim, middim, outdim, depth = 1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(indim, middim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(middim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(middim, middim, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm2d(middim),
                nn.SiLU(),
        ) for _ in range(depth)])
        self.out = nn.Sequential(
            nn.Conv2d(middim, outdim, kernel_size=1, bias=False),
            nn.BatchNorm2d(outdim),
        )

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x) + x
        return self.out(x)

class mixViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        channels = 3
    ):
        super().__init__()

        self.stage1 = nn.Sequential(
            convBlock(channels, 64, 64, depth=5),
            convBlock(64, 128, 128, depth=5),
        )

        self.stage2 = convBlock(128, 256, 256, depth=10)
        self.stage3 = convBlock(256, 512, 512, depth=10)
        self.stage4 = convBlock(512, 1024, dim, depth=0)

        self.flatten = Rearrange("b c h w -> b (h w) c")
        self.pos_embedding = nn.Parameter(posemb_sincos_2d(h = image_size // 32, w = image_size // 32, dim = dim).unsqueeze(0))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.split = Rearrange("b (h w) c -> b c h w", h = image_size // 32)

    def forward(self, img):
        out = []
        x = self.stage1(img)
        out.append(x)
        x = self.stage2(x)
        out.append(x)
        x = self.stage3(x)
        out.append(x)
        x = self.stage4(x)
        x = self.flatten(x)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.split(x)
        out.append(x)
        return tuple(out)

if __name__ == '__main__':
    v = mixViT(
        image_size = 512,
        dim = 512,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    print(v)

    img = torch.randn(1, 3, 512, 512)
    print([x.shape for x in v(img)])
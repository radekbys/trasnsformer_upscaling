import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x)  # (B, embed_dim, H//P, W//P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = x + self.pos_embed
        return x, self.grid_size


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(in_features=hidden_features, out_features=in_features),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=drop_rate,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim, hidden_features=mlp_dim, drop_rate=drop_rate
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, out_channels, upsample_factor):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                embed_dim,
                out_channels,
                kernel_size=upsample_factor,
                stride=upsample_factor,
            ),
        )

    def forward(self, x, grid_size):
        B, N, E = x.shape
        H, W = grid_size  # grid size from patch embedding
        x = x.transpose(1, 2).reshape(B, E, H, W)  # (B, E, H, W)
        x = self.decode(x)  # (B, 3, H*scale, W*scale)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_dim,
        drop_rate=0.2,
        in_channels=3,
        upsample_factor=2,
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=upsample_factor, mode="bilinear", align_corners=False
        )

        upsampled_size = (img_size[0] * upsample_factor, img_size[1] * upsample_factor)

        self.patchEmbed = PatchEmbedding(
            img_size=upsampled_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    drop_rate=drop_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = Decoder(
            embed_dim=embed_dim, out_channels=3, upsample_factor=patch_size
        )

    def forward(self, x):
        x = self.upsample(x)
        x, grid_size = self.patchEmbed(x)  # (B, N, E)
        x = self.encoder(x)  # (B, N, E)
        x = self.norm(x)  # (B, N, E)
        x = self.decoder(x, grid_size)  # (B, 3, H, W)
        return x

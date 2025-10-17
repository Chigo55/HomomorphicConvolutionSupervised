import torch.nn as nn
from torch import Tensor


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int, dropout_ratio: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=embed_dim * mlp_ratio,
                bias=False,
            ),
            nn.GELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(
                in_features=embed_dim * mlp_ratio,
                out_features=embed_dim * mlp_ratio,
                bias=False,
            ),
            nn.GELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(
                in_features=embed_dim * mlp_ratio,
                out_features=embed_dim,
                bias=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_norm(x)
        x = self.mlp(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_ratio: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_ratio,
            batch_first=True,
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        return attn


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout_ratio: float,
    ):
        super().__init__()
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_ratio=dropout_ratio,
        )
        self.mlp = MultiLayerPerceptron(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x) + x
        x = self.mlp(x) + x
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormSuper(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim: int, eps: float = 1e-5):
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

    def set_sample_config(self, sample_embed_dim: int):
        self.sample_embed_dim = sample_embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            (self.sample_embed_dim,),
            weight=self.weight[: self.sample_embed_dim],
            bias=self.bias[: self.sample_embed_dim],
            eps=self.eps,
        )

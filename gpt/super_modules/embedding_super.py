import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperEmbedding(torch.nn.Embedding):
    def __init__(self, vocab_size: int, super_embed_dim: int):
        super().__init__(vocab_size, super_embed_dim)

        # the largest embed dim
        self.vocab_size = vocab_size
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

    def set_sample_config(self, sample_embed_dim: int):
        self.sample_embed_dim = sample_embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            x,
            self.weight[:, : self.sample_embed_dim],
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

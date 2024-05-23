import torch.nn as nn
import torch
from typing import Any, Optional, Tuple
class SuperRotaryEmbedding(nn.Module):
    def __init__(self, config: Any, max_seq_length: int) -> None:
        super().__init__()
        self.config = config
        self.max_seq_length = max_seq_length
        self.sample_head_size = None
    
    def build_rope_cache(
        self, seq_len: int,  device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license."""
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        head_size = int(self.sample_head_size * self.config.rotary_percentage)
        theta = 1.0 / (base ** (torch.arange(0, head_size, 2, device=device).float() / head_size))
        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=device) / condense_ratio

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

        return torch.cos(idx_theta).to(device), torch.sin(idx_theta).to(device)
    
    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.build_rope_cache(
            seq_len=self.max_seq_length,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )
    
    def set_sample_config(self, sample_embed_dim: int, sample_num_heads) -> None:
        self.sample_head_size = self.config.head_size


    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        head_size = int(self.sample_head_size * self.config.rotary_percentage)
        #print(head_size)
        x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
        x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
        rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
        roped = (x * cos) + (rotated * sin)
        return roped.to(dtype=x.dtype)
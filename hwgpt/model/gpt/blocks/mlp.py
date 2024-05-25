import torch
import torch.nn as nn
from typing import Any, Optional, Tuple
from hwgpt.model.gpt.config import Config
from hwgpt.model.gpt.super_modules.linear_super import SuperLinear
from torch.utils.checkpoint import checkpoint


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = SuperLinear(config.n_embd, config.intermediate_size)
        self.proj = SuperLinear(config.intermediate_size, config.n_embd)
        self.config = config
        self.act = torch.nn.GELU(approximate=self.config.gelu_approximate)

    def set_sample_config(
        self,
        sample_embed_dim: int,
        sample_intermediate_size: int,
        sample_bias_flag: bool,
    ) -> None:
        # print(sample_embed_dim, sample_intermediate_size, sample_bias_flag)
        self.fc.set_sample_config(
            sample_embed_dim, sample_intermediate_size, sample_bias_flag
        )
        self.proj.set_sample_config(
            sample_intermediate_size, sample_embed_dim, sample_bias_flag
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = checkpoint(self.act, x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = SuperLinear(config.n_embd, config.intermediate_size)
        self.fc_2 = SuperLinear(config.n_embd, config.intermediate_size)
        self.proj = SuperLinear(config.intermediate_size, config.n_embd)
        self.act = nn.SiLU()

    def set_sample_config(
        self,
        sample_embed_dim: int,
        sample_intermediate_size: int,
        sample_bias_flag: bool,
    ) -> None:
        self.fc_1.set_sample_config(
            sample_embed_dim, sample_intermediate_size, sample_bias_flag
        )
        self.fc_2.set_sample_config(
            sample_embed_dim, sample_intermediate_size, sample_bias_flag
        )
        self.proj.set_sample_config(
            sample_intermediate_size, sample_embed_dim, sample_bias_flag
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = checkpoint(self.act, x_fc_1) * x_fc_2
        return self.proj(x)

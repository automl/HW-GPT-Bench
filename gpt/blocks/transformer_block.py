import torch
import torch.nn as nn
from typing import Any, Optional, Tuple
from gpt.config import Config
from gpt.blocks.causal_self_attention import CausalSelfAttention


class Block(nn.Module):
    def __init__(self, config: Config, rotary_emb: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.norm_1 = self.norm_class()(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, rotary_emb)
        self.norm_2 = (
            None
            if config.shared_attention_norm
            else self.norm_class()(config.n_embd, eps=config.norm_eps)
        )
        self.mlp = self.mlp_class()(config)

    def set_sample_config(
        self,
        sample_embed_dim: int,
        sample_intermediate_size: int,
        sample_num_heads: int,
        sample_bias_flag: bool,
    ) -> None:
        self.norm_1.set_sample_config(sample_embed_dim)
        self.attn.set_sample_config(
            sample_embed_dim, sample_num_heads, sample_bias_flag
        )
        if not self.config.shared_attention_norm:
            self.norm_2.set_sample_config(sample_embed_dim)
        self.mlp.set_sample_config(
            sample_embed_dim, sample_intermediate_size, sample_bias_flag
        )

    def norm_class(self):
        # `self._norm_class` cannot be the type to keep the config json serializable
        from gpt.super_modules.rmsnorm_super import RMSNormSuper
        from gpt.super_modules.layernorm_super import LayerNormSuper

        if self.config._norm_class == "RMSNorm":

            return RMSNormSuper
        return LayerNormSuper

    def mlp_class(self):
        # `self._mlp_class` cannot be the type to keep the config json serializable
        from gpt.blocks.mlp import GptNeoxMLP, LLaMAMLP

        if self.config._mlp_class == "LLaMAMLP":
            return LLaMAMLP
        return GptNeoxMLP

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_1 = self.norm_1(x)
        h = self.attn(n_1, mask, input_pos)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(n_2) + h + x
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = h + x
            x = self.mlp(self.norm_2(x)) + x
        return x

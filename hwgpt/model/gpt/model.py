"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self

from hwgpt.model.gpt.config import Config
from hwgpt.model.gpt.blocks import Block
from hwgpt.model.gpt.super_modules.embedding_super import SuperEmbedding
from hwgpt.model.gpt.super_modules.rotary_embedding import SuperRotaryEmbedding
from hwgpt.model.gpt.super_modules.lmhead_super import LMHeadSuper


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = LMHeadSuper(
            config.n_embd, config.padded_vocab_size, config.lm_head_bias
        )
        self.rotary_embeddings = [
            SuperRotaryEmbedding(config, config.block_size)
            for _ in range(config.n_layer)
        ]
        self.rotary_dummy = SuperRotaryEmbedding(config, config.block_size)
        self.transformer = nn.ModuleDict(
            dict(
                wte=SuperEmbedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(
                    Block(config, self.rotary_embeddings[i])
                    for i in range(config.n_layer)
                ),
                ln_f=self.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_layer = config.n_layer
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

        self.sample_embed_dim = None  # type: Optional[int]
        self.sample_intermediate_size = None
        self.sample_num_heads = None
        self.transformer.wte.weight = self.lm_head.weight

    @property
    def norm_class(self):
        # `self._norm_class` cannot be the type to keep the config json serializable
        from hwgpt.model.gpt.super_modules.rmsnorm_super import RMSNormSuper
        from hwgpt.model.gpt.super_modules.layernorm_super import LayerNormSuper

        if self.config._norm_class == "RMSNorm":

            return RMSNormSuper
        return LayerNormSuper

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def set_sample_config(
        self,
        sample_embed_dim: int,
        sample_intermediate_size: list,
        sample_num_heads: list,
        sample_n_layer: int,
        sample_bias_flag: bool,
        sample_layer_indices: list,
    ) -> None:
        self.sample_embed_dim = sample_embed_dim
        self.sample_intermediate_size = sample_intermediate_size
        self.sample_num_heads = sample_num_heads
        self.sample_n_layer = sample_n_layer
        self.sample_bias_flag = sample_bias_flag
        self.sample_layer_indices = sample_layer_indices
        self.transformer.wte.set_sample_config(sample_embed_dim)
        self.transformer.ln_f.set_sample_config(sample_embed_dim)
        self.rotary_dummy.set_sample_config(self.config.n_embd, self.config.n_head)
        # self.rotary_embedding.set_sample_config(self.config.n_embd, self.config.n_head)
        # print(sample_layer_indices)
        for i in sample_layer_indices:
            block = self.transformer.h[i]
            block.set_sample_config(
                sample_embed_dim,
                sample_intermediate_size[i],
                sample_num_heads[i],
                sample_bias_flag,
            )
        self.lm_head.set_sample_config(sample_embed_dim, sample_bias_flag)

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}"
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            self.rotary_dummy.set_sample_config(self.config.n_embd, self.config.n_head)
            cos, sin = self.rotary_dummy.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rotary_dummy.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )
        # self.reset_parameters() #TODO: we need to reset rope cache every time (might be inefficient)
        if input_pos is not None:  # use the kv cache
            # cos = self.cos.index_select(0, input_pos)
            # sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            # cos = self.cos[:T]
            # sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # print(self.sample_layer_indices)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)
        for i in self.sample_layer_indices:
            block = self.transformer.h[i]
            x = block(x, mask, input_pos)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )"""

        if self.mask_cache is None or self.mask_cache.size(3) != self.max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(self.max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        # for block in self.transformer.h:
        #    block.attn.kv_cache = None


def build_mask_cache(
    max_seq_length: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)

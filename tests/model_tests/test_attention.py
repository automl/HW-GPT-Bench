from gpt.blocks.causal_self_attention import CausalSelfAttention
import torch
from gpt.config import Config
from gpt.super_modules.rotary_embedding import SuperRotaryEmbedding
sample_embed_dim = [512,256]
sample_n_head = [2, 4]
sample_bias = [True]
n_layer = 2
n_head = 4
n_embd = 1024
block_size = 100
vocab_size = 500
config_dict = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size,padded_vocab_size=vocab_size, rotary_percentage=1.0, intermediate_size=1024*4)

config = Config(**config_dict)
config.head_size=64
config.n_query_groups = 1
config.device= "cpu"
rotary_emb = SuperRotaryEmbedding(config, 1000)
causal_self_attention = CausalSelfAttention(config, rotary_emb)
#causal_self_attention.attn.weight.data = torch.ones((config.n_head + 2 * config.n_query_groups) * config.head_size, config.n_embd)
#causal_self_attention.attn.bias.data = torch.ones((config.n_head + 2 * config.n_query_groups) * config.head_size)
#causal_self_attention.proj.weight.data = torch.ones(config.n_embd, 64*config.n_head)
#causal_self_attention.proj.bias.data = torch.ones(config.n_embd)

for emb in sample_embed_dim:
    for n_head in sample_n_head:
        for bias in sample_bias:
            causal_self_attention.set_sample_config(emb, n_head, bias)
            x = torch.ones(1, 100, emb)
            print(emb)
            print(n_head)
            print(bias)
            y = causal_self_attention(x)
            #if not bias:
            #    assert torch.sum(y) == emb*emb*100*emb
            #else:
            #    assert torch.sum(y) == emb*emb*n_head*3 + emb*3
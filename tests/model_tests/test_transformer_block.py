from gpt.blocks.transformer_block import Block
import torch
from gpt.config import Config
from gpt.super_modules.rotary_embedding import SuperRotaryEmbedding
sample_embed_dim = [512, 256]
sample_mlp_ratio = [4, 2]
sample_n_head = [2, 4]
sample_bias = [True, False]
n_layer = 2
n_head = 4
n_embd = 1024
block_size = 100
vocab_size = 500
config_dict = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, 
                  bias=False, vocab_size=vocab_size,padded_vocab_size=vocab_size, rotary_percentage=1.0, intermediate_size=1024*4)


block = Block(Config(**config_dict), SuperRotaryEmbedding(Config(**config_dict), 1000))
block.attn.attn.weight.data = torch.ones((block.config.n_head + 2 * block.config.n_query_groups) * block.config.head_size, block.config.n_embd)
block.attn.attn.bias.data = torch.ones((block.config.n_head + 2 * block.config.n_query_groups) * block.config.head_size)
block.attn.proj.weight.data = torch.ones(block.config.n_embd, block.config.n_embd)
block.attn.proj.bias.data = torch.ones(block.config.n_embd)
block.norm_1.weight.data = torch.ones(block.config.n_embd)
block.norm_1.bias.data = torch.ones(block.config.n_embd)
block.mlp.fc.weight.data = torch.ones(block.config.intermediate_size, block.config.n_embd)
block.mlp.fc.bias.data = torch.ones(block.config.intermediate_size)
block.mlp.proj.weight.data = torch.ones(block.config.n_embd, block.config.intermediate_size)
block.mlp.proj.bias.data = torch.ones(block.config.n_embd)
block.norm_2.weight.data = torch.ones(block.config.n_embd)
block.norm_2.bias.data = torch.ones(block.config.n_embd)

for emb in sample_embed_dim:
    for mlp_ratio in sample_mlp_ratio:
        for n_head in sample_n_head:
          for bias in sample_bias:
            block.set_sample_config(emb, emb*mlp_ratio, n_head, bias)
            x = torch.ones(1, 100, emb)
            print(emb)
            print(n_head)
            print(bias)
            y = block(x)
            #if not bias:
            #    assert torch.sum(y) == emb*emb*100*emb
            #else:
            #    assert torch.sum(y) == emb*emb*n_head*3 + emb*3

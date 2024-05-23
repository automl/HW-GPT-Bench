from gpt.super_modules.rotary_embedding import SuperRotaryEmbedding
from gpt.super_modules.embedding_super import SuperEmbedding
from gpt.config import Config
import torch


super_embed_dim = 1024
sample_embed_dim = [512, 256]
sample_heads = [2, 4]
max_seq_len = 1000
vocab_size = 500
n_layer = 2
n_head = 2
n_embd = 1024
block_size = 100
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size,padded_vocab_size=vocab_size, rotary_percentage=1.0)
embed_super = SuperEmbedding(vocab_size, super_embed_dim)
embed_super.weight.data = torch.ones(vocab_size, super_embed_dim)
config = Config(**model_args)
rotary_embed_super = SuperRotaryEmbedding(config, max_seq_len)

for emb in sample_embed_dim:
    embed_super.set_sample_config(emb)
    x = torch.randint(0, vocab_size, (1, max_seq_len))
    y = embed_super(x)
    print(torch.sum(y).item())
    assert torch.sum(y) == emb*max_seq_len

for heads in sample_heads:
    for emb in sample_embed_dim:
        rotary_embed_super.set_sample_config(emb, heads)
        head_size = emb // heads
        x = torch.ones(1, max_seq_len, head_size)
        cos, sin = rotary_embed_super.rope_cache()
        y = rotary_embed_super.apply_rope(x[...,:int(head_size*config.rotary_percentage)], cos, sin)
        y = torch.cat((y,x[...,int(head_size*config.rotary_percentage):]), dim=-1)
        print(torch.sum(y).item())
        rope_input = x[...,:int(head_size* config.rotary_percentage)]
        rope_input_x1 = rope_input[...,::2]
        rope_input_x2 = rope_input[...,1::2]
        rope_output_manual = cos + sin  * torch.cat((-rope_input_x2, rope_input_x1), dim=-1)
        out_manual = (head_size-(config.rotary_percentage*head_size))*max_seq_len+torch.sum(rope_output_manual)
        #print(out_manual)
        assert torch.allclose(torch.sum(y), out_manual, atol=1e-2)
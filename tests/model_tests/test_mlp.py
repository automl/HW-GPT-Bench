
from gpt.blocks.mlp import GptNeoxMLP, LLaMAMLP
import torch 
from gpt.config import Config


sample_embed_dim = [512, 256]
sample_mlp_ratio = [4, 2]
sample_bias = [True, False]
n_layer = 2
n_head = 2
n_embd = 1024
block_size = 100
vocab_size = 500
config_dict = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size,padded_vocab_size=vocab_size, rotary_percentage=1.0, intermediate_size=1024*4)

config = Config(**config_dict)
mlp = GptNeoxMLP(config)
mlp.fc.weight.data = torch.ones(config.intermediate_size, config.n_embd)
mlp.fc.bias.data = torch.ones(config.intermediate_size)
mlp.proj.weight.data = torch.ones(config.n_embd, config.intermediate_size)
mlp.proj.bias.data = torch.ones(config.n_embd)
llama_mlp = LLaMAMLP(config)
llama_mlp.fc_1.weight.data = torch.ones(config.intermediate_size, config.n_embd)
llama_mlp.fc_1.bias.data = torch.ones(config.intermediate_size)
llama_mlp.fc_2.weight.data = torch.ones(config.intermediate_size, config.n_embd)
llama_mlp.fc_2.bias.data = torch.ones(config.intermediate_size)
llama_mlp.proj.weight.data = torch.ones(config.n_embd, config.intermediate_size)
llama_mlp.proj.bias.data = torch.ones(config.n_embd)


for emb in sample_embed_dim:
    for m in sample_mlp_ratio:
        for bias in sample_bias:
            mlp.set_sample_config(emb, emb*m, bias)
            llama_mlp.set_sample_config(emb, emb*m, bias)
            x = torch.ones(1, emb)
            y = mlp(x)
            y_llama = llama_mlp(x)
            if not bias:
                assert  torch.allclose(torch.sum(y), emb*emb*m*torch.nn.functional.gelu(torch.tensor(emb*1.0), approximate=config.gelu_approximate), atol=1e-2)
                assert torch.sum(y_llama) == emb*emb*m*torch.nn.functional.silu(torch.tensor(emb*1.0))*emb
            else:
                assert torch.allclose(torch.sum(y), emb*emb*m*torch.nn.functional.gelu(torch.tensor(emb+1.0), approximate=config.gelu_approximate) + emb, atol=1e-2)
                assert torch.allclose(torch.sum(y_llama),emb*emb*m*torch.nn.functional.silu(torch.tensor(emb+1.0))*(emb+1.0) + emb, atol=1e-2)

                #assert torch.sum(y_llama) == emb*emb*m*3 + emb*m*2 + emb



from gpt.super_modules.rmsnorm_super import RMSNormSuper
from gpt.super_modules.layernorm_super import LayerNormSuper
import torch

super_embed_dim = 1024
sample_embed_dim = [512, 256]

rmsnorm_super = RMSNormSuper(super_embed_dim)
rmsnorm_super.weight.data = torch.ones(super_embed_dim)
layernorm_super = LayerNormSuper(super_embed_dim)
layernorm_super.weight.data = torch.ones(super_embed_dim)
layernorm_super.bias.data = torch.ones(super_embed_dim)

for emb in sample_embed_dim:
    rmsnorm_super.set_sample_config(emb)
    layernorm_super.set_sample_config(emb)
    x = torch.ones(1, 10, emb)
    y = rmsnorm_super(x)
    print(torch.sum(y).item())
    assert torch.allclose(torch.sum(y), torch.tensor(emb).float()*10, atol=1e-1)
    y = layernorm_super(x)
    print(torch.sum(y).item())
    assert torch.sum(y) == emb*10

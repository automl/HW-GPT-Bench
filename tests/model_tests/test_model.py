from gpt.model import GPT
from gpt.config import Config
import torch 
import torch.nn as nn
import numpy as np
from gpt.utils import *
sample_embed_dim = [512, 256]
sample_n_head = [2, 4]
sample_mlp_ratio = [4, 2]
sample_bias = [True, False]
n_layer = 12
sample_layer = [1,2,3,4,5,6,7,8,9,10,11,12]
n_head = 4
n_embd = 1024
block_size = 100
vocab_size = 500
config_dict = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size,padded_vocab_size=vocab_size, rotary_percentage=1.0, intermediate_size=1024*4)


config = Config(**config_dict)
gpt = GPT(config)
choices_dict = {}
choices_dict['n_layer_choices'] = sample_layer
choices_dict['n_head_choices'] = sample_n_head
choices_dict['embed_dim_choices'] = sample_embed_dim
choices_dict['mlp_ratio_choices'] = sample_mlp_ratio
choices_dict['bias_choices'] = sample_bias
sampled_config = sample_config(choices_dict, layer_sampling_scheme="normal")
gpt.set_sample_config(sampled_config["sample_embed_dim"], sampled_config["sample_mlp_ratio"]*sampled_config["sample_embed_dim"], sampled_config["sample_n_head"], sampled_config["sample_n_layer"], sampled_config["sample_bias"], sampled_config["sample_layer_indices"])
x = torch.randint(0, 500, (1, 100))
y = gpt(x)
print(y.shape)

sampled_config = sample_config(choices_dict, layer_sampling_scheme="strided")
gpt.set_sample_config(sampled_config["sample_embed_dim"], sampled_config["sample_mlp_ratio"]*sampled_config["sample_embed_dim"], sampled_config["sample_n_head"], sampled_config["sample_n_layer"], sampled_config["sample_bias"], sampled_config["sample_layer_indices"])
x = torch.randint(0, 500, (1, 100))
y = gpt(x)
print(y.shape)

#print(y)
                    



import torch
import torchvision.models as models
from data_collection.gpt_profiler.utils.flop_utils import get_flops_macs_params

"""model = models.alexnet()
batch_size = 1
input_shape = (batch_size, 3, 224, 224)
flops_value, macs_value, params_value, flops_unit, macs_unit, params_unit = get_flops_macs_params(model, input_shape)
print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops_value, macs_value, params_value))
# with units
print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops_unit, macs_unit, params_unit))

model_name = "mistralai/Mistral-7B-v0.1"
batch_size, max_seq_length = 1, 128
flops_value, macs_value, params_value, flops_unit, macs_unit, params_unit = get_flops_macs_params(model_name, (batch_size, max_seq_length), model_type="llm")
print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(model_name, flops_value, macs_value, params_value))
# with units
print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(model_name, flops_unit, macs_unit, params_unit))
"""
from deepspeed.profiling.flops_profiler import FlopsProfiler

from generate.base import generate
import torch


def get_config(model_name="gpt2"):
    if model_name == "gpt2":
        num_layers = 12
        num_heads = 12
        d_model = 12 * 64
    elif model_name == "gpt2-medium":
        num_layers = 24
        num_heads = 16
        d_model = 1024
    elif model_name == "gpt2-large":
        num_layers = 36
        num_heads = 20
        d_model = 1280
    elif model_name == "gpt2-xl":
        num_layers = 48
        num_heads = 25
        d_model = 1600
    return num_layers, num_heads, d_model


model_name = "gpt2"
from gpt.config import Config
from gpt.model import GPT
from gpt.utils import *

sample_embed_dim = [512, 256]
sample_n_head = [2, 4]
sample_mlp_ratio = [4, 2]
sample_bias = [True, False]
n_layer = 12
n_head = 8
n_embd = 1024
block_size = 1000
vocab_size = 5000
sample_layer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
choices_dict = {}
choices_dict["n_layer_choices"] = sample_layer
choices_dict["n_head_choices"] = sample_n_head
choices_dict["embed_dim_choices"] = sample_embed_dim
choices_dict["mlp_ratio_choices"] = sample_mlp_ratio
choices_dict["bias_choices"] = sample_bias
config_dict = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=False,
    vocab_size=vocab_size,
    padded_vocab_size=vocab_size,
    rotary_percentage=1.0,
    intermediate_size=1024 * 4,
)
sampled_config = sample_config(choices_dict, layer_sampling_scheme="normal")


model = GPT(Config(**config_dict))
# inputs
model.set_sample_config(
    sampled_config["sample_embed_dim"],
    sampled_config["sample_mlp_ratio"] * sampled_config["sample_embed_dim"],
    sampled_config["sample_n_head"],
    sampled_config["sample_n_layer"],
    sampled_config["sample_bias"],
    sampled_config["sample_layer_indices"],
)
model_inputs_x = torch.randint(0, 5000, (1, 1000))
model_inputs_y = torch.randint(0, 5000, (1, 1000))
# model(model_inputs_x)
flops, macs, params = get_flops_macs_params(model, model_inputs_x)
print("FLOPs:%s  MACs:%s  Params:%s " % (flops, macs, params))

from profiler.utils.latency_profile_utils import torch_profiler_conv, torch_record_function_conv, torch_profiler_llm, torch_record_function_llm
import torchvision.models as models
import torch
# test convnet
model = models.alexnet().cuda()
inputs = torch.randn([2,3,224,224]).cuda()
mean_cpu, std_cpu, mean_gpu, std_gpu, unit_cpu, unit_gpu = torch_profiler_conv(model, inputs, n=100)
print("Mean time on cpu {} {}".format(mean_cpu,unit_cpu))
print("Standard deviation time on cpu {} {}".format(std_cpu, unit_cpu))
print("Mean time on gpu {} {}".format(mean_gpu,unit_gpu))
print("Standard deviation time on gpu {} {}".format(std_gpu, unit_gpu))
mean_cpu, std_cpu, mean_gpu, std_gpu, unit_cpu, unit_gpu = torch_record_function_conv(model, inputs, n=100)
print("Mean time on cpu {} {}".format(mean_cpu,unit_cpu))
print("Standard deviation time on cpu {} {}".format(std_cpu, unit_cpu))
print("Mean time on gpu {} {}".format(mean_gpu,unit_gpu))
print("Standard deviation time on gpu {} {}".format(std_gpu, unit_gpu))

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
sample_layer = [1,2,3,4,5,6,7,8,9,10,11,12]
choices_dict = {}
choices_dict['n_layer_choices'] = sample_layer
choices_dict['n_head_choices'] = sample_n_head
choices_dict['embed_dim_choices'] = sample_embed_dim
choices_dict['mlp_ratio_choices'] = sample_mlp_ratio
choices_dict['bias_choices'] = sample_bias
config_dict = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size,padded_vocab_size=vocab_size, rotary_percentage=1.0, intermediate_size=1024*4)
sampled_config = sample_config(choices_dict, layer_sampling_scheme="normal")
        

model = GPT(Config(**config_dict))
# inputs
model.set_sample_config(sampled_config["sample_embed_dim"], sampled_config["sample_mlp_ratio"]*sampled_config["sample_embed_dim"], sampled_config["sample_n_head"], sampled_config["sample_n_layer"], sampled_config["sample_bias"], sampled_config["sample_layer_indices"])
model_inputs_x = torch.randint(0, 5000, (1,1000))
model_inputs_y = torch.randint(0, 5000, (1,1000))
# profile

print("Mean time on cpu {} {}".format(mean_cpu,unit_cpu))
print("Standard deviation time on cpu {} {}".format(std_cpu, unit_cpu))
print("Mean time on gpu {} {}".format(mean_gpu,unit_gpu))
print("Standard deviation time on gpu {} {}".format(std_gpu, unit_gpu))
mean_cpu, std_cpu, mean_gpu, std_gpu, unit_cpu, unit_gpu = torch_record_function_llm(model, model_inputs_x, model_inputs_y, n=10,use_cpu=True)
print("Mean time on cpu {} {}".format(mean_cpu,unit_cpu))
print("Standard deviation time on cpu {} {}".format(std_cpu, unit_cpu))
print("Mean time on gpu {} {}".format(mean_gpu,unit_gpu))
print("Standard deviation time on gpu {} {}".format(std_gpu, unit_gpu))
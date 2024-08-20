from litgpt import Config
from whittle.models.gpt import GPT
from downstream_eval.utils import convert_and_evaluate
import torch
from syne_tune.config_space import choice, randint
from whittle.sampling.random_sampler import RandomSampler
from transformers import GPT2TokenizerFast
import yaml

class HWGPT(GPT):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
    
    def select_sub_network_hwgpt(self, config):
        self.set_sub_network(
            config["embed_dim"],
            [config["mlp_ratio_"+str(i)] * config["embed_dim"] for i in range(config["depth"])],
            [config["num_heads_"+str(i)] for i in range(config["depth"])],
            config["depth"],
        )
# Load the YAML file
class Struct:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

with open('configs/config_eval/owt_l_eval.yaml', 'r') as file:
    data = yaml.safe_load(file)

data = Struct(data)
data.model["fix_head_size"] = True
config = Config(n_embd=data.model["n_embd"], n_head = data.model["n_head"], n_layer=data.model["n_layer"], intermediate_size=data.model["intermediate_size"], n_query_groups=data.model["n_query_groups"], 
padded_vocab_size=50304, scale_embeddings=data.model["scale_embeddings"], vocab_size=50304,
bias=True, lm_head_bias=data.model["lm_head_bias"], norm_eps=1e-5, rope_condense_ratio=data.model["rope_condense_ratio"], rope_base=data.model["rope_base"], rotary_percentage=data.model["rotary_percentage"], block_size=data.model["block_size"])
config.fix_head_size = True

whittle_hwgpt = HWGPT(config)
whittle_hwgpt.name_or_path = "gpt"
whittle_hwgpt.config.model_type = "gpt"
whittle_hwgpt.config.tie_embeddings = True
# NOTE: replace this with the local model path 
load_path = "/p/project/projectnucleus/sukthanker1/HW-Aware-LLM-Bench/experiments/owt_large_fast/owt_large_fast/default_juls/last.ckpt"
state_dict = torch.load(load_path, map_location="cpu")
updated_state_dict = {}
for n in state_dict["state_dict"].keys(): 
    if "model" in n:
        updated_state_dict[n[6:]] = state_dict["state_dict"][n]
    else:
        updated_state_dict[n] = state_dict["state_dict"][n]
whittle_hwgpt.load_state_dict(updated_state_dict)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=".")
tokenizer.add_special_tokens({"pad_token": "<pad>"})
search_space_gpt = {
    "embed_dim": choice(list(data.model["embed_choices"])),
    "depth": choice(list(data.model["layer_choices"])),
}
for i in range(data.model["n_layer"]):
    search_space_gpt["mlp_ratio_"+str(i)] = choice(list(data.model["mlp_ratio_choices"]))
    search_space_gpt["num_heads_"+str(i)] = choice(list(data.model["head_choices"]))
sampler = RandomSampler(config_space=search_space_gpt, seed=42)
for i in range(100):
    config = sampler.sample()
    whittle_hwgpt.select_sub_network_hwgpt(config)
    convert_and_evaluate(
        whittle_hwgpt,
        out_dir="test/",
        device=None,
        dtype=torch.float32,
        tasks="arc_easy",
        batch_size=16,  # Test for non-positive integer
        tokenizer=tokenizer
    )
import os
import re
import yaml
import pathlib
import copy
from typing import Dict, List
import importlib
import inspect
from collections.abc import MutableMapping
import torch


class SimpleNestedNamespace(Dict):
    def __init__(self, *args, **kwargs):

        super().__init__(**kwargs)

        for k, v in kwargs.items():
            if isinstance(v, Dict):
                kwargs[k] = SimpleNestedNamespace(**v)

        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return self.__dict__.__str__()

    @property
    def mlp_class(self):
        # `self._mlp_class` cannot be the type to keep the config json serializable
        from hwgpt.model.gpt_base.model import GptNeoxMLP, LLaMAMLP

        if self._mlp_class == "GptNeoMLP":
            return GptNeoxMLP
        else:
            return LLaMAMLP

    @property
    def norm_class(self):
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from functools import partial

            from hwgpt.model.gpt_base.rmsnorm import RMSNorm

            return partial(RMSNorm, add_unit_offset="Gemma" in self.name)
        return getattr(torch.nn, self._norm_class)


class Config(SimpleNestedNamespace):

    def __init__(self, config_file=None, config_dict=None):

        if config_file is None and config_dict is None:
            raise UserWarning("ConfigHandler: config_file and config_dict is None")

        elif config_file is not None and config_dict is None:
            with open(config_file, "r") as f:
                config_dict = yaml.load(f, Loader=yaml.Loader)

        def convert_exponential_string(s):
            pattern = r"^(?:\s|\b)([+-]?[0-9]*\.?[0-9]+)[eE]([+-]?[0-9]+)?(?:\s|\b)$"

            match = re.search(pattern, s)
            if match:
                mantissa = float(match.group(1))
                exponent = int(match.group(2))
                result = mantissa * (10**exponent)
                if result.is_integer():
                    return int(result)
                else:
                    return result
            else:
                return s

        def get_attr_by_link(obj, links):
            attr = obj[links[0]]
            if isinstance(attr, Dict) and len(links) > 1:
                return get_attr_by_link(attr, links[1:])
            return attr

        def replace_linker(dictionary):
            for k, v in dictionary.items():
                if isinstance(v, str):
                    dictionary[k] = convert_exponential_string(v)
                if isinstance(v, Dict):
                    replace_linker(v)
                if (
                    isinstance(v, str)
                    and len(v) > 3
                    and v[0] == "$"
                    and v[1] == "{"
                    and v[-1] == "}"
                ):
                    links = v[2:-1].split(".")
                    dictionary[k] = get_attr_by_link(config_dict, links)

        replace_linker(config_dict)

        super().__init__(**config_dict)

    def get_dict(self):
        def resolve_namespace(dictionary):
            for k, v in dictionary.items():
                if isinstance(v, SimpleNestedNamespace):
                    dictionary[k] = resolve_namespace(v.__dict__)
            return dictionary

        dictionary = copy.deepcopy(self.__dict__)
        return resolve_namespace(dictionary)

    def save_config(self, directory, file_name="config.yml"):
        dir = pathlib.Path(directory)
        dir.mkdir(parents=True, exist_ok=True)
        with open(dir / file_name, "w+") as f:
            config_dict = self.get_dict()
            yaml.dump(config_dict, f, default_flow_style=False, encoding="utf-8")
        return dir / file_name


if __name__ == "__main__":
    config = Config(
        config_file=os.path.join(
            "config/juwels_default.yaml"
        )
    )

    print(type(config.deepspeed.allgather_bucket_size))
    print(config)
    from hwgpt.model.gpt.model import GPT
    from hwgpt.model.gpt.utils import sample_config

    sample_embed_dim = config.model.embed_choices
    sample_n_head = config.model.head_choices
    sample_mlp_ratio = config.model.mlp_ratio_choices
    sample_bias = config.model.mlp_ratio_choices
    n_layer = config.model.n_layer
    sample_layer = config.model.layer_choices
    n_head = config.model.n_head
    n_embd = config.model.n_embd
    block_size = config.model.block_size
    vocab_size = config.model.vocab_size
    config.model.n_head = n_head
    config.model.n_embd = n_embd
    config.model.block_size = block_size
    config.model.padded_vocab_size = vocab_size
    config.model.n_layer = n_layer
    config.model.head_size = n_embd // n_head
    config.model.n_query_groups = n_head
    gpt = GPT(config.model).cuda()  # .half()
    choices_dict = {}
    choices_dict["n_layer_choices"] = sample_layer
    choices_dict["n_head_choices"] = sample_n_head
    choices_dict["embed_dim_choices"] = sample_embed_dim
    choices_dict["mlp_ratio_choices"] = sample_mlp_ratio
    choices_dict["bias_choices"] = sample_bias

    for i in range(10):
        sampled_config = sample_config(choices_dict, layer_sampling_scheme="normal")
        gpt.set_sample_config(
            sampled_config["sample_embed_dim"],
            sampled_config["sample_mlp_ratio"] * sampled_config["sample_embed_dim"],
            sampled_config["sample_n_head"],
            sampled_config["sample_n_layer"],
            sampled_config["sample_bias"],
            sampled_config["sample_layer_indices"],
        )
        x = torch.randint(0, vocab_size, (2, block_size)).cuda()  # .half()
        print(x.shape)
        y = gpt(x)
        print(y.shape)

        sampled_config = sample_config(choices_dict, layer_sampling_scheme="strided")
        gpt.set_sample_config(
            sampled_config["sample_embed_dim"],
            sampled_config["sample_mlp_ratio"] * sampled_config["sample_embed_dim"],
            sampled_config["sample_n_head"],
            sampled_config["sample_n_layer"],
            sampled_config["sample_bias"],
            sampled_config["sample_layer_indices"],
        )
        x = torch.randint(0, vocab_size, (2, block_size)).cuda()  # .half()
        print(x.shape)
        y = gpt(x)
        print(y.shape)

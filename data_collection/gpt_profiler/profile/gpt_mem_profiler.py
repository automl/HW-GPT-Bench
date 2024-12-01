from hwgpt.model.gpt_base.model import GPT
from hwgpt.model.gpt.utils import sample_config
import pickle
from data_collection.gpt_profiler.utils.measure_memory_usage import (
    compute_memory_consumed,
)
from typing import Any
import os
import torch
import argparse


class GPTMemProfiler:
    """
    class for gpu memory profiling
    """

    def __init__(
        self,
        args: argparse.Namespace,
        cfg_model: Any,
        batch_size: int = 8,
        num_archs_to_evaluate: int = 10000,
        num_evals: int = 10,
        save_path: str = "latency_a100/",
        resume_path: str = "none",
        search_space: str = "s",
    ) -> None:
        super().__init__()
        # build choices dict
        self.args = args
        self.choices_dict = {}
        self.gpu_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg_model.gpu_dtype]
        self.choices_dict["n_layer_choices"] = cfg_model.layer_choices
        self.choices_dict["n_head_choices"] = cfg_model.head_choices
        self.choices_dict["embed_dim_choices"] = cfg_model.embed_choices
        self.choices_dict["mlp_ratio_choices"] = cfg_model.mlp_ratio_choices
        self.choices_dict["bias_choices"] = cfg_model.bias_choices
        self.num_archs_to_evaluate = num_archs_to_evaluate
        self.cfg_model = cfg_model
        self.batch_size = batch_size
        self.num_evals = num_evals
        self.save_path = save_path
        self.search_space = search_space
        self.arch_path = (
            "sampled_archs/" + "sampled_archs_" + str(search_space) + ".pkl"
        )
        self.lat_bench = []
        self.archs_evaluated = []
        if resume_path != "none" and os.path.exists(resume_path):
            import pickle

            with open(resume_path, "rb") as f:
                self.lat_bench = pickle.load(f)
            self.evaluated_archs()
            self.num_archs_to_evaluate = self.num_archs_to_evaluate - len(
                self.archs_evaluated
            )
        else:
            self.archs_evaluated = []
        os.makedirs(save_path, exist_ok=True)

    def evaluated_archs(self):
        self.archs_evaluated = []
        for arch in self.lat_bench:
            self.archs_evaluated.append(arch["arch"])

    def sample_n_random_archs(self):
        self.archs_sampled = []
        while len(self.archs_sampled) < self.num_archs_to_evaluate:
            arch_sampled = sample_config(
                self.choices_dict, layer_sampling_scheme="normal"
            )
            if (
                arch_sampled not in self.archs_sampled
            ) and arch_sampled not in self.archs_evaluated:
                self.archs_sampled.append(arch_sampled)
        # save archs to pickle file
        with open(self.arch_path, "wb") as f:
            pickle.dump(self.archs_sampled, f)

    def reset_config(self, arch_config):
        self.cfg_model.n_embd = arch_config["sample_embed_dim"]
        self.cfg_model.n_layer = arch_config["sample_n_layer"]
        self.cfg_model.n_head = arch_config["sample_n_head"]
        self.cfg_model.mlp_ratio = arch_config["sample_mlp_ratio"]
        self.cfg_model.bias = arch_config["sample_bias"]

    def create_model(self, arch_config):
        self.reset_config(arch_config)
        return GPT(self.cfg_model)

    def compute_metrics(self, arch_config):
        model = self.create_model(arch_config)
        model_inputs_x = torch.randint(
            0, self.cfg_model.vocab_size, (self.batch_size, self.cfg_model.block_size)
        )  # .cuda()#.half()
        model_inputs_y = torch.randint(
            0, self.cfg_model.vocab_size, (self.batch_size, self.cfg_model.block_size)
        )  # .cuda()
        # compute memory
        mean_mem, std_mem = compute_memory_consumed(
            model,
            model_inputs_x,
            n=5,
            use_gpu=True,
            use_cpu=False,
            gpu_dtype=self.gpu_dtype,
        )
        self.lat_bench.append(
            {"arch": arch_config, "mean_mem": mean_mem, "std_mem": std_mem}
        )
        with open(
            self.save_path
            + "/efficiency_observations_"
            + str(self.args.start_index)
            + "_"
            + str(self.args.end_index)
            + ".pkl",
            "wb",
        ) as f:
            pickle.dump(self.lat_bench, f)

    def run(self):
        if os.path.exists(self.arch_path):
            with open(self.arch_path, "rb") as f:
                self.archs_sampled = pickle.load(f)
            self.archs_sampled = self.archs_sampled[
                self.args.start_index : self.args.end_index
            ]
            if self.archs_evaluated != []:
                self.archs_sampled = [
                    arch
                    for arch in self.archs_sampled
                    if arch not in self.archs_evaluated
                ]

        else:
            self.sample_n_random_archs()

        for arch_config in self.archs_sampled:
            self.compute_metrics(arch_config)


if __name__ == "__main__":
    from pl_gpt.utils.configuration import Config
    import argparse

    parser = argparse.ArgumentParser(description="GPT Profiler")
    parser.add_argument(
        "--config",
        type=str,
        default="config_latency/latency_rtx2080.yaml",
        help="path to config file",
    )
    parser.add_argument("--start_index", type=int, default=0, help="start index")
    parser.add_argument("--end_index", type=int, default=10000, help="end index")
    parser.add_argument("--scale", type=str, default="s", help="search space")
    args = parser.parse_args()
    config = Config(config_file=args.config)

    config_model = config.model
    args.resume = (
        config_model.latency_bench_save_path
        + "/efficiency_observations_"
        + str(args.start_index)
        + "_"
        + str(args.end_index)
        + ".pkl"
    )
    print(config_model)
    profiler = GPTMemProfiler(
        args,
        config_model,
        save_path=config_model.latency_bench_save_path,
        resume_path=args.resume,
        search_space=args.scale,
    )
    profiler.run()

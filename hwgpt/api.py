import torch
import random
from hwgpt.model.gpt.utils import sample_config
from lib.utils import (
    search_spaces,
    get_hw_predictor_surrogate,
    get_ppl_predictor_surrogate,
    get_arch_feature_map,
    convert_config_to_one_hot,
    normalize_arch_feature_map,
    predict_hw_surrogate,
    convert_arch_to_str,
    convert_str_to_arch,
)
from hwgpt.predictors.hwmetric.models.mlp.net import Net
from hwgpt.api_utils import estimate_flops, num_parameters
from hwgpt.model.gpt_base.model import GPT
from data_collection.pl_gpt.utils.configuration import Config
from data_collection.gpt_profiler.profile.gpt_perplexity_profiler import GPTProfilerPPL
from hwgpt.predictors.hwmetric.models.autogluon.autogluon_latencies import (
    MultilabelPredictor,
)
from typing import Any, Dict
from argparse import Namespace
import pickle


class HWGPTBenchAPI:
    def __init__(
        self,
        search_space: str,
        use_supernet_surrogate: bool = False,
    ):
        print(search_spaces)

        self.search_space = search_spaces[search_space]
        self.search_space_name = search_space
        self.device_list = [
            "a100",
            "a40",
            "h100",
            "rtx2080",
            "rtx3080",
            "a6000",
            "v100",
            "P100",
            "cpu_xeon_silver",
            "cpu_xeon_gold",
            "cpu_amd_7502",
            "cpu_amd_7513",
            "cpu_amd_7452",
        ]
        self.hw_metrics_surrogate = [
            "latencies",
            "energies",
            "float16_memory",
            "bfloat16_memory",
        ]
        self.hw_metrics_true = ["flops", "params", "float16_memory", "bfloat16_memory"]
        self.metrics = ["perplexity", "accuracy"]
        self.config = None
        self.use_supernet_surrogate = use_supernet_surrogate
        self.surrogate_ppl = get_ppl_predictor_surrogate(self.search_space_name)
        self.on_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg_model = self.get_model_config()
        gt_stats_path = (
            "data_collection/gpt_datasets/gpt_"
            + str(self.search_space_name)
            + "/stats.pkl"
        )
        with open(gt_stats_path, "rb") as f:
            self.gt_stats = pickle.load(f)
        if self.use_supernet_surrogate:
            self.gpt_ppl_profiler = GPTProfilerPPL(
                self.prepare_args_for_ppl_profiler(), self.cfg_model
            )

    def init_devices(self):
        self.device_query = self.device_list
        self.hw_metrics_surrogate_query = self.hw_metrics_surrogate
        self.hw_metrics_true_query = self.hw_metrics_true

    def get_model_config(self) -> Config:
        config = Config(
            config_file="hwgpt/configs_api/gpt_" + self.search_space_name + ".yaml"
        )
        return config

    def prepare_args_for_ppl_profiler(self) -> Namespace:
        args = Namespace()
        args.config = "hwgpt/configs_api/gpt_" + self.search_space_name + ".yaml"
        args.num_archs_to_evaluate = 0
        args.num_evals = 10
        args.resume_path = "none"
        args.model_scale = self.search_space_name
        return args

    def reset_config(self) -> None:
        self.cfg_model.model.n_embd = self.config["sample_embed_dim"]
        self.cfg_model.model.n_layer = self.config["sample_n_layer"]
        self.cfg_model.model.n_head = self.config["sample_n_head"]
        self.cfg_model.model.mlp_ratio = self.config["sample_mlp_ratio"]
        self.cfg_model.model.bias = self.config["sample_bias"]

    def create_model(self) -> GPT:
        self.reset_config()
        return GPT(self.cfg_model.model)

    def sample_random_arch(self) -> dict:
        seed = hash(
            (
                random.randint(0, 1000000),
                random.randint(0, 1000000),
                random.randint(0, 1000000),
            )
        )
        config = sample_config(self.search_space, seed)
        self.config = config
        self.reset_config()

    def set_arch(self, config: dict):
        self.config = config
        self.reset_config()

    def compute_predictions_ppl(self):
        arch_feature = convert_config_to_one_hot(
            self.config, search_space=self.search_space_name
        )
        predictions_surrogate = self.surrogate_ppl(
            arch_feature.to(self.on_device).unsqueeze(0)
        ).item()
        return predictions_surrogate

    def set_metrics_and_devices(self, hw_metric: str = None, device: str = None):
        if hw_metric is not None:
            if hw_metric in self.hw_metrics_true:
                self.hw_metrics_true_query = [hw_metric]
                self.hw_metrics_surrogate_query = []
            else:
                self.hw_metrics_surrogate_query = [hw_metric]
                self.hw_metrics_true_query = []
        if device is not None:
            self.device_query = [device]

    def get_flops(self):
        flops = estimate_flops(self.create_model())
        return flops

    def get_params(self):
        params = num_parameters(self.create_model())
        return params

    def get_memory(self, objective):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_layers = max(search_spaces[search_space]["n_layer_choices"])
        hw_predictor = Net(num_layers, False, 128, 128).to(device)
        hw_predictor.load_state_dict(
            torch.load(
                "data_collection/gpt_datasets/predictor_ckpts/hwmetric/mlp/"
                + str(objective)
                + "_"
                + str(search_space)
                + ".pth",
                map_location=device,
            )
        )
        arch_feature_map = get_arch_feature_map(self.config, self.search_space_name)
        arch_feature_map_predictor = normalize_arch_feature_map(
            arch_feature_map, self.search_space_name
        )
        hw_metric = hw_predictor(
            torch.tensor(arch_feature_map_predictor).to(device).unsqueeze(0)
        )

        return hw_metric.item()

    def compute_predictions_hw(
        self,
        hw_metric: str,
        device: str,
    ) -> Any:
        arch_feature = get_arch_feature_map(self.config, self.search_space_name)
        # arch_feature = normalize_arch_feature_map(arch_feature, self.search_space_name)
        surrogate = get_hw_predictor_surrogate(
            self.search_space_name,
            device,
            hw_metric,
        )
        predictions_hw = predict_hw_surrogate(
            [arch_feature],
            surrogate,
            hw_metric,
            device,
        )
        return predictions_hw

    def eval_supernet_surrogate(self) -> Dict[str, float]:
        self.gpt_ppl_profiler.create_model(self.config)
        results = self.gpt_ppl_profiler.return_metrics(self.config)
        return results

    def get_gt_latencies(self, device: str, arch_str: str) -> list:
        return self.gt_stats[arch_str][device]["latencies"]

    def get_gt_energies(self, device: str, arch_str: str) -> list | float:
        return self.gt_stats[arch_str][device]["energies"]

    def get_gt_agnostic(self, arch_str: str, metric: str) -> float:
        return self.gt_stats[arch_str][metric]

    def query_gt_all_archs(self, metric: str, device: str = None):
        metric_stats = {}
        for arch in self.gt_stats:
            if metric == "energies":
                metric_stats[arch] = self.get_gt_energies(device, arch)
            elif metric == "latencies":
                metric_stats[arch] = self.get_gt_latencies(device, arch)
            elif metric in [
                "flops",
                "params",
                "bfloat16_memory",
                "float16_memory",
                "perplexity",
            ]:
                metric_stats[arch] = self.get_gt_agnostic(arch, metric)
            else:
                raise ValueError("Unsupported metric type")
        return metric_stats

    def query(
        self,
        device: str = "a100",
        hw_metric: str = "latencies",
    ):

        if self.config is None:
            raise ValueError("Please set arch config before querying")
        results = {}
        results["perplexity"] = self.compute_predictions_ppl()
        self.set_metrics_and_devices(hw_metric, device)
        for hw_metric in self.hw_metrics_surrogate_query:
            results[hw_metric] = {}
            for device in self.device_query:
                results[hw_metric][device] = self.compute_predictions_hw(
                    hw_metric, device
                )
        for hw_metric_true in self.hw_metrics_true_query:
            if hw_metric_true == "flops":
                results[hw_metric_true] = self.get_flops()
            elif hw_metric_true == "params":
                results[hw_metric_true] = self.get_params()
            elif hw_metric_true in ["float16_memory", "bfloat16_memory"]:
                results[hw_metric_true] = self.get_memory(hw_metric_true)
            else:
                raise ValueError("Invalid hw metric")
        if self.use_supernet_surrogate:
            results.update(self.eval_supernet_surrogate())
        return results


# test
if __name__ == "__main__":
    metrics = [
        "latencies",
        "energies",
        "flops",
        "params",
        "float16_memory",
        "bfloat16_memory",
    ]
    devices = [
        "a100",
        "rtx2080",
        "cpu_xeon_silver",
        "cpu_amd_7513",
        "h100",
        "a40",
        "rtx3080",
        "a6000",
        "P100",
        "v100",
        "cpu_xeon_gold",
        "cpu_amd_7502",
        "cpu_amd_7452",
    ]
    search_spaces_str = ["s", "m", "l"]
    for search_space in search_spaces_str:
        for device in devices:
            for metric in metrics:
                api = HWGPTBenchAPI(search_space)
                api.sample_random_arch()
                print(api.query(device=device, hw_metric=metric))

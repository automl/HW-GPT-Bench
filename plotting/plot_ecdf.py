import pickle
from lib.utils import (
    get_hw_predictor_surrogate,
    search_spaces,
    predict_hw_surrogate,
    choice_arch_config_map,
    metrics_map,
    dims_map,
    get_arch_feature_map,
    normalize_arch_feature_map,
)
from hwgpt.model.gpt.utils import sample_config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 16
plt.rcParams["figure.autolayout"] = True


class ECDFPlotterHW:
    def __init__(
        self,
        device="a40",
        metric="energies",
        type="median",
        surrogate="conformal_quantile",
        search_space="s",
        num_archs=50000,
    ):
        self.device = device
        self.metric = metric
        self.type = type
        self.exclude_keys = ["mlp_ratio_choices", "n_head_choices", "bias_choices"]
        self.search_space_str = search_space
        self.search_space = search_spaces[search_space]
        self.surrogate_type = surrogate
        max_layers = max(self.search_space["n_layer_choices"])
        self.surrogate = get_hw_predictor_surrogate(
            max_layers=max_layers,
            search_space=search_space,
            device=device,
            surrogate_type=surrogate,
            type=type,
            metric=metric,
        )
        self.sampled_archs = self.sample_archs(n=num_archs)
        print(len(self.sampled_archs))
        self.arch_results = self.compute_predictions()
        print(len(self.arch_results))
        self.stratified_results = self.stratify_by_dim()
        self.metric_label = metrics_map[metric]
        self.colors = ["r", "b", "g"]
        self.plot()

    def compute_predictions(self):
        predictions = []
        for arch in self.sampled_archs:
            arch_feature = get_arch_feature_map(arch, self.search_space_str)
            normalized_arch_feature_map = normalize_arch_feature_map(
                arch_feature, self.search_space_str
            )
            predictions_surrogate = predict_hw_surrogate(
                [normalized_arch_feature_map],
                self.surrogate,
                self.surrogate_type,
                return_all=True,
            )[0]
            for pred in predictions_surrogate:
                predictions.append((arch, pred))
        return predictions

    def stratify_by_dim(self):
        stratified_results = {}
        for dim in self.search_space.keys():
            if dim not in self.exclude_keys:
                stratified_results[dim] = {}
                for choice in self.search_space[dim]:
                    stratified_results[dim][choice] = []
                for result in self.arch_results:
                    sampled_dim = result[0][choice_arch_config_map[dim]]
                    # print(sampled_dim)
                    if self.metric == "energies":
                        stratified_results[dim][sampled_dim].append(result[1] * 1000)
                    else:
                        stratified_results[dim][sampled_dim].append(result[1])
        return stratified_results

    def plot(self):
        for dim in self.stratified_results:
            plt.figure(figsize=(8, 8))
            i = 0
            for choice in self.stratified_results[dim]:
                plt.grid(linestyle="--")
                plt.ecdf(
                    self.stratified_results[dim][choice],
                    label=choice,
                    color=self.colors[i],
                )
                i = i + 1
            save_path = (
                "ecdf_plots/"
                + self.metric
                + "_"
                + "ecdf_"
                + self.device
                + "_"
                + dim
                + "_"
                + self.surrogate_type
                + "_"
                + self.type
                + "_"
                + str(self.search_space_str)
                + ".pdf"
            )
            plt.legend(loc="upper left")
            plt.xlabel(self.metric_label)
            plt.ylabel("ECDF")
            plt.title("ECDF of " + self.metric_label + " " + dims_map[dim])
            plt.savefig(save_path)
            plt.clf()
            plt.close()

    def sample_archs(self, n=10000):
        sampled_archs = []
        for i in range(n):
            sampled_archs.append(sample_config(self.search_space, seed=i))
        return sampled_archs


types = ["median"]
models = ("conformal_quantile", "mlp", "quantile")
metrics = ("latencies", "energies", "float16_memory", "bfloat16_memory")
search_space_choices = ("m", "s", "l")
devices = (
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
)
for type in types:
    for model in models:
        for metric in metrics:
            for ss in search_space_choices:
                for device in devices:
                    if "memory" in metric and "quantile" in model:
                        continue
                    test_plot = ECDFPlotterHW(device, metric, type, model, ss)

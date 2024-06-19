import matplotlib.pyplot as plt

import numpy as np
from plotting.eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot
import pickle
from lib.utils import metrics_map
from hwgpt.api import HWGPTBenchAPI
from lib.utils_norm import (
    denormalize_energy,
    denormalize_latency,
    denormalize_ppl,
    get_max_min_true_metric,
)

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 16
# plt tight layout
plt.rcParams["figure.autolayout"] = True


class PlotEAF:
    def __init__(
        self,
        dim: int,
        result_base_path: str = "results_gpt_baselines_2d",
        search_space: str = "s",
    ):
        self.dim = dim
        self.api = HWGPTBenchAPI(search_space)
        self.result_base_path = result_base_path + "_" + search_space + "/"
        self.search_space = search_space
        self.methods = ["RS", "MOREA", "LS", "NSGA2", "LSBO", "RSBO", "MOASHA", "EHVI"]
        self.device_metrics = ("latencies", "energies")
        self.device_agnostic = ("float16_memory", "bfloat16_memory", "flops", "params")
        self.color_map = {
            "RS": "red",
            "MOREA": "blue",
            "LS": "green",
            "NSGA2": "yellow",
            "LSBO": "purple",
            "RSBO": "orange",
            "MOASHA": "pink",
            "EHVI": "cyan",
        }
        self.marker_map = {
            "RS": "o",
            "MOREA": "s",
            "LS": "D",
            "NSGA2": "X",
            "LSBO": "^",
            "RSBO": "v",
            "MOASHA": "<",
            "EHVI": ">",
        }
        self.linestyle_map = {
            "RS": "-",
            "MOREA": "--",
            "LS": "-.",
            "NSGA2": ":",
            "LSBO": "-",
            "RSBO": "--",
            "MOASHA": "-.",
            "EHVI": ":",
        }
        self.seeds = [9001, 9002, 9003, 9004]
        self.n_independent_runs = len(self.seeds)

    def get_baseline_results(
        self,
        method: str,
        hw_metric: str,
        surrogate_type: str,
        device: str,
        data_type: str,
        denormalize: bool = True,
    ):
        costs_all_seeds = []
        min_shape = 1000000000
        max_hw, min_hw = (
            get_max_min_true_metric(self.api, hw_metric)["max"],
            get_max_min_true_metric(self.api, hw_metric)["min"],
        )
        for seed in self.seeds:
            if hw_metric in self.device_metrics:
                results_path = (
                    f"{self.result_base_path}/{method}/{hw_metric}/mogpt"
                    + "_"
                    + str(device)
                    + "_"
                    + f"{seed}"
                    + ".pickle"
                )
            else:
                results_path = (
                    f"{self.result_base_path}/{method}/{hw_metric}/mogpt"
                    + "_"
                    + f"{seed}"
                    + ".pickle"
                )
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            perplexity = []
            for ppl in results["perplexity"]:
                if denormalize:
                    perplexity.append(
                        denormalize_ppl(min(ppl), self.search_space, method="max-min")
                    )
                else:
                    perplexity.append(min(ppl))
            hw_metric_arr = []
            for hw_metric_i in results["hw_metric"]:
                if hw_metric == "energies":
                    if denormalize:
                        hw_metric_arr.append(
                            denormalize_energy(
                                min(hw_metric_i),
                                device=device,
                                surrogate="",
                                data_type="",
                                scale=self.search_space,
                                metric=hw_metric,
                                method="max-min",
                            )
                        )
                    else:
                        hw_metric_arr.append(min(hw_metric_i))
                elif hw_metric == "latencies":
                    if denormalize:
                        hw_metric_arr.append(
                            denormalize_latency(
                                min(hw_metric_i),
                                device=device,
                                surrogate="",
                                data_type="",
                                scale=self.search_space,
                                metric=hw_metric,
                                method="max-min",
                            )
                        )
                    else:
                        hw_metric_arr.append(min(hw_metric_i))
                elif hw_metric == "flops":
                    if denormalize:
                        hw_metric_arr.append(
                            min(hw_metric_i) * (max_hw - min_hw) + min_hw
                        )
                    else:
                        hw_metric_arr.append(min(hw_metric_i))
            costs_all_seeds.append(
                np.transpose(
                    np.stack(
                        [
                            np.array(perplexity).reshape(-1, 1),
                            np.array(hw_metric_arr).reshape(-1, 1),
                        ]
                    ),
                    (2, 1, 0),
                )
            )
            if costs_all_seeds[-1].shape[1] < min_shape:
                min_shape = costs_all_seeds[-1].shape[1]
        costs_all_seeds = np.concatenate([c[:, :min_shape, :] for c in costs_all_seeds])
        costs_all_seeds = np.squeeze(costs_all_seeds)
        return costs_all_seeds

    def plot_eaf(
        self,
        methods: list,
        hw_metric: str,
        surrogate_type: str,
        device: str,
        data_type: str = "median",
    ):
        plt.rcParams["figure.figsize"] = (6, 6)
        _, ax = plt.subplots()
        for method in methods:
            costs = self.get_baseline_results(
                method, hw_metric, surrogate_type, device, data_type
            )
            levels = [
                self.n_independent_runs // 4,
                self.n_independent_runs // 2,
                3 * self.n_independent_runs // 4,
            ]
            surfs = get_empirical_attainment_surface(costs=costs, levels=levels)

            eaf_plot = EmpiricalAttainmentFuncPlot()
            eaf_plot.plot_surface_with_band(
                ax,
                color=self.color_map[method],
                label=method,
                surfs=surfs,
                linestyle=self.linestyle_map[method],
                marker=self.marker_map[method],
            )
            plt.xlabel("Perplexity")
            plt.ylabel(metrics_map[hw_metric])
            plt.title("EAF for FLOPS on GPT-" + self.search_space.upper())
            plt.tight_layout()
        plt.savefig(f"eaf_plots/{hw_metric}_{self.search_space}.pdf")
        plt.clf()

    def plot_hv_over_time(
        self,
        methods: list,
        hw_metric: str,
        surrogate_type: str,
        device: str,
        data_type: str = "median",
    ):
        plt.rcParams["figure.figsize"] = (6, 6)
        _, ax = plt.subplots()
        max_hv = 0
        min_hv = 1000000000
        for method in methods:
            costs = self.get_baseline_results(
                method, hw_metric, surrogate_type, device, data_type, denormalize=False
            )
            costs = costs[np.newaxis, ...]
            ref_point = np.array([1.5, 1.5])
            eaf_plot = EmpiricalAttainmentFuncPlot(ref_point=ref_point)
            colors = [self.color_map[method]]
            labels = [method]
            _, max_hv_curr, min_hv_curr = (
                eaf_plot.plot_multiple_hypervolume2d_with_band(
                    ax, costs_array=costs, colors=colors, labels=labels, normalize=False
                )
            )
            if max_hv_curr > max_hv:
                max_hv = max_hv_curr
            if min_hv_curr < min_hv:
                min_hv = min_hv_curr
            plt.legend(prop={"size": 14}, loc="lower right", ncol=2)
            plt.title("HV over Time for FLOPS on " + self.search_space.upper())
            plt.xlabel("Number of Surrogate Evaluations")
            plt.ylabel("Hypervolume")
            plt.tight_layout()

        plt.ylim(min_hv + 0.8, max_hv)
        plt.savefig(
            f"eaf_plots/hv_over_time_{hw_metric}_{self.search_space}.pdf",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    plot_eaf = PlotEAF(dim=2, search_space="m")
    plot_eaf.plot_eaf(
        ["LS", "LSBO", "EHVI", "RSBO", "NSGA2", "MOASHA"],
        "flops",
        "conformal_quantile",
        "rtx2080",
        data_type="quantile",
    )

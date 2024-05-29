import matplotlib.pyplot as plt

import numpy as np
from plotting.eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot
import pickle
from lib.utils import metrics_map

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 16
# plt tight layout
plt.rcParams["figure.autolayout"] = True


class PlotEAF:
    def __init__(
        self,
        dim: int,
        result_base_path: str = "results_gpt_baselines_2d/",
        search_space: str = "s",
    ):
        self.dim = dim
        self.result_base_path = result_base_path
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
        self.seeds = [1, 2, 3, 4]
        self.n_independent_runs = len(self.seeds)

    def get_baseline_results(
        self,
        method: str,
        hw_metric: str,
        surrogate_type: str,
        device: str,
        data_type: str,
    ):
        costs_all_seeds = []
        for seed in self.seeds:
            if hw_metric in self.device_metrics:
                results_path = (
                    f"{self.result_base_path}/"
                    + "mogpt_"
                    + f"{method}"
                    + "_"
                    + f"{device}"
                    + "_"
                    + f"{self.search_space}"
                    + "_"
                    + f"{hw_metric}"
                    + "_"
                    + f"{surrogate_type}"
                    + "_"
                    + f"{data_type}"
                    + "_"
                    + f"{seed}"
                    + ".pickle"
                )
            else:
                results_path = (
                    f"{self.result_base_path}/{method}/{hw_metric}"
                    + "_"
                    + f"{seed}"
                    + ".pickle"
                )
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            perplexity = results["perplexity"]
            hw_metric_arr = results["hw_metric"]
            # print(perplexity, hw_metric_arr)
            costs_all_seeds.append([[perplexity], [hw_metric_arr]])
        costs_all_seeds = np.array(costs_all_seeds)
        costs_all_seeds = np.squeeze(costs_all_seeds)
        costs_all_seeds = np.transpose(costs_all_seeds, (0, 2, 1))
        return costs_all_seeds

    def plot_eaf(
        self,
        methods: list,
        hw_metric: str,
        surrogate_type: str,
        device: str,
        data_type: str = "median",
    ):
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

            # print(costs.shape)
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

            # plt.tight_layout()
            # plt.savefig(f"eaf_plots/{method}_{hw_metric}_{surrogate_type}_{device}_{data_type}.pdf")

    def plot_hv_over_time(
        self,
        methods: list,
        hw_metric: str,
        surrogate_type: str,
        device: str,
        data_type: str = "median",
    ):
        _, ax = plt.subplots()
        for method in methods:
            costs = self.get_baseline_results(
                method, hw_metric, surrogate_type, device, data_type
            )
            costs = costs[np.newaxis, ...]
            ref_point = np.array([1, 1])
            eaf_plot = EmpiricalAttainmentFuncPlot(ref_point=ref_point)
            colors = [self.color_map[method]]
            labels = [method]
            eaf_plot.plot_multiple_hypervolume2d_with_band(
                ax, costs_array=costs, colors=colors, labels=labels, normalize=False
            )
            plt.legend()
            plt.xlabel("Number of Surrogate Evaluations")


if __name__ == "__main__":
    plot_eaf = PlotEAF(dim=2)
    plot_eaf.plot_eaf(
        ["RS", "EHVI"], "energies", "conformal_quantile", "v100", data_type="quantile"
    )
    # plot_eaf.plot_eaf("EHVI", "energies", "conformal_quantile", "v100", data_type="quantile")
    plt.legend()
    plt.savefig("eaf_plots/RS_EHVI_energies_conformal_quantile_v100_quantile.pdf")
    plt.clf()
    plot_eaf.plot_hv_over_time(
        ["RS", "EHVI"], "energies", "conformal_quantile", "v100", data_type="quantile"
    )
    # plot_eaf.plot_hv_over_time("EHVI", "energies", "conformal_quantile", "v100", data_type="quantile")
    plt.legend()
    plt.savefig("eaf_plots/RS_EHVI_energies_conformal_quantile_v100_quantile_hv.pdf")

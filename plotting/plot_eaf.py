import matplotlib.pyplot as plt

import numpy as np
from plotting.eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot
import pickle
from lib.utils import metrics_map
from lib.utils_norm import (
    denormalize_energy,
    denormalize_latency,
    denormalize_ppl,
    denormalize_memory,
)

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 14
# plt tight layout
plt.rcParams["figure.autolayout"] = True
# se t size
plt.rcParams["figure.figsize"] = (6, 6)


# plt.rcParams['bbox_inches'] = 'tight'
class PlotEAF:
    def __init__(
        self,
        dim: int,
        result_base_path: str = "results_gpt_baselines_2d",
        search_space: str = "s",
    ):
        self.dim = dim
        self.result_base_path = result_base_path + "_" + search_space + "_log/"
        self.search_space = search_space
        if self.search_space == "s":
            self.mul_factor = 0.5
        elif self.search_space == "m":
            self.mul_factor = 1
        else:
            self.mul_factor = 4
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
                            * self.mul_factor
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
                            * self.mul_factor
                        )
                    else:
                        hw_metric_arr.append(min(hw_metric_i))
                else:
                    if denormalize:
                        hw_metric_arr.append(
                            denormalize_memory(
                                min(hw_metric_i),
                                scale=self.search_space,
                                metric=hw_metric,
                            )
                            * self.mul_factor
                        )
                    else:
                        hw_metric_arr.append(min(hw_metric_i))
                    # hw_metric_arr.append(denormalize_memory(min(hw_metric_i),scale=self.search_space,metric=hw_metric))
            costs = np.transpose(
                np.stack(
                    [
                        np.array(perplexity).reshape(-1, 1),
                        np.array(hw_metric_arr).reshape(-1, 1),
                    ]
                ),
                (2, 1, 0),
            )
            # costs_en>0
            costs = costs[costs[:, :, -1] > 0].reshape(1, -1, 2)
            costs_all_seeds.append(costs)
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

            plt.tight_layout()
            plt.legend()
        if hw_metric == "energies":
            plt.title("EAF for Energy on " + device.upper())
        else:
            plt.title("EAF for Latencies on " + device.upper())
        plt.tight_layout()
        plt.savefig(
            f"eaf_plots_latencies_low_res/{hw_metric}_{device}_{self.search_space}.pdf",
            bbox_inches="tight",
            dpi=2,
        )
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
            # costs = costs[:,:1000,:]
            # costs[:,:,0]/=37
            # costs[:,:,1]/=17
            costs = costs[np.newaxis, ...]
            ref_point = np.array([2, 2])
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
            if hw_metric == "energies":
                plt.title("HV over Time for Energy on " + device.upper())
                plt.xlabel("Number of Surrogate Evaluations")
                plt.ylabel("Hypervolume")
            else:
                plt.title("HV over Time for Latencies on " + device.upper())
                plt.xlabel("Number of Surrogate Evaluations")
                plt.ylabel("Hypervolume")
            plt.tight_layout()
        plt.tight_layout()
        plt.ylim(min_hv + 0.8, max_hv)
        plt.savefig(
            f"eaf_plots_latencies_low_res/hv_over_time_{hw_metric}_{device}_{self.search_space}.pdf",
            bbox_inches="tight",
            dpi=2,
        )


if __name__ == "__main__":
    scales = ["s"]
    devices = [
        "a100",
        "a6000",
        "v100",
        "rtx2080",
        "rtx3080",
        "P100",
        "cpu_xeon_gold",
        "cpu_xeon_silver",
        "h100",
        "a40",
        "cpu_amd_7513",
        "cpu_amd_7502",
        "cpu_amd_7452",
    ]
    for d in devices:
        for s in scales:
            plot_eaf = PlotEAF(dim=2, search_space=s)
            plot_eaf.plot_eaf(
                ["RS", "LS", "LSBO", "MOREA", "EHVI", "RSBO", "NSGA2", "MOASHA"],
                "latencies",
                "conformal_quantile",
                d,
                data_type="quantile",
            )
            plot_eaf.plot_hv_over_time(
                ["RS", "LS", "LSBO", "MOREA", "EHVI", "RSBO", "NSGA2", "MOASHA"],
                "latencies",
                "conformal_quantile",
                d,
                data_type="quantile",
            )

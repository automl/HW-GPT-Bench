import matplotlib.pyplot as plt

import numpy as np
from plotting.eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot
import pickle
from lib.utils import metrics_map

class PlotEAF:
    def __init__(self, dim:int, n_samples:int, n_independent_runs:int, result_base_path:str="results_gpt_baselines_2d/"):
        self.dim = dim
        self.n_samples = n_samples
        self.n_independent_runs = n_independent_runs
        self.result_base_path = result_base_path
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
            "EHVI": "cyan"
        }
        self.marker_map = {
            "RS": "o",
            "MOREA": "s",
            "LS": "D",
            "NSGA2": "X",
            "LSBO": "^",
            "RSBO": "v",
            "MOASHA": "<",
            "EHVI": ">"
        }
        self.linestyle_map = {
            "RS": "-",
            "MOREA": "--",
            "LS": "-.",
            "NSGA2": ":",
            "LSBO": "-",
            "RSBO": "--",
            "MOASHA": "-.",
            "EHVI": ":"
        }
        self.seeds = [9001, 9002, 9003, 9004, 9005]

    def get_baseline_results(self, method:str, hw_metric:str, surrogate_type:str, device:str, data_type:str):
      costs_all_seeds = []
      for seed in self.seeds:
        if hw_metric in self.device_metrics:
            results_path = f"{self.result_base_path}/{method}/{hw_metric}"+"_"+f"{surrogate_type}"+"_"+f"{data_type}"+"_"+f"{device}"+"_"+f"{seed}"+".pickle"
        else:
            results_path = f"{self.result_base_path}/{method}/{hw_metric}"+"_"+f"{seed}"+".pickle"
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        perplexity = results["perplexity"]
        hw_metric = [results["hw_metric"][i] for i in results["hw_metric"].argmin(axis=-1)]
        costs_all_seeds.append([perplexity, hw_metric])
      costs_all_seeds = np.array(costs_all_seeds)
      return costs_all_seeds
      
    def plot_eaf(self, method:str, hw_metric:str, surrogate_type:str, device:str, data_type:str="median"):
        costs = self.get_baseline_results(method, hw_metric, surrogate_type, device, data_type)
        levels = [self.n_independent_runs // 4, self.n_independent_runs // 2, 3 * self.n_independent_runs // 4]
        surfs = get_empirical_attainment_surface(costs=costs, levels=levels)
        _, ax = plt.subplots()
        eaf_plot = EmpiricalAttainmentFuncPlot()
        eaf_plot.plot_surface_with_band(ax, color=self.color_map[method], label=method, surfs=surfs,linestyle=self.linestyle_map[method],marker=self.marker_map[method])
        plt.xlabel("Perplexity")
        plt.ylabel(metrics_map[hw_metric])
        plt.legend()
        #plt.tight_layout()
        #plt.savefig(f"eaf_plots/{method}_{hw_metric}_{surrogate_type}_{device}_{data_type}.pdf")

    def plot_hv_over_time(self, method:str, hw_metric:str, surrogate_type:str, device:str, data_type:str="median"):
        _, ax = plt.subplots()
        costs = self.get_baseline_results(method, hw_metric, surrogate_type, device, data_type)
        ref_point = np.array([1,1])
        eaf_plot = EmpiricalAttainmentFuncPlot(ref_point=ref_point)
        colors = [self.color_map[method]]
        labels = [method]
        eaf_plot.plot_multiple_hypervolume2d_with_band(ax ,costs_array=costs ,colors=colors ,labels=labels ,normalize =False)
        plt.legend()
        plt.xlabel("Number of Surrogate Evaluations")

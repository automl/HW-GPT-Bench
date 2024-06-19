import pickle
import numpy as np
import matplotlib.pyplot as plt
from hwgpt.model.gpt.utils import sample_config
from lib.utils import search_spaces
from hwgpt.api import HWGPTBenchAPI
import torch
from plotting.eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 16
# plt tight layout
plt.rcParams["figure.autolayout"] = True

def get_pareto_optimal(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self
    return is_pareto

class CorrScatter:
    def __init__(self, search_space=str, num_archs=int):
        self.search_space = search_space
        self.num_archs = num_archs
        self.sampled_archs = self.sample_archs()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.api = HWGPTBenchAPI(search_space=self.search_space)
        self.gt_stats = self.api.gt_stats
        print("API initialized")

    def sample_archs(self):
        sampled_archs = []
        for i in range(self.num_archs):
            arch = sample_config(search_spaces[self.search_space], seed=i)
            sampled_archs.append(arch)
        return sampled_archs

    def get_perplexity_list(self, hwmetric):
        perplexity = []
        for arch in self.gt_stats.keys():
            if hwmetric=="flops" or hwmetric=="params" or hwmetric=="float16_memory" or hwmetric=="bfloat16_memory":
                perplexity.append(self.gt_stats[arch]["perplexity"])
            else:
                for i in range(10):
                    perplexity.append(self.gt_stats[arch]["perplexity"])

        return perplexity
    

    def get_flops_list(self):
        flops = []
        for arch in self.gt_stats.keys():
            flops.append(self.gt_stats[arch]["flops"])

        return flops

    def get_params_list(self):
        params = []
        for arch in self.gt_stats.keys():
            params.append(self.gt_stats[arch]["params"])

        return params

    def get_float16_memory(self):
        float16_memory = []
        for arch in self.gt_stats.keys():
            float16_memory.append(self.gt_stats[arch]["float16_memory"])
        return float16_memory

    def get_bfloat16_memory(self):
        bfloat16_memory = []
        for arch in self.gt_stats.keys():
            bfloat16_memory.append(self.gt_stats[arch]["bfloat16_memory"])
        return bfloat16_memory
    
    def remove_outliers(self, hw_metric: str, device: str, arch: str):
        hw_stats = np.array(self.gt_stats[arch][device][hw_metric])
        # remove outliers
        hw_stats = hw_stats[hw_stats < np.percentile(hw_stats, 99)]
        return hw_stats
    
    def get_lat_en_random(self, hw_metric: str, device: str):
        lat_en = []
        for arch in self.gt_stats.keys():
            # remove outliers
            np.random.seed(np.random.randint(0, 1000))
            #if hw_metric == "energies":
            #    hw_stats = self.remove_outliers(hw_metric, device, arch)#[0:10]
            #else:
            hw_stats = self.gt_stats[arch][device][hw_metric]#[0:10]
            if hw_metric == "energies":
                lat_en.append(np.random.choice(hw_stats)*1000)
            else:
                lat_en.append(np.random.choice(hw_stats))
        return lat_en
    
    def get_lat_en_worst(self, hw_metric: str, device: str):
        lat_en = []
        for arch in self.gt_stats.keys():
            # remove outliers
            if hw_metric == "energies":
                hw_stats = self.remove_outliers(hw_metric, device, arch)[0:10]
            else:
               hw_stats = self.gt_stats[arch][device][hw_metric][0:10]
            if hw_metric == "energies":
                lat_en.append(np.max(hw_stats)*1000)
            else:
                lat_en.append(np.max(hw_stats))
        return lat_en
    
    def get_lat_en(
        self, hw_metric: str, device: str, 
    ):
        lat_en = []
        for arch in self.gt_stats.keys():
            # remove outliers
            if hw_metric == "energies":
               hw_stats = self.remove_outliers(hw_metric, device, arch)
            else:
               hw_stats = self.gt_stats[arch][device][hw_metric]

            for lat in hw_stats[0:10]:
                if hw_metric == "energies":
                    lat_en.append(lat*1000)
                else:
                    lat_en.append(lat)
        
        return lat_en
    def sample_random_for_arch(self, arch_list: list, hw_metric: str, device: str):
        lat_en = []
        for arch in arch_list:
            # remove outliers
            if hw_metric == "energies":
                hw_stats = self.remove_outliers(hw_metric, device, arch)
            else:
                hw_stats = self.gt_stats[arch][device][hw_metric]
            lat_en.append(np.random.choice(hw_stats))
        return lat_en
    
    def plot_corr_scatter(
        self, metrics_list: list, device: str, ppl_dependency:str
    ):
        metrics_dict = {}
        metrics_worst = {}
        metrics_random = {}
        self.n_independent_runs = 10
        levels = [
                1 , self.n_independent_runs // 2, self.n_independent_runs
            ]
        assert len(metrics_list) == 3
        plt.figure(figsize=(15, 5))
        for metric in metrics_list:
            if metric == "perplexity":
                metrics_dict[metric] = self.get_perplexity_list(ppl_dependency)
                metrics_worst[metric] = self.get_perplexity_list("flops")
                metrics_random[metric] = [self.get_perplexity_list("params") for i in range(10)]

            elif metric == "flops":
                metrics_dict[metric] = self.get_flops_list()
            elif metric == "params":
                metrics_dict[metric] = self.get_params_list()
            elif metric == "float16_memory":
                metrics_dict[metric] = self.get_float16_memory()
            elif metric == "bfloat16_memory":
                metrics_dict[metric] = self.get_bfloat16_memory()
            else:
                metrics_dict[metric] = self.get_lat_en(metric, device)
                metrics_worst[metric] = self.get_lat_en_worst(metric, device)
                metrics_random[metric] = [self.get_lat_en_random(metric, device) for i in range(10)]
        plt.subplot(1, 3, 1)
        metrics_dict_subset = {}
        for k in metrics_dict.keys():
            metrics_dict_subset[k] = metrics_dict[k][0:5000]
        sc = plt.scatter(
            metrics_dict_subset[metrics_list[0]],
            metrics_dict_subset[metrics_list[1]],
            c=metrics_dict_subset[metrics_list[2]],
            cmap="viridis",
            s=4,
        )
        # plot pareto front in red
        costs = np.array([metrics_dict[metrics_list[0]], metrics_dict[metrics_list[1]]]).T
        is_pareto = get_pareto_optimal(costs)
        plt.scatter(
            np.array(metrics_dict[metrics_list[0]])[is_pareto],
            np.array(metrics_dict[metrics_list[1]])[is_pareto],
            c="red",
            s=4,
        )
        #random pareto
        costs = np.array([metrics_random[metrics_list[0]], metrics_random[metrics_list[1]]]).T
        costs = np.transpose(costs, (1, 0, 2))
        surfs = get_empirical_attainment_surface(costs=np.squeeze(costs), levels=levels)
        eaf_plot = EmpiricalAttainmentFuncPlot()
        eaf_plot.plot_surface_with_band(
            ax=plt.gca(),
            surfs=surfs,
            label="median pareto",
            color="blue",
            alpha=0.6
        )
        #worst pareto
        costs = np.array([metrics_worst[metrics_list[0]], metrics_worst[metrics_list[1]]]).T
        is_pareto = get_pareto_optimal(costs)
        plt.scatter(
            np.array(metrics_worst[metrics_list[0]])[is_pareto],
            np.array(metrics_worst[metrics_list[1]])[is_pareto],
            c="black",
            s=4,
        )
        # order and line plot
        #plt.plot(
        #    np.sort(metrics_worst[metrics_list[0]][is_pareto]),
        #    np.sort(metrics_worst[metrics_list[1]][is_pareto]),
        #    c="black",
        #)
        #
        plt.colorbar(sc, label=metrics_list[2])
        plt.xlabel(metrics_list[0])
        plt.ylabel(metrics_list[1])
        # randomly samply only 1000 architectures

        plt.subplot(1, 3, 2)
        sc = plt.scatter(
            metrics_dict_subset[metrics_list[0]],
            metrics_dict_subset[metrics_list[2]],
            c=metrics_dict_subset[metrics_list[1]],
            cmap="viridis",
            s=4,
        )
        costs = np.array([metrics_dict[metrics_list[0]], metrics_dict[metrics_list[2]]]).T
        is_pareto = get_pareto_optimal(costs)
        plt.scatter(
            np.array(metrics_dict[metrics_list[0]])[is_pareto],
            np.array(metrics_dict[metrics_list[2]])[is_pareto],
            c="red",
            s=4,
        )
        costs = np.array([metrics_random[metrics_list[0]], metrics_random[metrics_list[2]]]).T
        costs = np.transpose(costs, (1, 0, 2))
        surfs = get_empirical_attainment_surface(np.squeeze(costs), levels)
        eaf_plot = EmpiricalAttainmentFuncPlot()
        eaf_plot.plot_surface_with_band(
            ax=plt.gca(),
            surfs=surfs,
            label = "median pareto",
            color = "blue",
            alpha=0.6
        )
        #worst pareto
        costs = np.array([metrics_worst[metrics_list[0]], metrics_worst[metrics_list[2]]]).T
        is_pareto = get_pareto_optimal(costs)
        plt.scatter(
            np.array(metrics_worst[metrics_list[0]])[is_pareto],
            np.array(metrics_worst[metrics_list[2]])[is_pareto],
            c="black",
            s=4,
        )
        #plt.plot(
        #    np.sort(metrics_worst[metrics_list[0]][is_pareto]),
        #    np.sort(metrics_worst[metrics_list[2]][is_pareto]),
        #    c="black",
        #)
        plt.colorbar(sc, label=metrics_list[1])
        plt.xlabel(metrics_list[0])
        plt.ylabel(metrics_list[2])
        plt.subplot(1, 3, 3)
        sc = plt.scatter(
            metrics_dict_subset[metrics_list[1]],
            metrics_dict_subset[metrics_list[2]],
            c=metrics_dict_subset[metrics_list[0]],
            cmap="viridis",
            s=4,
        )
        costs = np.array([metrics_dict[metrics_list[1]], metrics_dict[metrics_list[2]]]).T
        is_pareto = get_pareto_optimal(costs)
        plt.scatter(
            np.array(metrics_dict[metrics_list[1]])[is_pareto],
            np.array(metrics_dict[metrics_list[2]])[is_pareto],
            c="red",
            s=4,
            label="Pareto Front",
        )
        costs = np.array([metrics_worst[metrics_list[1]], metrics_worst[metrics_list[2]]]).T
        is_pareto = get_pareto_optimal(costs)
        plt.scatter(
            np.array(metrics_worst[metrics_list[1]])[is_pareto],
            np.array(metrics_worst[metrics_list[2]])[is_pareto],
            c="black",
            s=4,
        )
        plt.colorbar(sc, label=metrics_list[0])
        plt.xlabel(metrics_list[1])
        plt.ylabel(metrics_list[2])

        plt.suptitle(
            "Trade-offs between "
            + metrics_list[0]
            + ", "
            + metrics_list[1]
            + " and "
            + metrics_list[2]
            + " on "
            + device.upper()
        )
        plt.savefig(
            "corr_scatter_plots/"
            + metrics_list[0]
            + "_"
            + metrics_list[1]
            + "_"
            + metrics_list[2]
            + "_"
            + device.upper()
            + ".pdf"
        )
        # clear subplots


        

        plt.clf()
        plt.close()


if __name__ == "__main__":
    corr_scatter = CorrScatter("s", 10000)
    corr_scatter.plot_corr_scatter(
        ["perplexity", "energies", "latencies"], "rtx2080", "energies"
    )
import pickle
import numpy as np
import matplotlib.pyplot as plt
from hwgpt.model.gpt.utils import sample_config
from lib.utils import search_spaces
from hwgpt.api import HWGPTBenchAPI
import torch

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 16
# plt tight layout
plt.rcParams["figure.autolayout"] = True


class CorrScatter:
    def __init__(self, search_space=str, num_archs=int):
        self.search_space = search_space
        self.num_archs = num_archs
        self.sampled_archs = self.sample_archs()
        print(len(self.sampled_archs))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.api = HWGPTBenchAPI(search_space=self.search_space)
        print("API initialized")

    def sample_archs(self):
        sampled_archs = []
        for i in range(self.num_archs):
            arch = sample_config(search_spaces[self.search_space], seed=i)
            sampled_archs.append(arch)
        return sampled_archs

    def get_perplexity_list(self):
        ppl = []
        for arch in self.sampled_archs:
            self.api.set_arch(arch)
            ppl.append(self.api.compute_predictions_ppl())
        return ppl

    def get_flops_list(self):
        flops = []
        for arch in self.sampled_archs:
            self.api.set_arch(arch)
            flops.append(self.api.get_flops())

        return flops

    def get_params_list(self):
        params = []
        for arch in self.sampled_archs:
            self.api.set_arch(arch)
            params.append(self.api.get_params())

        return params

    def get_float16_memory(self):
        float16_memory = []
        for arch in self.sampled_archs:
            self.api.set_arch(arch)
            float16_memory.append(
                self.api.compute_predictions_hw(
                    hw_metric="float16_memory",
                    device="rtx2080",
                    surrogate_type="mlp",
                    data_type="median",
                ).item()
            )
        return float16_memory

    def get_bfloat16_memory(self):
        bfloat16_memory = []
        for arch in self.sampled_archs:
            self.api.set_arch(arch)
            bfloat16_memory.append(
                self.api.compute_predictions_hw(
                    hw_metric="bfloat16_memory",
                    device="a100",
                    surrogate_type="mlp",
                    data_type="median",
                ).item()
            )
        return bfloat16_memory

    def get_lat_en(
        self, hw_metric: str, device: str, surrogate_type: str, data_type: str
    ):
        lat_en = []
        for arch in self.sampled_archs:
            self.api.set_arch(arch)

            predictions = self.api.compute_predictions_hw(
                hw_metric, device, surrogate_type, data_type
            )
            if "quantile" in surrogate_type:
                for prediction in predictions:
                    if hw_metric == "energies":
                        prediction = prediction * 1000
                    lat_en.append(prediction.item())
            else:
                if hw_metric == "energies":
                    predictions = predictions * 1000
                lat_en.append(predictions.item())
        return lat_en

    def plot_corr_scatter(
        self, metrics_list: list, device: str, surrogate_type: str, data_type: str
    ):
        metrics_dict = {}
        repeat_for_quantiles = "quantile" in surrogate_type
        assert len(metrics_list) == 3
        plt.figure(figsize=(15, 5))
        for i in range(len(metrics_list)):
            metric = metrics_list[i]
            if metric == "perplexity":
                y = self.get_perplexity_list()
                # repeat each entry 9 quantiles times
                if repeat_for_quantiles:
                    y = np.repeat(y, 9)
            elif metric == "flops":
                y = self.get_flops_list()
                if repeat_for_quantiles:
                    y = np.repeat(y, 9)
            elif metric == "params":
                y = self.get_params_list()
                if repeat_for_quantiles:
                    y = np.repeat(y, 9)
            elif metric == "float16_memory":
                y = self.get_float16_memory()
                if repeat_for_quantiles:
                    y = np.repeat(y, 9)
            elif metric == "bfloat16_memory":
                y = self.get_bfloat16_memory()
                if repeat_for_quantiles:
                    y = np.repeat(y, 9)
            else:
                y = self.get_lat_en(metric, device, surrogate_type, data_type)
            metrics_dict[metric] = y
            print("Processed", metric)
        plt.subplot(1, 3, 1)
        sc = plt.scatter(
            metrics_dict[metrics_list[0]],
            metrics_dict[metrics_list[1]],
            c=metrics_dict[metrics_list[2]],
            cmap="viridis",
            s=4,
        )
        plt.colorbar(sc, label=metrics_list[2])
        plt.xlabel(metrics_list[0])
        plt.ylabel(metrics_list[1])
        plt.subplot(1, 3, 2)
        sc = plt.scatter(
            metrics_dict[metrics_list[0]],
            metrics_dict[metrics_list[2]],
            c=metrics_dict[metrics_list[1]],
            cmap="viridis",
            s=4,
        )
        plt.colorbar(sc, label=metrics_list[1])
        plt.xlabel(metrics_list[0])
        plt.ylabel(metrics_list[2])
        plt.subplot(1, 3, 3)
        sc = plt.scatter(
            metrics_dict[metrics_list[1]],
            metrics_dict[metrics_list[2]],
            c=metrics_dict[metrics_list[0]],
            cmap="viridis",
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
        )
        plt.savefig(
            "corr_scatter_plots/"
            + metrics_list[0]
            + "_"
            + metrics_list[1]
            + "_"
            + metrics_list[2]
            + "_"
            + device
            + "_"
            + surrogate_type
            + "_"
            + data_type
            + ".pdf"
        )
        # clear subplots

        plt.clf()
        plt.close()


if __name__ == "__main__":
    corr_scatter = CorrScatter("s", 10000)
    corr_scatter.plot_corr_scatter(
        ["perplexity", "energies", "float16_memory"], "rtx2080", "mlp", "quantile"
    )
    corr_scatter.plot_corr_scatter(
        ["perplexity", "latencies", "bfloat16_memory"], "a100", "mlp", "quantile"
    )
    corr_scatter.plot_corr_scatter(
        ["perplexity", "energies", "latencies"], "a100", "mlp", "quantile"
    )
    corr_scatter.plot_corr_scatter(
        ["perplexity", "latencies", "energies"],
        "rtx2080",
        "conformal_quantile",
        "quantile",
    )

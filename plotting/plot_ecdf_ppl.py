from lib.utils import (
    get_ppl_predictor_surrogate,
    search_spaces,
    choice_arch_config_map,
    metrics_map,
    dims_map,
    convert_config_to_one_hot
)
from hwgpt.model.gpt.utils import sample_config
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 16
plt.rcParams["figure.autolayout"] = True


class ECDFPlotterPPL:
    def __init__(
        self,
        metric="perplexity",
        search_space="s",
        num_archs=50000,
    ):
        self.metric = metric
        self.exclude_keys = ["mlp_ratio_choices", "n_head_choices", "bias_choices"]
        self.search_space_str = search_space
        self.search_space = search_spaces[search_space]
        self.surrogate = get_ppl_predictor_surrogate(search_space)
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
            arch_feature = convert_config_to_one_hot(arch, search_space=self.search_space_str)
            predictions_surrogate =self.surrogate(arch_feature.cuda().unsqueeze(0))
            predictions.append((arch, predictions_surrogate.item()))
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
                + str(self.search_space_str)
                +"_"
                +str(dim)
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


for scale in ["s","m","l"]:
    plot = ECDFPlotterPPL(search_space=scale)

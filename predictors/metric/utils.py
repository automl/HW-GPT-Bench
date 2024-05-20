import torch
import pickle

device_to_path_map_s = {
    "a100": "arch_stats/metric/a100_s/",
    "v100": "arch_stats/metric/v100_s/",
    "rtx2080": "arch_stats/metric/rtx2080_s/",
    "rtx3080": "arch_stats/metric/rtx3080_s/",
    "a6000": "arch_stats/metric/a6000_s/",
    "a40": "arch_stats/metric/a40_s/",
    "p100": "arch_stats/metric/p100_s/",
    "amd": "arch_stats/metric/amd_s/",
    "cpu_meta": "arch_stats/metric/cpu_meta_s/",
    "cpu_mlgpu": "arch_stats/metric/cpu_mlgpu_s/",
    "cpu_alldlc": "arch_stats/metric/cpu_alldlc_s/",
    "cpu_a6000": "arch_stats/metric/cpu_a6000_s/",
    "cpu_p100": "arch_stats/metric/cpu_p100_s/",
}

device_to_path_map_m = {
    "a100": "arch_stats/metric/a100_m/",
    "v100": "arch_stats/metric/v100_m/",
    "rtx2080": "arch_stats/metric/rtx2080_m/",
    "rtx3080": "arch_stats/metric/rtx3080_m/",
    "a6000": "arch_stats/metric/a6000_m/",
    "a40": "arch_stats/metric/a40_m/",
    "p100": "arch_stats/metric/p100_m/",
    "amd": "arch_stats/metric/amd_m/",
    "cpu_meta": "arch_stats/metric/cpu_meta_m/",
    "cpu_mlgpu": "arch_stats/metric/cpu_mlgpu_m/",
    "cpu_alldlc": "arch_stats/metric/cpu_alldlc_m/",
    "cpu_a6000": "arch_stats/metric/cpu_a6000_m/",
    "cpu_p100": "arch_stats/metric/cpu_p100_m/",
}

device_to_path_map_l = {
    "a100": "arch_stats/metric/a100_l/",
    "v100": "arch_stats/metric/v100_l/",
    "rtx2080": "arch_stats/metric/rtx2080_l/",
    "rtx3080": "arch_stats/metric/rtx3080_l/",
    "a6000": "arch_stats/metric/a6000_l/",
    "a40": "arch_stats/metric/a40_l/",
    "p100": "arch_stats/metric/p100_l/",
    "amd": "arch_stats/metric/amd_l/",
    "cpu_meta": "arch_stats/metric/cpu_meta_l/",
    "cpu_mlgpu": "arch_stats/metric/cpu_mlgpu_l/",
    "cpu_alldlc": "arch_stats/metric/cpu_alldlc_l/",
    "cpu_a6000": "arch_stats/metric/cpu_a6000_l/",
    "cpu_p100": "arch_stats/metric/cpu_p100_l/",
}

search_spaces = {
    "s": {
        "embed_dim_choices": [192, 384, 768],
        "n_layer_choices": [10, 11, 12],
        "mlp_ratio_choices": [2, 3, 4],
        "n_head_choices": [4, 8, 12],
        "bias_choices": [True, False],
    },
    "m": {
        "embed_dim_choices": [256, 512, 1024],
        "n_layer_choices": [22, 23, 24],
        "mlp_ratio_choices": [2, 3, 4],
        "n_head_choices": [8, 12, 16],
        "bias_choices": [True, False],
    },
    "l": {
        "embed_dim_choices": [320, 640, 1280],
        "n_layer_choices": [34, 35, 36],
        "mlp_ratio_choices": [2, 3, 4],
        "n_head_choices": [8, 16, 20],
        "bias_choices": [True, False],
    },
}


def convert_config_to_one_hot(config, choices_dict):
    one_hot_embed = torch.zeros(len(choices_dict["embed_dim_choices"]))
    one_hot_layer = torch.zeros(len(choices_dict["n_layer_choices"]))
    max_layers = max(choices_dict["n_layer_choices"])
    one_hot_mlp = torch.zeros(max_layers, len(choices_dict["mlp_ratio_choices"]))
    one_hot_head = torch.zeros(max_layers, len(choices_dict["n_head_choices"]))
    one_hot_bias = torch.zeros(len(choices_dict["bias_choices"]))
    # get selected index for embed dim
    embed_idx = choices_dict["embed_dim_choices"].index(config["sample_embed_dim"])
    one_hot_embed[embed_idx] = 1
    # get selected index for layer
    layer_idx = choices_dict["n_layer_choices"].index(config["sample_n_layer"])
    one_hot_layer[layer_idx] = 1
    # get selected index for mlp ratio and head
    for i in range(config["sample_n_layer"]):
        mlp_idx = choices_dict["mlp_ratio_choices"].index(config["sample_mlp_ratio"][i])
        head_idx = choices_dict["n_head_choices"].index(config["sample_n_head"][i])
        one_hot_mlp[i][mlp_idx] = 1
        one_hot_head[i][head_idx] = 1
    # get selected index for bias
    bias_idx = choices_dict["bias_choices"].index(config["sample_bias"])
    one_hot_bias[bias_idx] = 1
    one_hot = torch.cat(
        [
            one_hot_embed,
            one_hot_layer,
            one_hot_mlp.view(-1),
            one_hot_head.view(-1),
            one_hot_bias,
        ]
    )
    return one_hot


class PPLDataset(torch.utils.data.Dataset):
    "Dataset to load the hardware metrics data for training and testing"

    def __init__(
        self, search_space: str = "s", transform=None, metric: str = "perplexity"
    ):
        "Initialization"
        self.archs = []
        self.ppl = []
        self.metric = metric
        self.search_space = search_space
        self.load_data()

    def load_data(self):
        if self.search_space == "s":
            path = "arch_ppl_bench_s/ppl.pkl"
        elif self.search_space == "m":
            path = "arch_ppl_bench_m/ppl.pkl"
        else:
            path = "arch_ppl_bench_l/ppl.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)

        for arch in data:

            self.archs.append(
                convert_config_to_one_hot(
                    arch["arch"], search_spaces[self.search_space]
                )
            )
            self.ppl.append(arch[self.metric])

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ppl)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        one_hot = self.archs[index]
        metric = self.ppl[index]
        return one_hot, metric

import torch
import pickle
import os


def get_arch_feature_map_s(arch):
    arch_embed_choices = [768, 384, 192]
    layer_choices = [10, 11, 12]
    mlp_ratio_choices = [2, 3, 4]
    head_choices = [4, 8, 12]
    bias_choices = [True, False]
    # arch_feature_map
    arch_feature_map = []
    arch_feature_map.append(arch["sample_embed_dim"])
    arch_feature_map.append(arch["sample_n_layer"])
    arch_feature_map.extend(arch["sample_mlp_ratio"][0 : arch["sample_n_layer"]])
    for i in range(max(layer_choices) - arch["sample_n_layer"]):
        arch_feature_map.append(0)
    arch_feature_map.extend(arch["sample_n_head"][0 : arch["sample_n_layer"]])
    for i in range(max(layer_choices) - arch["sample_n_layer"]):
        arch_feature_map.append(0)
    if arch["sample_bias"]:
        arch_feature_map.append(1)
    else:
        arch_feature_map.append(0)
    # print(len(arch_feature_map))
    return arch_feature_map


def get_arch_feature_map_m(arch):
    arch_embed_choices = [1024, 512, 256]
    layer_choices = [22, 23, 24]
    mlp_ratio_choices = [2, 3, 4]
    head_choices = [8, 12, 16]
    bias_choices = [True, False]
    # arch_feature_map
    arch_feature_map = []
    arch_feature_map.append(arch["sample_embed_dim"])
    arch_feature_map.append(arch["sample_n_layer"])
    arch_feature_map.extend(arch["sample_mlp_ratio"][0 : arch["sample_n_layer"]])
    for i in range(max(layer_choices) - arch["sample_n_layer"]):
        arch_feature_map.append(0)
    arch_feature_map.extend(arch["sample_n_head"][0 : arch["sample_n_layer"]])
    for i in range(max(layer_choices) - arch["sample_n_layer"]):
        arch_feature_map.append(0)
    if arch["sample_bias"]:
        arch_feature_map.append(1)
    else:
        arch_feature_map.append(0)
    # print(len(arch_feature_map))
    return arch_feature_map


def get_arch_feature_map_l(arch):
    arch_embed_choices = [320, 640, 1280]
    layer_choices = [34, 35, 36]
    mlp_ratio_choices = [2, 3, 4]
    head_choices = [8, 16, 20]
    bias_choices = [True, False]
    # arch_feature_map
    arch_feature_map = []
    arch_feature_map.append(arch["sample_embed_dim"])
    arch_feature_map.append(arch["sample_n_layer"])
    arch_feature_map.extend(arch["sample_mlp_ratio"][0 : arch["sample_n_layer"]])
    for i in range(max(layer_choices) - arch["sample_n_layer"]):
        arch_feature_map.append(0)
    arch_feature_map.extend(arch["sample_n_head"][0 : arch["sample_n_layer"]])
    for i in range(max(layer_choices) - arch["sample_n_layer"]):
        arch_feature_map.append(0)
    if arch["sample_bias"]:
        arch_feature_map.append(1)
    else:
        arch_feature_map.append(0)
    # print(len(arch_feature_map))
    return arch_feature_map


def filter_archs(arch_list, observations):
    archs_unique = []
    for arch in arch_list:
        for obs in observations:
            if arch == obs["arch"]:
                archs_unique.append(obs)
                break
    return archs_unique


device_to_path_map_s = {
    "a100": "arch_stats/hwmetric/a100_s/",
    "v100": "arch_stats/hwmetric/v100_s/",
    "rtx2080": "arch_stats/hwmetric/rtx2080_s/",
    "rtx3080": "arch_stats/hwmetric/rtx3080_s/",
    "a6000": "arch_stats/hwmetric/a6000_s/",
    "a40": "arch_stats/hwmetric/a40_s/",
    "p100": "arch_stats/hwmetric/p100_s/",
    "amd": "arch_stats/hwmetric/amd_s/",
    "cpu_meta": "arch_stats/hwmetric/cpu_meta_s/",
    "cpu_mlgpu": "arch_stats/hwmetric/cpu_mlgpu_s/",
    "cpu_alldlc": "arch_stats/hwmetric/cpu_alldlc_s/",
    "cpu_a6000": "arch_stats/hwmetric/cpu_a6000_s/",
    "cpu_p100": "arch_stats/hwmetric/cpu_p100_s/",
}

device_to_path_map_m = {
    "a100": "arch_stats/hwmetric/a100_m/",
    "v100": "arch_stats/hwmetric/v100_m/",
    "rtx2080": "arch_stats/hwmetric/rtx2080_m/",
    "rtx3080": "arch_stats/hwmetric/rtx3080_m/",
    "a6000": "arch_stats/hwmetric/a6000_m/",
    "a40": "arch_stats/hwmetric/a40_m/",
    "p100": "arch_stats/hwmetric/p100_m/",
    "amd": "arch_stats/hwmetric/amd_m/",
    "cpu_meta": "arch_stats/hwmetric/cpu_meta_m/",
    "cpu_mlgpu": "arch_stats/hwmetric/cpu_mlgpu_m/",
    "cpu_alldlc": "arch_stats/hwmetric/cpu_alldlc_m/",
    "cpu_a6000": "arch_stats/hwmetric/cpu_a6000_m/",
    "cpu_p100": "arch_stats/hwmetric/cpu_p100_m/",
}

device_to_path_map_l = {
    "a100": "arch_stats/hwmetric/a100_l/",
    "v100": "arch_stats/hwmetric/v100_l/",
    "rtx2080": "arch_stats/hwmetric/rtx2080_l/",
    "rtx3080": "arch_stats/hwmetric/rtx3080_l/",
    "a6000": "arch_stats/hwmetric/a6000_l/",
    "a40": "arch_stats/hwmetric/a40_l/",
    "p100": "arch_stats/hwmetric/p100_l/",
    "amd": "arch_stats/hwmetric/amd_l/",
    "cpu_meta": "arch_stats/hwmetric/cpu_meta_l/",
    "cpu_mlgpu": "arch_stats/hwmetric/cpu_mlgpu_l/",
    "cpu_alldlc": "arch_stats/hwmetric/cpu_alldlc_l/",
    "cpu_a6000": "arch_stats/hwmetric/cpu_a6000_l/",
    "cpu_p100": "arch_stats/hwmetric/cpu_p100_l/",
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


class HWDataset(torch.utils.data.Dataset):
    "Dataset to load the hardware metrics data for training and testing"

    def __init__(
        self,
        mode: str = "train",
        device_name: str = "a100",
        search_space: str = "s",
        transform=None,
        metric: str = "latency",
    ):
        "Initialization"
        self.device_name = device_name
        self.search_space = search_space
        self.transform = False
        self.metric = metric
        self.archs_to_one_hot = []
        self.metric_obs = []
        self.load_data()
        self.mode = mode

    def load_data(self):
        if self.metric == "energy_gpu":
            self.load_data_energy_gpu()
        elif self.metric == "latency":
            self.load_data_latency()
        elif self.metric == "energy_cpu":
            self.load_data_energy_cpu()

    def load_data_latency(self):
        cat_list = []
        increment = 2500
        if self.search_space != "":
            self.search_space = "_" + self.search_space
        for i in range(0, 10000, increment):

            path = (
                "raw_results_hw/latency_"
                + self.device_name
                + self.search_space
                + "/efficiency_observations_"
                + str(i)
                + "_"
                + str(i + increment)
                + ".pkl"
            )
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                a = pickle.load(f)
                cat_list.extend(a)
        a = cat_list
        arch_path = "sampled_archs" + self.search_space + ".pkl"
        with open(arch_path, "rb") as f:
            arch_list = pickle.load(f)
        a = filter_archs(arch_list, a)
        if len(a) == 10000:
            arch_features_train = []
            latencies_train = []
            arch_features_test = []
            latencies_test = []
            train_fraction = 0.8
            a_train = a[: int(train_fraction * len(a))]
            a_test = a[int(train_fraction * len(a)) :]
            if "cpu" in self.device_name:
                key = "times_profiler_cpu"
                key_unit = "unit_cpu"
            else:
                key = "times_profiler_gpu"
                key_unit = "unit_gpu"
            for i in range(len(a_train)):
                for j in range(len(a_train[i][key])):
                    if self.search_space == "":
                        arch_features_train.append(
                            get_arch_feature_map_s(a_train[i]["arch"])
                        )
                    elif self.search_space == "_m":
                        arch_features_train.append(
                            get_arch_feature_map_m(a_train[i]["arch"])
                        )
                    elif self.search_space == "_l":
                        arch_features_train.append(
                            get_arch_feature_map_l(a_train[i]["arch"])
                        )
                    if "cpu" in self.device_name:
                        if a[i][key_unit][j] == "s":
                            latencies_train.append(a_train[i][key][j] * 1000)
                        else:
                            latencies_train.append(a_train[i][key][j])
                    else:
                        latencies_train.append(a_train[i][key][j])
            for i in range(len(a_test)):
                for j in range(len(a_test[i][key])):
                    if self.search_space == "":
                        arch_features_test.append(
                            get_arch_feature_map_s(a_test[i]["arch"])
                        )
                    elif self.search_space == "_m":
                        arch_features_test.append(
                            get_arch_feature_map_m(a_test[i]["arch"])
                        )
                    elif self.search_space == "_l":
                        arch_features_test.append(
                            get_arch_feature_map_l(a_test[i]["arch"])
                        )
                    if "cpu" in self.device_name:
                        if a_test[i][key_unit][j] == "s":
                            latencies_test.append(a_test[i][key][j] * 1000)
                        else:
                            latencies_test.append(a_test[i][key][j])
                    else:
                        latencies_test.append(a_test[i][key][j])
            self.arch_features_train = torch.tensor(arch_features_train)
            self.latencies_train = torch.tensor(latencies_train)
            self.arch_features_test = torch.tensor(arch_features_test)
            self.latencies_test = torch.tensor(latencies_test)

    def load_data_energy_cpu(self):
        assert "cpu" in self.device_name
        cat_list = []
        increment = 2500
        if self.search_space != "":
            self.search_space = "_" + self.search_space
        for i in range(0, 10000, increment):
            path = (
                "raw_results_hw/latency_"
                + self.device_name
                + self.search_space
                + "/efficiency_observations_"
                + str(i)
                + "_"
                + str(i + increment)
                + ".pkl"
            )
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                a = pickle.load(f)
            cat_list.extend(a)
        a = cat_list
        arch_path = "sampled_archs" + self.search_space + ".pkl"
        with open(arch_path, "rb") as f:
            arch_list = pickle.load(f)
        a = filter_archs(arch_list, a)
        if len(a) == 10000:
            arch_features_train = []
            latencies_train = []
            arch_features_test = []
            latencies_test = []
            train_fraction = 0.8
            a_train = a[: int(train_fraction * len(a))]
            a_test = a[int(train_fraction * len(a)) :]
            if "cpu" in self.device_name:
                key = "mean_energy_cpu"
                key_unit = "unit_cpu"
            else:
                print("unknown device")
            for i in range(len(a_train)):
                if self.search_space == "":
                    arch_features_train.append(
                        get_arch_feature_map_s(a_train[i]["arch"])
                    )
                elif self.search_space == "_m":
                    arch_features_train.append(
                        get_arch_feature_map_m(a_train[i]["arch"])
                    )
                elif self.search_space == "_l":
                    arch_features_train.append(
                        get_arch_feature_map_l(a_train[i]["arch"])
                    )
                latencies_train.append(a_train[i][key])
                # print(arch_features[-1], latencies[-1])
            for i in range(len(a_test)):
                if self.search_space == "":
                    arch_features_test.append(get_arch_feature_map_s(a_test[i]["arch"]))
                elif self.search_space == "_m":
                    arch_features_test.append(get_arch_feature_map_m(a_test[i]["arch"]))
                elif self.search_space == "_l":
                    arch_features_test.append(get_arch_feature_map_l(a_test[i]["arch"]))
                latencies_test.append(a_test[i][key])
                # print(arch_features[-1], latencies[-1])
            self.arch_features_train = torch.tensor(arch_features_train)
            self.latencies_train = torch.tensor(latencies_train)
            self.arch_features_test = torch.tensor(arch_features_test)
            self.latencies_test = torch.tensor(latencies_test)

    def load_data_energy_gpu(self):
        assert "cpu" not in self.device_name
        cat_list = []
        increment = 2500
        if self.search_space != "":
            self.search_space = "_" + self.search_space
        for i in range(0, 10000, increment):

            path = (
                "raw_results_hw/latency_"
                + self.device_name
                + self.search_space
                + "/efficiency_energy_observations_tracker_"
                + str(i)
                + "_"
                + str(i + increment)
                + ".pkl"
            )
            print(path)
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                a = pickle.load(f)
            cat_list.extend(a)
        a = cat_list
        arch_path = "sampled_archs" + self.search_space + ".pkl"
        with open(arch_path, "rb") as f:
            arch_list = pickle.load(f)
        a = filter_archs(arch_list, a)
        print(len(a))
        assert len(a) == 10000
        if len(a) == 10000:
            print("Processing", self.device_name)
            arch_features_train = []
            latencies_train = []
            arch_features_test = []
            latencies_test = []
            train_fraction = 0.8
            a_train = a[: int(train_fraction * len(a))]
            a_test = a[int(train_fraction * len(a)) :]
            for i in range(len(a_train)):
                for j in range(len(a_train[i]["energy_gpu"])):
                    if self.search_space == "":
                        arch_features_train.append(
                            get_arch_feature_map_s(a_train[i]["arch"])
                        )
                    elif self.search_space == "_m":
                        arch_features_train.append(
                            get_arch_feature_map_m(a_train[i]["arch"])
                        )
                    elif self.search_space == "_l":
                        arch_features_train.append(
                            get_arch_feature_map_l(a_train[i]["arch"])
                        )
                    latencies_train.append(a_train[i]["energy_gpu"][j])
                    # print(arch_features[-1], latencies[-1])
            for i in range(len(a_test)):
                for j in range(len(a_test[i]["energy_gpu"])):
                    if self.search_space == "":
                        arch_features_test.append(
                            get_arch_feature_map_s(a_test[i]["arch"])
                        )
                    elif self.search_space == "_m":
                        arch_features_test.append(
                            get_arch_feature_map_m(a_test[i]["arch"])
                        )
                    elif self.search_space == "_l":
                        arch_features_test.append(
                            get_arch_feature_map_l(a_test[i]["arch"])
                        )
                    latencies_test.append(a_test[i]["energy_gpu"][j])
                    # print(arch_features[-1], latencies[-1])
            self.arch_features_train = torch.tensor(arch_features_train)
            self.latencies_train = torch.tensor(latencies_train)
            self.arch_features_test = torch.tensor(arch_features_test)
            self.latencies_test = torch.tensor(latencies_test)
            # print(self.arch_features_train.shape)

    def __len__(self):
        "Denotes the total number of samples"
        if self.mode == "train":
            return self.arch_features_train.shape[0]
        else:
            return self.arch_features_test.shape[0]

    def __getitem__(self, idx):
        "Generates one sample of data"
        # Select sample
        if self.mode == "train":
            one_hot = self.arch_features_train[idx]
            metric = self.latencies_train[idx]
        else:
            one_hot = self.arch_features_test[idx]
            metric = self.latencies_test[idx]
        return one_hot, metric


if __name__ == "__main__":
    devices_all = [
        "P100",
        "a6000",
        "rtx2080",
        "rtx3080",
        "v100",
        "a100",
        "a40",
        "h100",
        "cpu_mlgpu",
        "cpu_alldlc",
        "cpu_p100",
        "cpu_p100",
        "cpu_a6000",
        "cpu_meta",
        "helix_cpu",
    ]
    models = ["", "m", "l"]
    gpus = ["P100", "a6000", "rtx2080", "rtx3080", "v100", "a100", "a40", "h100"]
    cpus = [
        "cpu_mlgpu",
        "cpu_alldlc",
        "cpu_p100",
        "cpu_p100",
        "cpu_a6000",
        "cpu_meta",
        "helix_cpu",
    ]
    metric_energy_gpu = "energy_gpu"
    metric_energy_cpu = "energy_cpu"
    metric_latency = "latency"
    """for device in devices_all:
           for model in models:
               dset = HWDataset(mode = "train", device_name=device, search_space=model, transform=None, metric =metric_latency)
               print(len(dset.arch_features_train))
               print(len(dset.arch_features_test))
               print(device)
               print(model)
               assert len(dset.latencies_test) == len(dset.arch_features_test)
               assert len(dset.latencies_train) == len(dset.latencies_train)
               if device == "helix_cpu" and model == "":
                 assert len(dset.latencies_train) == 8000
                 assert len(dset.latencies_test) == 2000
               else:
                 assert len(dset.latencies_train) == 80000
                 assert len(dset.latencies_test) == 20000 """
    for device in gpus:
        for model in models:
            dset = HWDataset(
                mode="train",
                device_name=device,
                search_space=model,
                transform=None,
                metric=metric_energy_gpu,
            )
            print(len(dset.arch_features_train))
            print(len(dset.arch_features_test))
            print(device)
            print(model)
            assert len(dset.latencies_test) == len(dset.arch_features_test)
            assert len(dset.latencies_train) == len(dset.latencies_train)
            assert len(dset.latencies_train) == 400000
            assert len(dset.latencies_test) == 100000
    for device in cpus:
        for model in models:
            dset = HWDataset(
                mode="train",
                device_name=device,
                search_space=model,
                transform=None,
                metric=metric_energy_cpu,
            )
            print(len(dset.arch_features_train))
            print(len(dset.arch_features_test))
            print(device)
            print(model)
            assert len(dset.latencies_test) == len(dset.arch_features_test)
            assert len(dset.latencies_train) == len(dset.latencies_train)
            assert len(dset.latencies_train) == 8000
            assert len(dset.latencies_test) == 2000

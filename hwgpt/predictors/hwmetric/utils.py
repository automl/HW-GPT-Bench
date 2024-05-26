import torch
import pickle
import os
from hwgpt.predictors.hwmetric.net import Net
from lib.utils import (
    convert_str_to_arch,
    get_arch_feature_map,
    normalize_arch_feature_map,
    search_spaces,
)
import numpy as np
from hwgpt.predictors.hwmetric.conformal.surrogate.quantile_regression_model import (
    GradientBoostingQuantileRegressor,
)
from hwgpt.predictors.hwmetric.conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
)


def get_model_and_datasets(args):
    train_dataset = HWDataset(
        mode="train",
        device_name=args.device,
        search_space=args.search_space,
        metric=args.metric,
        type=args.type,
    )
    test_dataset = HWDataset(
        mode="test",
        device_name=args.device,
        search_space=args.search_space,
        metric=args.metric,
        type=args.type,
    )
    model = get_model(args)
    return model, train_dataset, test_dataset


def get_model(args):
    if args.model == "conformal_quantile":
        model = SymmetricConformalizedGradientBoostingQuantileRegressor(
            quantiles=args.num_quantiles
        )
    elif args.model == "quantile":
        model = GradientBoostingQuantileRegressor(quantiles=args.num_quantiles)
    elif args.model == "mlp":
        search_space = search_spaces[args.search_space]
        max_layers = max(search_space["n_layer_choices"])
        model = Net(max_layers, False, 128, 128)
    else:
        raise ValueError("Model type not supported")
    return model


class HWDataset(torch.utils.data.Dataset):
    "Dataset to load the hardware metrics data for training and testing"

    def __init__(
        self,
        device_name: str = "a100",
        search_space: str = "s",
        metric: str = "latencies",
        type: str = "median",
        mode: str = "train",
    ):
        "Initialization"
        self.device_name = device_name
        self.search_space = search_space
        self.transform = False
        self.metric = metric
        self.type = type
        self.mode = mode
        self.archs_to_one_hot = []
        self.metric_obs = []
        arch_stats_path = (
            "data_collection/gpt_datasets/gpt_" + str(self.search_space) + "/stats.pkl"
        )
        with open(arch_stats_path, "rb") as f:
            self.arch_stats = pickle.load(f)
        self.archs_all = list(self.arch_stats.keys())
        self.archs_train = self.archs_all[0:8000]
        self.archs_test = self.archs_all[8000:]
        self.load_data()

    def load_data(self):
        self.load_data()

    def process_arch_device(self, arch, metric, arch_features):
        arch_config = convert_str_to_arch(arch)
        feature = get_arch_feature_map(arch_config, self.search_space)
        feature = normalize_arch_feature_map(feature, self.search_space)
        if self.metric == "latencies" or self.metric == "energies":
            if self.type == "median":
                metric.append(
                    np.median(self.arch_stats[arch][self.device_name][self.metric])
                )
                arch_features.append(feature)
            else:
                if "cpu" in self.device_name and self.metric == "energies":
                    latencies_arch = [
                        self.arch_stats[arch][self.device_name][self.metric]
                    ]
                else:
                    latencies_arch = self.arch_stats[arch][self.device_name][
                        self.metric
                    ]
                latencies_arch = list(latencies_arch)
                for lat in latencies_arch:
                    metric.append(lat)
                    arch_features.append(feature)
        elif "memory" in self.metric:
            metric.append(self.arch_stats[arch][self.metric])
            arch_features.append(feature)
        else:
            raise ValueError("Invalid metric")

        return metric, arch_features

    def load_data(self):
        arch_features_train = []
        arch_features_test = []
        metric_train = []
        metric_test = []
        for arch in self.archs_train:
            metric_train, arch_features_train = self.process_arch_device(
                arch, metric_train, arch_features_train
            )
        for arch in self.archs_test:
            metric_test, arch_features_test = self.process_arch_device(
                arch, metric_test, arch_features_test
            )
        self.arch_features_train = torch.tensor(arch_features_train)
        self.latencies_train = torch.tensor(metric_train)
        self.arch_features_test = torch.tensor(arch_features_test)
        self.latencies_test = torch.tensor(metric_test)

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
        "cpu_xeon_silver",
        "cpu_xeon_gold",
        "cpu_amd_7502",
        "cpu_amd_7513",
        "cpu_amd_7452",
    ]
    models = ["s", "m", "l"]
    metrics = ["energies", "latencies", "float16_memory", "bfloat16_memory"]
    type = "quantile"
    for device in devices_all:
        for model in models:
            for metric in metrics:
                dset = HWDataset(
                    mode="train",
                    device_name=device,
                    search_space=model,
                    metric=metric,
                    type="quantile",
                )
                print(len(dset.arch_features_train))
                print(len(dset.arch_features_test))
                print(device)
                print(model)
                assert len(dset.latencies_test) == len(dset.arch_features_test)
                assert len(dset.latencies_train) == len(dset.latencies_train)
                if metric == "energies" and "cpu" not in device:
                    assert len(dset.latencies_train) == 400000
                    assert len(dset.latencies_test) == 100000
                elif metric == "latencies":
                    assert len(dset.latencies_train) == 80000
                    assert len(dset.latencies_test) == 20000
                else:
                    assert len(dset.latencies_train) == 8000
                    assert len(dset.latencies_test) == 2000

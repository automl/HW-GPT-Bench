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
from typing import List, Tuple
import argparse


def get_model_and_datasets(args: argparse.Namespace):
    train_dataset = HWDataset(
        mode="train",
        device_name=args.device,
        search_space=args.search_space,
        metric=args.metric,
    )
    test_dataset = HWDataset(
        mode="test",
        device_name=args.device,
        search_space=args.search_space,
        metric=args.metric,
    )
    model = get_model(args)
    return model, train_dataset, test_dataset


def get_model(args: argparse.Namespace):
    if args.model == "mlp":
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
        mode: str = "train",
    ):
        "Initialization"
        self.device_name = device_name
        self.search_space = search_space
        self.transform = False
        self.metric = metric
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

    def process_arch_device(
        self, arch: str, metric: str, arch_features: List
    ) -> Tuple[List, List]:
        arch_config = convert_str_to_arch(arch)
        feature = get_arch_feature_map(arch_config, self.search_space)
        feature = normalize_arch_feature_map(feature, self.search_space)
        if self.metric == "latencies" or self.metric == "energies":
            if "cpu" in self.device_name and self.metric == "energies":
                latencies_arch = [self.arch_stats[arch][self.device_name][self.metric]]
            else:
                latencies_arch = self.arch_stats[arch][self.device_name][self.metric]
            latencies_arch = list(latencies_arch)
            for lat in latencies_arch:
                metric.append(lat)
                arch_features.append(feature)
        elif "memory" in self.metric:
            metric.append(self.arch_stats[arch][self.metric])
            arch_features.append(feature)
        elif self.metric == "flops":
            metric.append(self.arch_stats[arch][self.metric] / 10**12)
            arch_features.append(feature)
        elif self.metric == "params":
            metric.append(self.arch_stats[arch][self.metric] / 10**6)
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

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        "Generates one sample of data"
        # Select sample
        if self.mode == "train":
            one_hot = self.arch_features_train[idx]
            metric = self.latencies_train[idx]
        else:
            one_hot = self.arch_features_test[idx]
            metric = self.latencies_test[idx]
        return one_hot, metric

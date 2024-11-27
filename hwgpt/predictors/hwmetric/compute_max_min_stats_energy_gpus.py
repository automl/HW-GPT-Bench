import pickle
from lib.utils import (
    predict_hw_surrogate,
    predict_hw_surrogate_multiple,
    get_arch_feature_map,
    search_spaces,
    normalize_arch_feature_map,
    get_hw_predictor_surrogate,
)
from hwgpt.api import HWGPT
from hwgpt.model.gpt.utils import sample_config_max, sample_config_min, sample_config
import argparse
from hwgpt.predictors.hwmetric.utils import get_model
import numpy as np
import os
from hwgpt.predictors.hwmetric.models.autogluon.multipredictor_train import MultilabelPredictor
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch HW Metric Predictor")
    parser.add_argument("--device", type=str, default="v100", help="device name")
    parser.add_argument(
        "--metric",
        type=str,
        default="energies",
    )
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--model", type=str, default="conformal_quantile")
    parser.add_argument("--type", type=str, default="quantile")
    parser.add_argument("--num_quantiles", type=str, default=9)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed ((default: 1)"
    )
    parser.add_argument("--base_path", type=str, default=".")
    args = parser.parse_args()
    base_path = os.path.join(
        args.base_path, "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
    )
    for s in ["s", "m", "l"]:
        for metric in ["latencies", "energies"]:
            for device in [
                "a100",
                "a40",
                "h100",
                "rtx2080",
                "rtx3080",
                "a6000",
                "v100",
                "P100",
                "cpu_xeon_silver",
                "cpu_xeon_gold",
                "cpu_amd_7502",
                "cpu_amd_7513",
                "cpu_amd_7452",
            ][::-1]:
                args.search_space = s
                args.metric = metric
                args.device = device
                #model = get_model(args)
                api = HWGPT(
                    search_space=s, use_supernet_surrogate=False
                )  # initialize API
                search_space = search_spaces[args.search_space]
                model = get_hw_predictor_surrogate(
                    args.search_space,
                    args.device,
                    args.metric,
                )
                max_config = sample_config_max(search_space)
                min_config = sample_config_min(search_space)
                random_arch = api.sample_arch()  # sample random architecture
                random_arch["sample_embed_dim"] = max(search_space["embed_dim_choices"])
                random_arch["sample_n_layer"] = max(search_space["n_layer_choices"])
                random_arch["sample_n_head"] = [max(search_space["n_head_choices"])] * max(search_space["n_layer_choices"])
                random_arch["sample_mlp_ratio"] = [max(search_space["mlp_ratio_choices"])] * max(search_space["n_layer_choices"])
                random_arch["sample_bias"] = "True"
                print(random_arch)
                api.set_arch(random_arch)  # set  arch
                hwmetric_max = api.query(metric=metric, device=device)  # query energy
                random_arch = api.sample_arch()  # sample random architecture
                random_arch["sample_embed_dim"] = min(search_space["embed_dim_choices"])
                random_arch["sample_n_layer"] = min(search_space["n_layer_choices"])
                random_arch["sample_n_head"] = [min(search_space["n_head_choices"])] * min(search_space["n_layer_choices"])
                random_arch["sample_mlp_ratio"] = [min(search_space["mlp_ratio_choices"])] * min(search_space["n_layer_choices"])
                random_arch["sample_bias"] = "False"
                print(random_arch)
                api.set_arch(random_arch)  # set  arch
                hwmetric_min = api.query(metric=metric, device=device)  # query energy
                max_min_stats = {"max": hwmetric_max[metric][device], "min": hwmetric_min[metric][device]}
                if args.type == "":
                    model_stats_path = (
                        base_path
                        + "stats_max_min_"
                        + args.metric
                        + "_"
                        + args.search_space
                        + ".pkl"
                    )
                else:
                    model_stats_path = (
                        base_path
                        + "stats_max_min_"
                        + args.metric
                        + "_"
                        + args.search_space
                        + "_"
                        + args.model
                        + "_"
                        + args.type
                        + "_"
                        + args.device
                        + ".pkl"
                    )
                with open(model_stats_path, "wb") as f:
                    pickle.dump(max_min_stats, f)
                print(max_min_stats)
                # compute mean and std
                lat_all = []
                for i in range(10000):
                    arch = api.sample_arch()
                    api.set_arch(arch)
                    hwmetric = api.query(metric=metric, device=device)
                    lat_all.append(hwmetric[metric][device])

                mean = np.mean(lat_all, axis=0)
                std = np.std(lat_all, axis=0)
                mean_std_stats = {"mean": mean, "std": std}
                if args.type == "":
                    model_stats_path = (
                        base_path
                        + "stats_mean_std_"
                        + args.metric
                        + "_"
                        + args.search_space
                        + ".pkl"
                    )
                else:
                    model_stats_path = (
                        base_path
                        + "stats_mean_std_"
                        + args.metric
                        + "_"
                        + args.search_space
                        + "_"
                        + args.model
                        + "_"
                        + args.type
                        + "_"
                        + args.device
                        + ".pkl"
                    )
                print(mean_std_stats)
                with open(model_stats_path, "wb") as f:
                    pickle.dump(mean_std_stats, f)

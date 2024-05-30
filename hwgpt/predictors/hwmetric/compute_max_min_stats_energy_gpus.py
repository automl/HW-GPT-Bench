import pickle
from lib.utils import (
    predict_hw_surrogate,
    get_arch_feature_map,
    search_spaces,
    normalize_arch_feature_map,
    get_hw_predictor_surrogate,
)
from hwgpt.model.gpt.utils import sample_config_max, sample_config_min, sample_config
import argparse
from hwgpt.predictors.hwmetric.utils import get_model
import numpy as np

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
    args = parser.parse_args()
    base_path = "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
    model = get_model(args)
    search_space = search_spaces[args.search_space]
    model = get_hw_predictor_surrogate(
        max(search_space["n_layer_choices"]),
        args.search_space,
        args.device,
        args.model,
        args.type,
        args.metric,
    )
    max_config = sample_config_max(search_space)
    min_config = sample_config_min(search_space)

    max_feature = normalize_arch_feature_map(
        get_arch_feature_map(max_config, args.search_space), args.search_space
    )
    min_feature = normalize_arch_feature_map(
        get_arch_feature_map(min_config, args.search_space), args.search_space
    )
    lats_max = max(
        predict_hw_surrogate([max_feature], model, args.model, return_all=True)[0]
    )
    lats_min = min(
        predict_hw_surrogate([min_feature], model, args.model, return_all=True)[0]
    )
    max_min_stats = {"max": lats_max, "min": lats_min}
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

    # compute mean and std
    lat_all = []
    for i in range(10000):
        arch = sample_config(search_space, seed=np.random.randint(0, 100000))
        feature = normalize_arch_feature_map(
            get_arch_feature_map(arch, args.search_space), args.search_space
        )
        lat = predict_hw_surrogate([feature], model, args.model, return_all=True)[0]
        for latency in lat:
            lat_all.append(latency.item())

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

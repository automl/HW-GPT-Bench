import pickle
from lib.utils import (
    predict_hw_surrogate,
    get_arch_feature_map,
    search_spaces,
    normalize_arch_feature_map,
)
from hwgpt.model.gpt.utils import sample_config_max, sample_config_min
import argparse
from hwgpt.predictors.hwmetric.utils import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch HW Metric Predictor")
    parser.add_argument("--device", type=str, default="a100", help="device name")
    parser.add_argument(
        "--metric",
        type=str,
        default="energies",
    )
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("---model", type="str", default="conformal_quantile")
    parser.add_argument("--type", type="str", default="quantile")
    parser.add_argument("--num_quantiles", type="str", default=10)
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
    model = get_model(args)
    search_space = search_spaces[args.search_space]
    base_path = (
        "data_collection/gpt_datasets/predictor_ckpts/hwmetric/" + str(args.model) + "/"
    )
    model_path = (
        base_path + args.metric + "_" + args.search_space + "_" + args.device + ".pkl"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    max_config = sample_config_max(search_space)
    min_config = sample_config_min(search_space)
    max_feature = normalize_arch_feature_map(
        get_arch_feature_map(max_config, args.search_space)
    )
    min_feature = normalize_arch_feature_map(
        get_arch_feature_map(min_config, args.search_space)
    )
    lats_max = max(
        predict_hw_surrogate(max_feature, model, args.model, return_quantile=True)
    )
    lats_min = min(
        predict_hw_surrogate(min_feature, model, args.model, return_quantiles=True)
    )
    max_min_stats = {"max": lats_max, "min": lats_min}
    model_stats_path = (
        base_path
        + "stats_max_min_"
        + args.metric
        + "_"
        + args.search_space
        + "_"
        + args.device
        + ".pkl"
    )
    with open(model_stats_path, "wb") as f:
        pickle.dump(max_min_stats, f)

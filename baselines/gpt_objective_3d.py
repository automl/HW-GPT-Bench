from syne_tune import Reporter
import time
from syne_tune import Reporter
import pickle
import torch
import numpy as np
import random
from lib.utils import (
    convert_config_to_one_hot,
    predict_hw_surrogate,
    get_arch_feature_map,
    normalize_objectives,
    normalize_ppl,
    get_all_hw_surrogates,
    get_ppl_predictor_surrogate,
    normalize_arch_feature_map,
    get_max_min_stats,
    search_spaces,
)


report = Reporter()


def objective(sampled_config, device_list, search_space, surrogate_type, objectives):
    max_layers = get_max_min_stats(search_space)["max_layers"]
    arch_feature_map = get_arch_feature_map(sampled_config, search_space)
    arch_feature_map_ppl_predictor = convert_config_to_one_hot(
        sampled_config, search_space=search_space
    )
    arch_feature_map_predictor = normalize_arch_feature_map(
        arch_feature_map, search_space
    )
    # print(arch_feature_map_predictor)
    ppl_predictor = get_ppl_predictor_surrogate(search_space)
    perplexity = ppl_predictor(arch_feature_map_ppl_predictor.cuda().unsqueeze(0))
    hw_metric_1_surrogate, hw_metric_2_surrogate = get_all_hw_surrogates(
        max_layers, search_space, objectives, device_list, surrogate_type
    )
    hw_metric_1 = predict_hw_surrogate(
        arch_feature_map_predictor, hw_metric_1_surrogate, surrogate_type
    )
    hw_metric_2 = predict_hw_surrogate(
        arch_feature_map_predictor, hw_metric_2_surrogate, surrogate_type, objectives[1]
    )

    hw_metric_1_norm, hw_metric_2_norm = normalize_objectives(
        [hw_metric_1, hw_metric_2], objectives, device_list, search_space
    )
    ppl = perplexity.item()
    ppl_norm = normalize_ppl(ppl)
    report(
        perplexity=ppl_norm, hw_metric_1=hw_metric_1_norm, hw_metric_2=hw_metric_2_norm
    )


if __name__ == "__main__":
    import logging
    import argparse

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--surrogate_type", type=str, default="conformal_quantile")
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--device_1", type=str, default="a6000")
    parser.add_argument("--device_2", type=str, default="rtx2080")
    parser.add_argument("--max_layers", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--objective_1", type=str, default="energy")
    parser.add_argument("--objective_2", type=str, default="latency")
    args = parser.parse_known_args()[0]
    search_space = search_spaces[args.search_space]
    max_layers = max(search_spaces["n_layer_choices"])
    for i in range(max_layers):
        parser.add_argument(f"--num_heads_{i}", type=int, default=12)
        parser.add_argument(f"--mlp_ratio_{i}", type=int, default=4)

    # Evaluate objective and report results to Syne Tune
    args, _ = parser.parse_known_args()
    print(vars(args))
    sample_config = {}
    sample_config["sample_n_layer"] = args.num_layers
    sample_config["sample_embed_dim"] = args.embed_dim
    sample_config["sample_bias"] = args.bias
    sample_config["sample_n_head"] = []
    sample_config["sample_mlp_ratio"] = []
    for i in range(max_layers):
        sample_config["sample_n_head"].append(getattr(args, f"num_heads_{i}"))
        sample_config["sample_mlp_ratio"].append(getattr(args, f"mlp_ratio_{i}"))
    objective(
        sampled_config=sample_config,
        search_space=args.search_space,
        surrogate_type=args.surrogate_type,
        device_list=[args.device_1, args.device_2],
        objectives=[args.objective_1, args.objective2],
    )

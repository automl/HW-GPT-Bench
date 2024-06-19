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

from typing import List, Dict, Any
from analysis.autogluon_gpu_latencies import MultilabelPredictor

report = Reporter()


def objective(
    sampled_config: Dict[str, Any],
    device_list: List[str],
    search_space: str,
    surrogate_types: List[str],
    type: str,
    objectives: List[str],
) -> Reporter:
    max_layers = get_max_min_stats(search_space)["max_layers"]
    arch_feature_map = get_arch_feature_map(sampled_config, search_space)
    arch_feature_map_ppl_predictor = convert_config_to_one_hot(
        sampled_config, search_space=search_space
    )
    # print(arch_feature_map_predictor)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppl_predictor = get_ppl_predictor_surrogate(search_space)
    perplexity = ppl_predictor(arch_feature_map_ppl_predictor.to(device).unsqueeze(0))
    hw_metric_1_surrogate, hw_metric_2_surrogate = get_all_hw_surrogates(
        max_layers, search_space, objectives, device_list, surrogate_types, type
    )
    hw_metric_1 = predict_hw_surrogate(
        [arch_feature_map], hw_metric_1_surrogate, objectives[0], device_list[0]
    )
    hw_metric_2 = predict_hw_surrogate(
        [arch_feature_map], hw_metric_2_surrogate, objectives[1], device_list[1]
    )
    if objectives[0] == "latencies":
        hw_metric_1 = hw_metric_1 / 1000
    if objectives[1] == "latencies":
        hw_metric_2 = hw_metric_2 / 1000
    hw_metric_1_norm, hw_metric_2_norm = normalize_objectives(
        [hw_metric_1, hw_metric_2],
        objectives,
        device_list,
        search_space,
        surrogate_types,
        type,
    )
    ppl = perplexity.item()
    ppl_norm = normalize_ppl(ppl, search_space)
    # print(hw_metric_1_norm, hw_metric_2_norm, ppl_norm)
    report(
        perplexity=ppl_norm,
        hw_metric_1=hw_metric_1_norm,
        hw_metric_2=hw_metric_2_norm,
    )


if __name__ == "__main__":
    import logging
    import argparse

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--surrogate_1", type=str, default="conformal_quantile")
    parser.add_argument("--surrogate_2", type=str, default="conformal_quantile")
    parser.add_argument("--type", type=str, default="quantile")
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--device_1", type=str, default="a6000")
    parser.add_argument("--device_2", type=str, default="rtx2080")
    parser.add_argument("--max_layers", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--objective_1", type=str, default="energies")
    parser.add_argument("--objective_2", type=str, default="latencies")
    args = parser.parse_known_args()[0]
    search_space = search_spaces[args.search_space]
    max_layers = max(search_space["n_layer_choices"])
    for i in range(max_layers):
        parser.add_argument(f"--num_heads_{i}", type=int, default=12)
        parser.add_argument(f"--mlp_ratio_{i}", type=int, default=4)

    # Evaluate objective and report results to Syne Tune
    args, _ = parser.parse_known_args()
    print(vars(args))
    sample_config = {}
    sample_config["sample_n_layer"] = args.num_layers
    sample_config["sample_embed_dim"] = args.embed_dim
    sample_config["sample_bias"] = str(args.bias)
    sample_config["sample_n_head"] = []
    sample_config["sample_mlp_ratio"] = []
    for i in range(max_layers):
        sample_config["sample_n_head"].append(getattr(args, f"num_heads_{i}"))
        sample_config["sample_mlp_ratio"].append(getattr(args, f"mlp_ratio_{i}"))
    objective(
        sampled_config=sample_config,
        search_space=args.search_space,
        surrogate_types=[args.surrogate_1, args.surrogate_2],
        type=args.type,
        device_list=[args.device_1, args.device_2],
        objectives=[args.objective_1, args.objective_2],
    )

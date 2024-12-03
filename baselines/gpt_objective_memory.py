from syne_tune import Reporter
import time
from syne_tune import Reporter
from typing import Dict, Any
from lib.utils import normalize_memory, normalize_ppl, search_spaces
import os
import torch
from hwgpt.api import HWGPT

report = Reporter()
from hwgpt.predictors.hwmetric.models.autogluon.multipredictor_train import (
    MultilabelPredictor,
)


def objective(
    sampled_config: Dict[str, Any],
    device: str,
    search_space: str,
    surrogate_type: str,
    type: str,
    objective: str,
    base_path: str = ".",
) -> Reporter:
    api = HWGPT(
        search_space=search_space, use_supernet_surrogate=False
    )  # initialize API
    api.set_arch(sampled_config)  # set  arch
    perplexity = api.query(metric="perplexity", predictor="mlp")["perplexity"]
    hw_metric = api.query(metric=objective)[objective]
    ppl = perplexity
    ppl_norm = normalize_ppl(ppl, search_space, method="max-min")
    hw_metric_norm = normalize_memory(hw_metric, search_space, objective)
    report(perplexity=ppl_norm, hw_metric=hw_metric_norm.item())


if __name__ == "__main__":
    import logging
    import argparse

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--surrogate_type", type=str, default="memory")
    parser.add_argument("--type", type=str, default="median")
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--device", type=str, default="v100")
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--objective", type=str, default="float16_memory")
    parser.add_argument("--base_path", type=str, default=".")
    args = parser.parse_known_args()[0]
    search_space = search_spaces[args.search_space]
    max_layers = max(search_space["n_layer_choices"])
    for i in range(max_layers):
        parser.add_argument(f"--num_heads_{i}", type=int, default=12)
        parser.add_argument(f"--mlp_ratio_{i}", type=int, default=4)

    args, _ = parser.parse_known_args()
    # Evaluate objective and report results to Syne Tune
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
        surrogate_type=args.surrogate_type,
        type=args.type,
        device=args.device,
        objective=args.objective,
        base_path=args.base_path,
    )

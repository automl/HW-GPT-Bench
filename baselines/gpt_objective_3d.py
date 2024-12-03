from syne_tune import Reporter
from lib.utils import (
    normalize_objectives,
    normalize_ppl,
    search_spaces,
)

from typing import List, Dict, Any
from hwgpt.predictors.hwmetric.models.autogluon.multipredictor_train import (
    MultilabelPredictor,
)
from hwgpt.api import HWGPT

report = Reporter()


def objective(
    sampled_config: Dict[str, Any],
    device_list: List[str],
    search_space: str,
    surrogate_types: List[str],
    type: str,
    objectives: List[str],
    base_path: str = ".",
) -> Reporter:
    api = HWGPT(search_space=search_space, use_supernet_surrogate=False)
    api.set_arch(sampled_config)
    perplexity = api.query(metric="perplexity", predictor="mlp")["perplexity"]
    hw_metrics = api.query(metric=objectives, device=device_list)
    hw_metrics = [
        hw_metrics[objective][device]
        for objective, device in zip(objectives, device_list)
    ]
    ppl = perplexity
    ppl_norm = normalize_ppl(ppl, search_space, method="max-min")
    hw_metrics_norm = normalize_objectives(
        hw_metrics, objectives, device_list, search_space, surrogate_types, type
    )
    report(
        perplexity=ppl_norm,
        hw_metric_1=hw_metrics_norm[0],
        hw_metric_2=hw_metrics_norm[1],
    )


if __name__ == "__main__":
    import logging
    import argparse

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--surrogate_1", type=str, default="autogluon")
    parser.add_argument("--surrogate_2", type=str, default="autogluon")
    parser.add_argument("--type", type=str, default="autogluon")
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--device_1", type=str, default="a6000")
    parser.add_argument("--device_2", type=str, default="rtx2080")
    parser.add_argument("--max_layers", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--objective_1", type=str, default="energies")
    parser.add_argument("--objective_2", type=str, default="latencies")
    parser.add_argument("--base_path", type=str, default=".")

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
        base_path=args.base_path,
    )

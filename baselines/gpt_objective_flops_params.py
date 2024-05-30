from syne_tune import Reporter
from syne_tune import Reporter
from lib.utils import search_spaces, get_max_min_true_metric, normalize_ppl
from hwgpt.api import HWGPTBenchAPI

report = Reporter()


def objective(sampled_config, search_space, objective):
    api = HWGPTBenchAPI(search_space=search_space)

    api.set_arch(sampled_config)
    if objective == "flops":
        hw_metric = api.get_flops()
    elif objective == "params":
        hw_metric = api.get_params()
    else:
        raise ValueError("Invalid hw metric")
    perplexity = api.compute_predictions_ppl()
    # print(perplexity)
    perplexity = normalize_ppl(perplexity, search_space, method="max-min")
    # print(perplexity)
    max_min_metric = get_max_min_true_metric(api, objective)
    hw_metric_norm = (hw_metric - max_min_metric["min"]) / (
        max_min_metric["max"] - max_min_metric["min"]
    )
    report(perplexity=perplexity, hw_metric=hw_metric_norm)


if __name__ == "__main__":
    import logging
    import argparse

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--objective", type=str, default="flops")
    args = parser.parse_known_args()[0]
    search_space = search_spaces[args.search_space]
    max_layers = max(search_space["n_layer_choices"])
    for i in range(max_layers):
        parser.add_argument(f"--num_heads_{i}", type=int, default=16)
        parser.add_argument(f"--mlp_ratio_{i}", type=int, default=4)

    args, _ = parser.parse_known_args()
    # Evaluate objective and report results to Syne Tune
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
        objective=args.objective,
    )

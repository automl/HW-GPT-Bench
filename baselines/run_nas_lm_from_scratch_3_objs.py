from syne_tune import Reporter
import time
from syne_tune import Reporter
from predictors.hwmetric.fit_all_quantile_regression_energy import (
    get_arch_feature_map_s,
    get_arch_feature_map_m,
    get_arch_feature_map_l,
)
from predictors.metric.utils import convert_config_to_one_hot, search_spaces
from conformal.surrogate.quantile_regression_model import (
    QuantileRegressorPredictions,
    GradientBoostingQuantileRegressor,
)
from conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
)
import pickle
import torch
from predictors.metric.net import Net
from predictors.hwmetric.net import Net as Nethw
import numpy as np
import random


def normalize_ppl(ppl):
    with open("ppl_predictor_ckpts/max_min_stats_s.pkl", "rb") as f:
        max_min_stats = pickle.load(f)
        max_ppl = max_min_stats["max"]
        min_ppl = max_min_stats["min"]
        ppl = (ppl - min_ppl) / (max_ppl - min_ppl)
        return ppl


def normalize_energy(energy, device):
    with open(
        "hwmetric_predictor_ckpts/max_min_stats_energy_" + device + ".pkl", "rb"
    ) as f:
        max_min_stats = pickle.load(f)
        max_energy = max_min_stats["max"]
        min_energy = max_min_stats["min"]
        energy = (energy - min_energy) / (max_energy - min_energy)
    return energy


def normalize_latency(latency, device):
    with open(
        "hwmetric_predictor_ckpts/max_min_stats_latency_" + device + ".pkl", "rb"
    ) as f:
        max_min_stats = pickle.load(f)
        max_latency = max_min_stats["max"]
        min_latency = max_min_stats["min"]
        # print(max_latency, min_latency)
        latency = (latency - min_latency) / (max_latency - min_latency)
    # print(latency)
    return latency


def get_predictor_surrogate_path_energy(
    max_layers, search_space, device, surrogate_type
):
    if surrogate_type == "conformal_quantile":
        if search_space == "s":
            surrogate_path = (
                "hwmetric_predictor_ckpts/conformal_quantile_regression_energy_"
                + str(device)
                + ".pkl"
            )
        elif search_space == "m":
            surrogate_path = (
                "hwmetric_predictor_ckpts/conformal_quantile_regression_energy_"
                + str(device)
                + "_m.pkl"
            )
        else:
            surrogate_path = (
                "hwmetric_predictor_ckpts/conformal_quantile_regression_energy_"
                + str(device)
                + "_l.pkl"
            )
            # print(f)
        with open(surrogate_path, "rb") as f:
            predictor = pickle.load(f)
    elif surrogate_type == "quantile":
        if search_space == "s":
            surrogate_path = (
                "hwmetric_predictor_ckpts/quantile_regression_energy_"
                + str(device)
                + ".pkl"
            )
        elif search_space == "m":
            surrogate_path = (
                "hwmetric_predictor_ckpts/quantile_regression_energy_"
                + str(device)
                + "_m.pkl"
            )
        else:
            surrogate_path = (
                "hwmetric_predictor_ckpts/quantile_regression_energy_"
                + str(device)
                + "_l.pkl"
            )
        with open(surrogate_path, "rb") as f:
            predictor = pickle.load(f)
    elif surrogate_type == "mlp":
        predictor = Nethw(max_layers, False, 256, 256).cuda()
        if search_space == "s":
            path = "hwmetric_predictor_ckpts/" + str(device) + "_energy_gpu_.pt"
        elif search_space == "m":
            path = "hwmetric_predictor_ckpts/" + str(device) + "_energy_gpu_m.pt"
        else:
            path = "hwmetric_predictor_ckpts/" + str(device) + "_energy_gpu_l.pt"
        predictor.load_state_dict(torch.load(path))
    return predictor


def get_predictor_surrogate_path_latency(
    max_layers, search_space, device, surrogate_type
):
    if surrogate_type == "conformal_quantile":
        if search_space == "s":
            surrogate_path = (
                "hwmetric_predictor_ckpts/conformal_quantile_regression_latency_"
                + str(device)
                + ".pkl"
            )
        elif search_space == "m":
            surrogate_path = (
                "hwmetric_predictor_ckpts/conformal_quantile_regression_latency_"
                + str(device)
                + "_m.pkl"
            )
        else:
            surrogate_path = (
                "hwmetric_predictor_ckpts/conformal_quantile_regression_latency_"
                + str(device)
                + "_l.pkl"
            )
            # print(f)
        with open(surrogate_path, "rb") as f:
            predictor = pickle.load(f)
    elif surrogate_type == "quantile":
        if search_space == "s":
            surrogate_path = (
                "hwmetric_predictor_ckpts/quantile_regression_latency_"
                + str(device)
                + ".pkl"
            )
        elif search_space == "m":
            surrogate_path = (
                "hwmetric_predictor_ckpts/quantile_regression_latency_"
                + str(device)
                + "_m.pkl"
            )
        else:
            surrogate_path = (
                "hwmetric_predictor_ckpts/quantile_regression_latency_"
                + str(device)
                + "_l.pkl"
            )
        with open(surrogate_path, "rb") as f:
            predictor = pickle.load(f)
    elif surrogate_type == "mlp":
        predictor = Nethw(max_layers, False, 256, 256).cuda()
        if search_space == "s":
            path = "hwmetric_predictor_ckpts/" + str(device) + "_latency_gpu_.pt"
        elif search_space == "m":
            path = "hwmetric_predictor_ckpts/" + str(device) + "_latency_gpu_m.pt"
        else:
            path = "hwmetric_predictor_ckpts/" + str(device) + "_latency_gpu_l.pt"
        predictor.load_state_dict(torch.load(path))
    return predictor


def predict_hw_surrogate(
    arch, surrogate, surrogate_type, return_all=False, return_quantiles=False
):
    if surrogate_type == "conformal_quantile" or surrogate_type == "quantile":
        energy = surrogate.predict(arch).results_stacked
        if return_quantiles:
            return surrogate.predict(arch)
        quantile = np.random.randint(low=0, high=31, size=1)
        # print(energy.shape)
        if return_all:
            return energy
        energy = energy[0, quantile]
    else:
        energy = surrogate(arch.cuda().unsqueze(0)).item()
    return energy


report = Reporter()


def objective(
    max_layers,
    sampled_config,
    device_latency,
    device_energy,
    search_space,
    surrogate_type,
):
    if search_space == "s":
        arch_feature_map_predictor = get_arch_feature_map_s(sampled_config)
    elif search_space == "m":
        arch_feature_map_predictor = get_arch_feature_map_m(sampled_config)
    else:
        arch_feature_map_predictor = get_arch_feature_map_l(sampled_config)
    arch_feature_map_ppl_predictor = convert_config_to_one_hot(
        sampled_config, search_spaces[search_space]
    )
    with open("hwmetric_predictor_ckpts/mean_std_stats_archs.pkl", "rb") as f:
        mean_std_stats = pickle.load(f)
    arch_feature_map_predictor = np.array([arch_feature_map_predictor])
    # print(arch_feature_map_predictor.shape)
    for i in range(arch_feature_map_predictor.shape[1]):
        # print(mean_std_stats["mean"][i], mean_std_stats["std"][i])
        arch_feature_map_predictor[:, i] = (
            arch_feature_map_predictor[:, i] - mean_std_stats["mean"][i]
        ) / mean_std_stats["std"][i]
    # print(arch_feature_map_predictor)
    ppl_predictor = Net(max_layers, 128).cuda()
    if search_space == "s":
        pred_path = "ppl_predictor_ckpts/perplexity_s.pt"
    elif search_space == "m":
        pred_path = "ppl_predictor_ckpts/perplexity_m.pt"
    else:
        pred_path = "ppl_predictor_ckpts/perplexity_l.pt"
    ppl_predictor.load_state_dict(torch.load(pred_path))
    perplexity = ppl_predictor(arch_feature_map_ppl_predictor.cuda().unsqueeze(0))
    predictor_energy = get_predictor_surrogate_path_energy(
        max_layers, search_space, device_energy, surrogate_type
    )
    energy = predict_hw_surrogate(
        arch_feature_map_predictor, predictor_energy, surrogate_type
    )

    predictor_latency = get_predictor_surrogate_path_latency(
        max_layers, search_space, device_latency, surrogate_type
    )
    latency = predict_hw_surrogate(
        arch_feature_map_predictor, predictor_latency, surrogate_type
    )
    latency_norm = normalize_latency(latency.item(), device_latency)
    ppl = perplexity.item()
    ppl_norm = normalize_ppl(ppl)
    energy_norm = normalize_energy(energy.item(), device_energy)
    report(perplexity=ppl_norm, energy=energy_norm, latency=latency_norm)


if __name__ == "__main__":
    import logging
    import argparse

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # [3]
    max_layers = 12
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--surrogate_type", type=str, default="conformal_quantile")
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--device_energy", type=str, default="a6000")
    parser.add_argument("--device_latency", type=str, default="rtx2080")
    parser.add_argument("--max_layers", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--bias", type=bool, default=True)
    for i in range(max_layers):
        parser.add_argument(f"--num_heads_{i}", type=int, default=12)
        parser.add_argument(f"--mlp_ratio_{i}", type=int, default=4)

    args, _ = parser.parse_known_args()
    # Evaluate objective and report results to Syne Tune
    print(vars(args))
    sample_config = {}
    sample_config["sample_n_layer"] = args.num_layers
    sample_config["sample_embed_dim"] = args.embed_dim
    sample_config["sample_bias"] = args.bias
    sample_config[f"sample_n_head"] = []
    sample_config[f"sample_mlp_ratio"] = []
    for i in range(max_layers):
        sample_config[f"sample_n_head"].append(getattr(args, f"num_heads_{i}"))
        sample_config[f"sample_mlp_ratio"].append(getattr(args, f"mlp_ratio_{i}"))
    objective(
        max_layers=args.max_layers,
        sampled_config=sample_config,
        search_space=args.search_space,
        surrogate_type=args.surrogate_type,
        device_energy=args.device_energy,
        device_latency=args.device_latency,
    )

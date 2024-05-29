import pickle
import torch
from hwgpt.predictors.hwmetric.net import Net as Nethw
from hwgpt.predictors.metric.net import Net
import numpy as np
from typing import Any, Dict, Tuple, List
from hwgpt.model.gpt.utils import sample_config_max, sample_config_min

metrics_map = {
    "energies": "Energy (Wh)",
    "latencies": "Latency (ms)",
    "perplexity": "PPL",
    "bfloat16_memory": "Memory (MB)",
    "float16_memory": "Memory (MB)",
    "flops": "FLOPS",
    "params": "Params (MB)",
}
dims_map = {
    "embed_dim_choices": "Embedding Dimensions",
    "n_layer_choices": "Number of Layers",
    "mlp_ratio_choices": "MLP Ratios",
    "n_head_choices": "No. of Heads",
    "bias_choices": "Bias Choice",
}
search_spaces = {
    "s": {
        "embed_dim_choices": [192, 384, 768],
        "n_layer_choices": [10, 11, 12],
        "mlp_ratio_choices": [2, 3, 4],
        "n_head_choices": [4, 8, 12],
        "bias_choices": [True, False],
    },
    "m": {
        "embed_dim_choices": [256, 512, 1024],
        "n_layer_choices": [22, 23, 24],
        "mlp_ratio_choices": [2, 3, 4],
        "n_head_choices": [8, 12, 16],
        "bias_choices": [True, False],
    },
    "l": {
        "embed_dim_choices": [320, 640, 1280],
        "n_layer_choices": [34, 35, 36],
        "mlp_ratio_choices": [2, 3, 4],
        "n_head_choices": [8, 16, 20],
        "bias_choices": [True, False],
    },
}

choice_arch_config_map = {
    "embed_dim_choices": "sample_embed_dim",
    "n_layer_choices": "sample_n_layer",
    "mlp_ratio_choices": "sample_mlp_ratio",
    "n_head_choices": "sample_n_head",
    "bias_choices": "sample_bias",
}


def get_max_min_true_metric(api, metric=str) -> Dict[str, float]:
    max_config = sample_config_max(search_spaces[api.search_space_name])
    min_config = sample_config_min(search_spaces[api.search_space_name])
    if metric == "flops":
        api.set_arch(max_config)
        max_metric = api.get_flops()
        api.set_arch(min_config)
        min_metric = api.get_flops()
    else:
        api.set_arch(max_config)
        max_metric = api.get_params()
        api.set_arch(min_config)
        min_metric = api.get_params()
    return {"max": max_metric, "min": min_metric}


def convert_arch_to_str(arch:Dict[str,Any],scale:str)->str:
    str_mlp = ""
    str_heads = ""
    for i in range(arch["sample_n_layer"]):
        str_mlp = str_mlp+str(arch["sample_mlp_ratio"][i])+"-"
        str_heads = str_heads+str(arch["sample_n_head"][i])+"-"
    name = "gpt-"+str(scale)+"-"+str(arch["sample_n_layer"])+'-'+str(arch["sample_embed_dim"])+'-'+str_mlp+str_heads+str(arch["sample_bias"])
    print(name)
    return name 


def convert_str_to_arch(arch_str: str) -> Dict[str, Any]:
    arch_parts = arch_str.split("-")
    num_layers = arch_parts[2]
    embed_dim = arch_parts[3]
    mlp_ratios = arch_parts[4 : (4 + int(num_layers))]
    heads = arch_parts[(4 + int(num_layers)) : (4 + 2 * int(num_layers))]
    bias = arch_parts[-1]
    sampled_arch = {}
    sampled_arch["sample_n_layer"] = int(num_layers)
    sampled_arch["sample_embed_dim"] = int(embed_dim)
    sampled_arch["sample_bias"] = bool(bias)
    sampled_arch["sample_mlp_ratio"] = []
    sampled_arch["sample_n_head"] = []
    for i in range(len(mlp_ratios)):
        sampled_arch["sample_mlp_ratio"].append(int(mlp_ratios[i]))
        sampled_arch["sample_n_head"].append(int(heads[i]))
    return sampled_arch


def get_max_min_stats(search_space: str) -> Dict[str, int]:
    search_space_max_min = {}
    space = search_spaces[search_space]
    search_space_max_min["max_layers"] = max(space["n_layer_choices"])
    search_space_max_min["min_layers"] = min(space["n_layer_choices"])
    search_space_max_min["max_embed"] = max(space["embed_dim_choices"])
    search_space_max_min["min_embed"] = min(space["embed_dim_choices"])
    search_space_max_min["max_heads"] = max(space["n_head_choices"])
    search_space_max_min["min_heads"] = min(space["n_head_choices"])
    search_space_max_min["max_mlp_ratio"] = max(space["mlp_ratio_choices"])
    search_space_max_min["min_mlp_ratio"] = min(space["mlp_ratio_choices"])

    return search_space_max_min


def normalize_arch_feature_map(feature_map: List, search_space: str) -> List:
    search_space_max_min = get_max_min_stats(search_space)
    feature_map[0] = (feature_map[0] - search_space_max_min["min_embed"]) / (
        search_space_max_min["max_embed"] - search_space_max_min["min_embed"]
    )
    feature_map[1] = (feature_map[1] - search_space_max_min["min_layers"]) / (
        search_space_max_min["max_layers"] - search_space_max_min["min_layers"]
    )
    for i in range(2, 2 + search_space_max_min["max_layers"]):
        if feature_map[i] != 0:
            feature_map[i] = (
                feature_map[i] - search_space_max_min["min_mlp_ratio"]
            ) / (
                search_space_max_min["max_mlp_ratio"]
                - search_space_max_min["min_mlp_ratio"]
            )
    for i in range(
        2 + search_space_max_min["max_layers"],
        2 + 2 * search_space_max_min["max_layers"],
    ):
        if feature_map[i] != 0:
            feature_map[i] = (feature_map[i] - search_space_max_min["min_heads"]) / (
                search_space_max_min["max_heads"] - search_space_max_min["min_heads"]
            )
    return feature_map


def get_arch_feature_map(arch: Dict[str, Any], scale: str) -> List:
    if scale == "s":
        layer_choices = [10, 11, 12]
    elif scale == "m":
        layer_choices = [22, 23, 24]
    elif scale == "l":
        layer_choices = [34, 35, 36]
    # arch_feature_map
    arch_feature_map = []
    arch_feature_map.append(arch["sample_embed_dim"])
    arch_feature_map.append(arch["sample_n_layer"])
    arch_feature_map.extend(arch["sample_mlp_ratio"][0 : arch["sample_n_layer"]])
    for i in range(max(layer_choices) - arch["sample_n_layer"]):
        arch_feature_map.append(0)
    arch_feature_map.extend(arch["sample_n_head"][0 : arch["sample_n_layer"]])
    for i in range(max(layer_choices) - arch["sample_n_layer"]):
        arch_feature_map.append(0)
    if arch["sample_bias"]:
        arch_feature_map.append(1)
    else:
        arch_feature_map.append(0)
    # print(len(arch_feature_map))
    return arch_feature_map


def normalize_ppl(ppl: float, scale: str) -> float:
    with open(
        "data_collection/gpt_datasets/predictor_ckpts/metric/max_min_stats_"
        + str(scale)
        + ".pkl",
        "rb",
    ) as f:
        max_min_stats = pickle.load(f)
    max_ppl = max_min_stats["max"]
    min_ppl = max_min_stats["min"]
    ppl = (ppl - min_ppl) / (max_ppl - min_ppl)
    return ppl


def normalize_energy(energy: float, device: str, surrogate:str, data_type:str, scale: str, metric:str) -> float:
    base_path = (
        "data_collection/gpt_datasets/predictor_ckpts/hwmetric/" + str(surrogate) + "/"
    )
    model_path = (
        base_path
        + "stats_max_min_"
        + str(metric)
        +"_"
        + scale
        + "_"
        + surrogate
        + "_"
        + data_type
        + "_"
        + device
        + ".pkl"
    )
    with open(
        model_path,"rb"
    ) as f:
        max_min_stats = pickle.load(f)
        max_energy = max_min_stats["max"]
        min_energy = max_min_stats["min"]
        energy = (energy - min_energy) / (max_energy - min_energy)
    return energy


def normalize_latency(latency: float, device: str, surrogate:str, data_type:str, scale: str, metric:str) -> float:
    base_path = "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
    model_path = (
        base_path
        + "stats_max_min_"
        + str(metric)
        +"_"
        + scale
        + "_"
        + surrogate
        + "_"
        + data_type
        + "_"
        + device
        + ".pkl"
    )
    with open(
        model_path,"rb"
    ) as f:
        max_min_stats = pickle.load(f)
        max_latency = max_min_stats["max"]
        min_latency = max_min_stats["min"]
        # print(max_latency, min_latency)
        latency = (latency - min_latency) / (max_latency - min_latency)
    # print(latency)
    return latency


def normalize_objectives(
    metric_values: List,
    objectives_list: List[str],
    devices: List[str],
    search_space: str,
) -> float:
    metric_values_normalized = []
    for i, objective in enumerate(objectives_list):
        if objective == "latency":
            metric_values_normalized.append(
                normalize_latency(metric_values[i], devices[i], search_space)
            )
        elif objective == "energy":
            metric_values_normalized.append(
                normalize_energy(metric_values[i], devices[i], search_space)
            )
        else:
            raise ValueError("Metric nor supported")
    return metric_values_normalized


def get_all_hw_surrogates(
    max_layers: int,
    search_space: str,
    objectives: List[str],
    devices: List[str],
    surrogate_type: str,
) -> List[Any]:
    all_surrogates = []
    for i, objective in enumerate(objectives):
        all_surrogates.append(
            get_hw_predictor_surrogate(
                max_layers, search_space, devices[i], surrogate_type, objectives[i]
            )
        )
    return all_surrogates


def get_hw_predictor_surrogate(
    max_layers: int,
    search_space: str,
    device: str,
    surrogate_type: str,
    type: str = "quantile",
    metric: str = "energies",
) -> Any:
    base_path = (
        "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
        + str(surrogate_type)
        + "/"
    )
    model_path = base_path + metric + "_" + type + "_" + search_space + "_" + device
    print(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if surrogate_type == "conformal_quantile":
        surrogate_path = model_path + ".pkl"
        with open(surrogate_path, "rb") as f:
            predictor = pickle.load(f)
    elif surrogate_type == "quantile":
        surrogate_path = model_path + ".pkl"
        with open(surrogate_path, "rb") as f:
            predictor = pickle.load(f)
    elif surrogate_type == "mlp":
        predictor = Nethw(max_layers, False, 256, 256).to(device)
        path = model_path + ".pth"
        predictor.load_state_dict(torch.load(path, map_location=device))
    return predictor


def predict_hw_surrogate(
    arch: np.array,
    surrogate: Any,
    surrogate_type: str,
    return_all: bool = False,
    return_quantiles: bool = False,
) -> float:
    if surrogate_type == "conformal_quantile" or surrogate_type == "quantile":
        energy = surrogate.predict(arch).results_stacked
        if return_quantiles:
            return surrogate.predict(arch)
        quantile = np.random.randint(low=0, high=9, size=1)
        # print(energy.shape)
        if return_all:
            return energy
        energy = energy[0, quantile]
    else:
        energy = surrogate(torch.tensor(arch).cuda())  # .item()
    return energy


def get_ppl_predictor_surrogate(search_space: str) -> Any:
    if search_space == "s":
        max_layers = 12
    elif search_space == "m":
        max_layers = 24
    elif search_space == "l":
        max_layers = 36
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppl_predictor = Net(max_layers, 128).to(device)
    if search_space == "s":
        pred_path = (
            "data_collection/gpt_datasets/predictor_ckpts/metric/perplexity_s.pt"
        )
    elif search_space == "m":
        pred_path = (
            "data_collection/gpt_datasets/predictor_ckpts/metric/perplexity_m.pt"
        )
    else:
        pred_path = (
            "data_collection/gpt_datasets/predictor_ckpts/metric/perplexity_l.pt"
        )
    ppl_predictor.load_state_dict(torch.load(pred_path, map_location=device))
    return ppl_predictor


def convert_config_to_one_hot(
    config: Dict[str, Any], search_space: str
) -> torch.Tensor:
    choices_dict = search_spaces[search_space]
    one_hot_embed = torch.zeros(len(choices_dict["embed_dim_choices"]))
    one_hot_layer = torch.zeros(len(choices_dict["n_layer_choices"]))
    max_layers = max(choices_dict["n_layer_choices"])
    one_hot_mlp = torch.zeros(max_layers, len(choices_dict["mlp_ratio_choices"]))
    one_hot_head = torch.zeros(max_layers, len(choices_dict["n_head_choices"]))
    one_hot_bias = torch.zeros(len(choices_dict["bias_choices"]))
    # get selected index for embed dim
    embed_idx = choices_dict["embed_dim_choices"].index(config["sample_embed_dim"])
    one_hot_embed[embed_idx] = 1
    # get selected index for layer
    layer_idx = choices_dict["n_layer_choices"].index(config["sample_n_layer"])
    one_hot_layer[layer_idx] = 1
    # get selected index for mlp ratio and head
    for i in range(config["sample_n_layer"]):
        mlp_idx = choices_dict["mlp_ratio_choices"].index(config["sample_mlp_ratio"][i])
        head_idx = choices_dict["n_head_choices"].index(config["sample_n_head"][i])
        one_hot_mlp[i][mlp_idx] = 1
        one_hot_head[i][head_idx] = 1
    # get selected index for bias
    bias_idx = choices_dict["bias_choices"].index(config["sample_bias"])
    one_hot_bias[bias_idx] = 1
    one_hot = torch.cat(
        [
            one_hot_embed,
            one_hot_layer,
            one_hot_mlp.view(-1),
            one_hot_head.view(-1),
            one_hot_bias,
        ]
    )
    return one_hot

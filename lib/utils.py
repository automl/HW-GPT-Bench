import pickle
import torch
from hwgpt.predictors.hwmetric.net import Net as Nethw
from hwgpt.predictors.metric.net import Net
import numpy as np

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


def convert_arch_to_str(arch, scale):
    str_mlp = ""
    str_heads = ""
    for i in range(arch["sample_n_layer"]):
        str_mlp = str_mlp + str(arch["sample_mlp_ratio"][i])
        str_heads = str_heads + str(arch["sample_n_head"][i])
    name = (
        "gpt-"
        + str(scale)
        + "-"
        + str(arch["sample_n_layer"])
        + "-"
        + str(arch["sample_embed_dim"])
        + "-"
        + str_mlp
        + "-"
        + str_heads
        + "-"
        + str(arch["sample_bias"])
    )
    return name


def convert_str_to_arch(arch_str):
    arch_parts = arch_str.split("-")
    scale = arch_parts[1]
    num_layers = arch_parts[2]
    embed_dim = arch_parts[3]
    mlp_ratios = arch_parts[4]
    heads = arch_parts[5]
    bias = arch_parts[6]
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


def get_max_min_stats(search_space):
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


def normalize_arch_feature_map(feature_map, search_space):
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


def get_arch_feature_map(arch, scale):
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


def normalize_ppl(ppl, scale):
    with open("ppl_predictor_ckpts/max_min_stats_" + str(scale) + ".pkl", "rb") as f:
        max_min_stats = pickle.load(f)
    max_ppl = max_min_stats["max"]
    min_ppl = max_min_stats["min"]
    ppl = (ppl - min_ppl) / (max_ppl - min_ppl)
    return ppl


def normalize_energy(energy, device, scale):
    with open(
        "hwmetric_predictor_ckpts/max_min_stats_energy_"
        + device
        + "_"
        + str(scale)
        + ".pkl"
    ) as f:
        max_min_stats = pickle.load(f)
        max_energy = max_min_stats["max"]
        min_energy = max_min_stats["min"]
        energy = (energy - min_energy) / (max_energy - min_energy)
    return energy


def normalize_latency(latency, device, scale):
    with open(
        "hwmetric_predictor_ckpts/max_min_stats_latency_"
        + device
        + "_"
        + str(scale)
        + ".pkl"
    ) as f:
        max_min_stats = pickle.load(f)
        max_latency = max_min_stats["max"]
        min_latency = max_min_stats["min"]
        # print(max_latency, min_latency)
        latency = (latency - min_latency) / (max_latency - min_latency)
    # print(latency)
    return latency


def normalize_objectives(metric_values, objectives_list, devices, search_space):
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
    max_layers, search_space, objectives, devices, surrogate_type
):
    all_surrogates = []
    for i, objective in enumerate(objectives):
        all_surrogates.append(
            get_hw_predictor_surrogate(
                max_layers, search_space, devices[i], surrogate_type, objectives[i]
            )
        )
    return all_surrogates


def get_hw_predictor_surrogate(
    max_layers, search_space, device, surrogate_type, metric="energy"
):  
    base_path = "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"+str(surrogate_type)+"/"
    model_path = base_path+metric+"_"+search_space+"_"+device
    if surrogate_type == "conformal_quantile":
        surrogate_path = (
            model_path+ ".pkl"
        )
        with open(surrogate_path, "rb") as f:
            predictor = pickle.load(f)
    elif surrogate_type == "quantile":
        surrogate_path = (
            model_path + ".pkl"
        )
        with open(surrogate_path, "rb") as f:
            predictor = pickle.load(f)
    elif surrogate_type == "mlp":
        predictor = Nethw(max_layers, False, 256, 256).cuda()
        path = (
            model_path
            + ".pth"
        )
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


def get_ppl_predictor_surrogate(search_space):
    if search_space == "s":
        max_layers = 12
    elif search_space == "m":
        max_layers = 24
    elif search_space == "l":
        max_layers = 36
    ppl_predictor = Net(max_layers, 128).cuda()
    if search_space == "s":
        pred_path = "ppl_predictor_ckpts/perplexity_s.pt"
    elif search_space == "m":
        pred_path = "ppl_predictor_ckpts/perplexity_m.pt"
    else:
        pred_path = "ppl_predictor_ckpts/perplexity_l.pt"
    ppl_predictor.load_state_dict(torch.load(pred_path))
    return ppl_predictor


def convert_config_to_one_hot(config, search_space):
    choices_dict = search_space[search_space]
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

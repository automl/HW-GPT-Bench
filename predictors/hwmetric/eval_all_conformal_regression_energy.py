import os 
import pickle
from conformal.surrogate.quantile_regression_model import (
    QuantileRegressorPredictions,
    GradientBoostingQuantileRegressor,
)
from conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
)
import numpy as np
def sample_random_arch_s():
    arch_embed_choices = [768, 384, 192]
    layer_choices = [10,11,12]
    mlp_ratio_choices = [2,3,4]
    head_choices = [4,8,12]
    bias_choices = [True, False]
    arch = {}
    arch["sample_embed_dim"] = np.random.choice(arch_embed_choices)
    arch["sample_n_layer"] = np.random.choice(layer_choices)
    arch["sample_mlp_ratio"] = np.random.choice(mlp_ratio_choices, arch["sample_n_layer"])
    arch["sample_n_head"] = np.random.choice(head_choices, arch["sample_n_layer"])
    arch["sample_bias"] = np.random.choice(bias_choices)
    return arch

def sample_random_arch_m():
    arch_embed_choices = [1024, 512, 256]
    layer_choices = [22,23,24]
    mlp_ratio_choices = [2,3,4]
    head_choices = [8,12,16]
    bias_choices = [True, False]
    arch = {}
    arch["sample_embed_dim"] = np.random.choice(arch_embed_choices)
    arch["sample_n_layer"] = np.random.choice(layer_choices)
    arch["sample_mlp_ratio"] = np.random.choice(mlp_ratio_choices, arch["sample_n_layer"])
    arch["sample_n_head"] = np.random.choice(head_choices, arch["sample_n_layer"])
    arch["sample_bias"] = np.random.choice(bias_choices)
    return arch
    
def sample_random_arch_l():
    arch_embed_choices = [320,640,1280]
    layer_choices = [34,35,36]
    mlp_ratio_choices = [2,3,4]
    head_choices = [8,16,20]
    bias_choices = [True, False]
    arch = {}
    arch["sample_embed_dim"] = np.random.choice(arch_embed_choices)
    arch["sample_n_layer"] = np.random.choice(layer_choices)
    arch["sample_mlp_ratio"] = np.random.choice(mlp_ratio_choices, arch["sample_n_layer"])
    arch["sample_n_head"] = np.random.choice(head_choices, arch["sample_n_layer"])
    arch["sample_bias"] = np.random.choice(bias_choices)
    return arch

def get_arch_feature_map_s(arch):
    arch_embed_choices = [768, 384, 192]
    layer_choices = [10,11,12]
    mlp_ratio_choices = [2,3,4]
    head_choices = [4,8,12]
    bias_choices = [True, False]
    # arch_feature_map 
    arch_feature_map = []
    arch_feature_map.append(arch["sample_embed_dim"])
    arch_feature_map.append(arch["sample_n_layer"])
    arch_feature_map.extend(arch["sample_mlp_ratio"][0:arch["sample_n_layer"]])
    for i in range(max(layer_choices)-arch["sample_n_layer"]):
        arch_feature_map.append(0)
    arch_feature_map.extend(arch["sample_n_head"][0:arch["sample_n_layer"]])
    for i in range(max(layer_choices)-arch["sample_n_layer"]):
        arch_feature_map.append(0)
    if arch["sample_bias"]:
        arch_feature_map.append(1)
    else:
        arch_feature_map.append(0)
    #print(len(arch_feature_map))
    return arch_feature_map

def get_arch_feature_map_m(arch):
    arch_embed_choices = [1024, 512, 256]
    layer_choices = [22,23,24]
    mlp_ratio_choices = [2,3,4]
    head_choices = [8,12,16]
    bias_choices = [True, False]
    # arch_feature_map 
    arch_feature_map = []
    arch_feature_map.append(arch["sample_embed_dim"])
    arch_feature_map.append(arch["sample_n_layer"])
    arch_feature_map.extend(arch["sample_mlp_ratio"][0:arch["sample_n_layer"]])
    for i in range(max(layer_choices)-arch["sample_n_layer"]):
        arch_feature_map.append(0)
    arch_feature_map.extend(arch["sample_n_head"][0:arch["sample_n_layer"]])
    for i in range(max(layer_choices)-arch["sample_n_layer"]):
        arch_feature_map.append(0)
    if arch["sample_bias"]:
        arch_feature_map.append(1)
    else:
        arch_feature_map.append(0)
    #print(len(arch_feature_map))
    return arch_feature_map

def get_arch_feature_map_l(arch):
    arch_embed_choices = [320,640,1280]
    layer_choices = [34,35,36]
    mlp_ratio_choices = [2,3,4]
    head_choices = [8,16,20]
    bias_choices = [True, False]
    # arch_feature_map 
    arch_feature_map = []
    arch_feature_map.append(arch["sample_embed_dim"])
    arch_feature_map.append(arch["sample_n_layer"])
    arch_feature_map.extend(arch["sample_mlp_ratio"][0:arch["sample_n_layer"]])
    for i in range(max(layer_choices)-arch["sample_n_layer"]):
        arch_feature_map.append(0)
    arch_feature_map.extend(arch["sample_n_head"][0:arch["sample_n_layer"]])
    for i in range(max(layer_choices)-arch["sample_n_layer"]):
        arch_feature_map.append(0)
    if arch["sample_bias"]:
        arch_feature_map.append(1)
    else:
        arch_feature_map.append(0)
    #print(len(arch_feature_map))
    return arch_feature_map

gpus = ["a100", "v100", "rtx2080", "rtx3080", "a6000", "a40", "P100", "h100"]
arch_preds = {}
for gpu in gpus:
  arch_preds[gpu] = []
  for i in range(100000):
    arch = sample_random_arch_s()
    arch_feature_map = get_arch_feature_map_s(arch)
    #predictor = GradientBoostingQuantileRegressor.load(
    #    f"hwmetric_predictor_ckpts/quantile_regressor_energy_{gpu}.pkl"
    #)
    with open(f"hwmetric_predictor_ckpts/conformal_quantile_regression_energy_{gpu}.pkl", "rb") as f:
        predictor = pickle.load(f)
    predictions = predictor.predict([arch_feature_map]).results_stacked
    print(predictions)
    arch_preds[gpu].append({"arch":arch, "predictions":predictions[0]})
  with open(f"hwmetric_predictor_ckpts/arch_preds_energy_{gpu}.pkl", "wb") as f:
    pickle.dump(arch_preds[gpu], f)
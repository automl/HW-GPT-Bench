import os
import pickle
from predictors.hwmetric.conformal.surrogate.quantile_regression_model import (
    QuantileRegressorPredictions,
    GradientBoostingQuantileRegressor,
)
from predictors.hwmetric.conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
)
import numpy as np


def get_arch_feature_map_s(arch):
    arch_embed_choices = [768, 384, 192]
    layer_choices = [10, 11, 12]
    mlp_ratio_choices = [2, 3, 4]
    head_choices = [4, 8, 12]
    bias_choices = [True, False]
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


def get_arch_feature_map_m(arch):
    arch_embed_choices = [1024, 512, 256]
    layer_choices = [22, 23, 24]
    mlp_ratio_choices = [2, 3, 4]
    head_choices = [8, 12, 16]
    bias_choices = [True, False]
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


def get_arch_feature_map_l(arch):
    arch_embed_choices = [320, 640, 1280]
    layer_choices = [34, 35, 36]
    mlp_ratio_choices = [2, 3, 4]
    head_choices = [8, 16, 20]
    bias_choices = [True, False]
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


def filter_archs(arch_list, observations):
    archs_unique = []
    for arch in arch_list:
        for obs in observations:
            if arch == obs["arch"]:
                archs_unique.append(obs)
                break
    return archs_unique


"""devices = ["P100","a6000", "rtx2080", "rtx3080", "v100", "a100", "a40", "h100"] # "cpu_mlgpu", "cpu_alldlc", "cpu_p100", "cpu_p100", "cpu_a6000", "cpu_meta"]
suffix = ["_m", "_l"]
for device in devices:
 for s in suffix:
  cat_list = []
  increment = 2500
  for i in range(0, 10000, increment):
   path = "raw_results_hw/latency_" + device + s + "/efficiency_energy_observations_tracker_" + str(i) + "_" + str(i+increment) + ".pkl"
   if not os.path.exists(path):
        continue
   with open(path,"rb") as f:
      a = pickle.load(f)
      cat_list.extend(a)
  a = cat_list
  arch_path = "sampled_archs"+s+".pkl"
  with open(arch_path,"rb") as f:
    arch_list = pickle.load(f)
  a = filter_archs(arch_list, a)
  if len(a) ==10000:
     print("Processing", device)
     arch_features = []
     latencies = []
     train_fraction = 0.8
     a = a[:int(train_fraction*len(a))]
     for i in range(len(a)):
        for j in range(len(a[i]["energy_gpu"])):
            if s == "":
                arch_features.append(get_arch_feature_map_s(a[i]["arch"]))
            elif s == "_m":
                arch_features.append(get_arch_feature_map_m(a[i]["arch"]))
            elif s == "_l":
                arch_features.append(get_arch_feature_map_l(a[i]["arch"]))
            latencies.append(a[i]["energy_gpu"][j])
            #print(arch_features[-1], latencies[-1])
     arch_features = np.array(arch_features)
     latencies = np.array(latencies)
     #print(arch_features.shape, latencies.shape)
     predictor = SymmetricConformalizedGradientBoostingQuantileRegressor(quantiles = 31)
     # standardize the features
     arch_features_mean = np.mean(arch_features, axis=0)
     arch_features_std = np.std(arch_features, axis=0)
     arch_features = (arch_features - arch_features_mean) / arch_features_std
     predictor.fit(arch_features, latencies)
     with open("hwmetric_predictor_ckpts/conformal_quantile_regression_energy_" + device + s + ".pkl","wb") as f:
            pickle.dump(predictor, f)
     predictor_gb = GradientBoostingQuantileRegressor(quantiles = 31)
     predictor_gb.fit(arch_features, latencies)
     with open("hwmetric_predictor_ckpts/quantile_regression_energy_" + device + s + ".pkl","wb") as f:
        pickle.dump(predictor_gb, f)"""

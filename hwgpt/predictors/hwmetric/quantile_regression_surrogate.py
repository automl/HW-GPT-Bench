from conformal.surrogate.quantile_regression_model import (
    QuantileRegressorPredictions,
    GradientBoostingQuantileRegressor,
)
from conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
)

predictor = SymmetricConformalizedGradientBoostingQuantileRegressor(quantiles=31)
import numpy as np
import pickle

i = 0
increment = 2500
cat_list = []
for i in range(0, 10000, increment):
    path = (
        "/work/dlclarge2/sukthank-hw-llm-bench/HW-Aware-LLM-Bench/latency_v100/efficiency_observations_"
        + str(i)
        + "_"
        + str(i + increment)
        + ".pkl"
    )
    with open(path, "rb") as f:
        a = pickle.load(f)
        cat_list.extend(a)
a = cat_list


def get_arch_feature_map(arch):
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
    print(len(arch_feature_map))
    return arch_feature_map


arch_features = []
latencies = []
for i in range(len(a)):
    for j in range(len(a[i]["times_profiler_gpu"])):
        arch_features.append(get_arch_feature_map(a[i]["arch"]))
        latencies.append(a[i]["times_profiler_gpu"][j])
        print(arch_features[-1], latencies[-1])
arch_features = np.array(arch_features)
latencies = np.array(latencies)
print(arch_features.shape, latencies.shape)
# standardize the features
arch_features = (arch_features - arch_features.mean(axis=0)) / arch_features.std(axis=0)
predictor.fit(arch_features, latencies)
out = predictor.predict(arch_features)
i = 10
print(arch_features[i * 10 : (i * 10 + 10)])
print(latencies[i * 10 : (i * 10 + 10)])
print(out.results_stacked[i * 10 : (i * 10 + 10), :])

# save the model
with open("latency_predictor.pkl", "wb") as f:
    pickle.dump(predictor, f)

# load the model
with open("latency_predictor.pkl", "rb") as f:
    predictor = pickle.load(f)

# predict
out = predictor.predict(arch_features)

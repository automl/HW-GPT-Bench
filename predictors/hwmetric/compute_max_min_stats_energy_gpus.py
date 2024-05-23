import torch
import pickle
from predictors.hwmetric.utils import get_arch_feature_map_s, get_arch_feature_map_m, get_arch_feature_map_l
from baselines.run_nas_lm_from_scratch_3_objs import get_predictor_surrogate_path_latency, predict_hw_surrogate
from gpt.utils import sample_config, sample_config_max, sample_config_min
from predictors.metric.utils import convert_config_to_one_hot, search_spaces
import numpy as np
devices_gpus = ["rtx2080"] # "rtx3080", "a6000", "a40", "P100", "h100"]
all_archs_max_min = []
for device in devices_gpus:
    all_archs_max_min = []
    predictor = get_predictor_surrogate_path_latency(12, "s", device, "conformal_quantile")
    arch = sample_config_max(search_spaces["s"])
    arch_feature_map = get_arch_feature_map_s(arch)
    with open("hwmetric_predictor_ckpts/mean_std_stats_archs.pkl","rb") as f:
        mean_std_stats = pickle.load(f)
    print(mean_std_stats)
    arch_feature_map = (arch_feature_map - mean_std_stats["mean"])/mean_std_stats["std"]
    #arch_feature_map = torch.tensor(arch_feature_map)#.unsqueeze(0).cuda()
    print(arch_feature_map)
    prediction = predict_hw_surrogate([arch_feature_map], predictor, "conformal_quantile", return_all=True)

    arch_min = sample_config_min(search_spaces["s"])
    arch_feature_map = get_arch_feature_map_s(arch_min)
    #arch_feature_map = torch.tensor(arch_feature_map)#.unsqueeze(0).cuda()
    arch_feature_map = (arch_feature_map - mean_std_stats["mean"])/mean_std_stats["std"]
    prediction_min = predict_hw_surrogate([arch_feature_map], predictor, "conformal_quantile", return_all=True)

    max_min_stats = {"max":np.max(prediction), "min":np.min(prediction_min)}
    print(max_min_stats)
    import pickle
    #with open("hwmetric_predictor_ckpts/max_min_stats_latency_"+device+".pkl", "wb") as f:
    #    pickle.dump(max_min_stats, f)

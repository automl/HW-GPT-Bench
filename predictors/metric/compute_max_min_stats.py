from predictors.metric.net import Net
from predictors.metric.utils import PPLDataset, search_spaces, convert_config_to_one_hot
from gpt.utils import sample_config, sample_config_max, sample_config_min
import torch
import pickle
import numpy as np
train_dataset = PPLDataset(search_space ="s", metric = "perplexity")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
net = Net(12,128).cuda()
net.load_state_dict(torch.load("ppl_predictor_ckpts/perplexity_s.pt"))
net.train()
out_list = []
for i in range(1):
        max_arch = sample_config_max(search_spaces["s"])
        max_arch_feature_map = convert_config_to_one_hot(max_arch, search_spaces["s"])
        max_arch_feature_map = max_arch_feature_map.unsqueeze(0).cuda()
        out = net(max_arch_feature_map)
        out_list.append(out.item())
        print(out.item())
        min_arch = sample_config_min(search_spaces["s"])
        min_arch_feature_map = convert_config_to_one_hot(min_arch, search_spaces["s"])
        min_arch_feature_map = min_arch_feature_map.unsqueeze(0).cuda()
        out = net(min_arch_feature_map)
        out_list.append(out.item())
        max_min_stats = {"max":max(out_list), "min":min(out_list)}
        print(max_min_stats)
max_min_save_path = "ppl_predictor_ckpts/max_min_stats_s.pkl"
max_min_stats = {"max":max(out_list), "min":min(out_list)}
with open(max_min_save_path, "wb") as f:
    pickle.dump(max_min_stats, f)
print(max_min_stats)

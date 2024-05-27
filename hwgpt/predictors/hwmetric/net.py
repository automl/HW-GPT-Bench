import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    The base model for MAML (Meta-SGD) for meta-NAS-predictor.
    """

    def __init__(self, num_layers, hw_embed_on, hw_embed_dim, layer_size):
        super(Net, self).__init__()
        self.layer_size = layer_size
        self.hw_embed_on = hw_embed_on
        nfeat = 3 + 2 * num_layers
        self.add_module("fc1", nn.Linear(nfeat, layer_size))
        self.add_module("fc2", nn.Linear(layer_size, layer_size))

        if hw_embed_on:
            self.add_module("fc_hw1", nn.Linear(hw_embed_dim, layer_size))
            self.add_module("fc_hw2", nn.Linear(layer_size, layer_size))
            hfeat = layer_size * 2
        else:
            hfeat = layer_size

        self.add_module("fc3", nn.Linear(hfeat, hfeat))
        self.add_module("fc4", nn.Linear(hfeat, hfeat))

        self.add_module("fc5", nn.Linear(hfeat, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, hw_embed=None):
        if self.hw_embed_on:
            hw_embed = hw_embed.repeat(len(x), 1)
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))

        if self.hw_embed_on:
            hw = self.relu(self.fc_hw1(hw_embed))
            hw = self.relu(self.fc_hw2(hw))
            out = torch.cat([out, hw], dim=-1)

        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        return out


# test
if __name__ == "__main__":
    from predictors.hwmetric.utils import convert_config_to_one_hot

    choices_dict = {
        "embed_dim_choices": [128, 256, 512],
        "n_layer_choices": [6, 8, 12],
        "mlp_ratio_choices": [1, 2, 4],
        "n_head_choices": [1, 2, 4],
        "bias_choices": [True, False],
    }
    sampled_config = {
        "sample_embed_dim": 128,
        "sample_n_layer": 6,
        "sample_mlp_ratio": [1, 2, 4, 1, 2, 4],
        "sample_n_head": [1, 2, 4, 1, 2, 4],
        "sample_bias": True,
    }
    one_hot_input = convert_config_to_one_hot(sampled_config, choices_dict)
    num_layers = max(choices_dict["n_layer_choices"])
    hw_embed_on = False
    hw_embed_dim = 128
    layer_size = 128
    net = Net(num_layers, hw_embed_on, hw_embed_dim, layer_size)
    out = net(one_hot_input.unsqueeze(0))
    print(out)

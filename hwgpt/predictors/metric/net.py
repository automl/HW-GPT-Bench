import torch
import torch.nn as nn


class Net(nn.Module):
    """
    The base model for MAML (Meta-SGD) for meta-NAS-predictor.
    """

    def __init__(self, num_layers: int, layer_size: int):
        super(Net, self).__init__()
        self.layer_size = layer_size
        nfeat = 6 + 6 * num_layers + 2
        self.add_module("fc1", nn.Linear(nfeat, layer_size))
        self.add_module("fc2", nn.Linear(layer_size, layer_size))

        hfeat = layer_size

        self.add_module("fc3", nn.Linear(hfeat, hfeat))
        self.add_module("fc4", nn.Linear(hfeat, hfeat))

        self.add_module("fc5", nn.Linear(hfeat, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))

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
    layer_size = 128
    net = Net(num_layers, layer_size)
    out = net(one_hot_input.unsqueeze(0))
    print(out)

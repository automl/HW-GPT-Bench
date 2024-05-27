import torch
import pickle
from lib.utils import convert_config_to_one_hot, convert_str_to_arch
from typing import Tuple


class PPLDataset(torch.utils.data.Dataset):
    "Dataset to load the hardware metrics data for training and testing"

    def __init__(
        self, search_space: str = "s", transform=None, metric: str = "perplexity"
    ):
        "Initialization"
        self.archs = []
        self.ppl = []
        self.metric = metric
        self.search_space = search_space
        self.load_data()

    def load_data(self):
        path = (
            "data_collection/gpt_datasets/gpt_" + str(self.search_space) + "/stats.pkl"
        )
        with open(path, "rb") as f:
            data = pickle.load(f)

        for arch in data:
            arch_config = convert_str_to_arch(arch)
            # print(arch_config)
            self.archs.append(convert_config_to_one_hot(arch_config, self.search_space))
            self.ppl.append(data[arch][self.metric])

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ppl)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        "Generates one sample of data"
        # Select sample
        one_hot = self.archs[index]
        metric = self.ppl[index]
        return one_hot, metric

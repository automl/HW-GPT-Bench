import torch
import torch.nn as nn
import torch.nn.functional as F


class LMHeadSuper(nn.Linear):
    def __init__(self, super_dim_in: int, output_dim: int, bias: bool = True):
        super().__init__(super_dim_in, output_dim)

        # the largest embed dim
        self.super_dim_in = super_dim_in
        self.output_dim = output_dim
        if bias is False:
            self.bias = None

        # the current sampled embed dim
        self.sample_dim_in = None

    def set_sample_config(self, sample_dim_in: int, sample_bias_flag: bool):
        self.sample_dim_in = sample_dim_in
        self.sample_bias_flag = sample_bias_flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return F.linear(x, self.weight[:, : self.sample_dim_in], self.bias)
        else:
            return F.linear(x, self.weight[:, : self.sample_dim_in])

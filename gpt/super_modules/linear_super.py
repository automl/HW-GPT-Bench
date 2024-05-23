import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperLinear(nn.Linear):
    def __init__(self, super_dim_in: int, super_dim_out: int):
        super().__init__(super_dim_in, super_dim_out)

        # the largest embed dim
        self.super_dim_in = super_dim_in
        self.super_dim_out = super_dim_out

        # the current sampled embed dim
        self.sample_dim_in = None
        self.sample_dim_out = None

    def set_sample_config(
        self, sample_dim_in: int, sample_dim_out: int, sample_bias_flag: bool
    ):
        self.sample_dim_in = sample_dim_in
        self.sample_dim_out = sample_dim_out
        self.sample_bias_flag = sample_bias_flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sample_bias_flag == True:
            # print(self.weight.shape)
            # print(self.weight[:self.sample_dim_out, :self.sample_dim_in].shape)
            # print(x.shape)
            # print(self.bias[:self.sample_dim_out].shape)
            # print(self.sample_dim_out)
            return F.linear(
                x,
                self.weight[: self.sample_dim_out, : self.sample_dim_in],
                self.bias[: self.sample_dim_out],
            )
        else:
            return F.linear(x, self.weight[: self.sample_dim_out, : self.sample_dim_in])

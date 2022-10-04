from typing import List, Callable

import torch
from torch import nn


class DefaultNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_layer_sizes: List[int], out_dims: List[int], activation_fn: Callable):
        super().__init__()

        fcs = []
        for i, n in enumerate(hidden_layer_sizes):
            fcs.append(nn.Linear(in_dim if i == 0 else hidden_layer_sizes[i - 1], n))
            fcs.append(activation_fn())
        self.fcs = nn.Sequential(*fcs)

        fc_outs = []
        for d in out_dims:
            fc_outs.append(nn.Linear(hidden_layer_sizes[-1], d))
        self.fc_outs = nn.ModuleList(fc_outs)

    def forward(self, x: torch.Tensor):
        assert 2 == len(x.shape)
        x = self.fcs(x)

        return [out(x) for out in self.fc_outs]


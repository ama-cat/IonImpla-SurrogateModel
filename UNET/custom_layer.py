import numpy as np
import torch
import torch.nn as nn


class WeightSkipConnection(nn.Module):
    def __init__(self):
        '''
        パラメータ:
            W: 重み
        '''
        super().__init__()
        self.W = nn.Parameter(torch.Tensor([np.random.normal()]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.W) * x
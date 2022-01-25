import numpy as np
import torch
import torch.nn as nn


class WeightSkipConnection(nn.Module):
    def __init__(self):
        '''
        引数:
            input_dim: 入力次元
            output_dim: 出力次元
            activation: 活性化関数
        パラメータ:
            W: 重み
            b: バイアス
        '''
        super().__init__()
        self.W = nn.Parameter(torch.Tensor([np.random.normal()]))

    def forward(self, x):
        return self.W * x
import torch
from torch import nn


class Laplacian(nn.Module):
    def __init__(self, input_channels, output_channels, padding=0):
        super().__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=padding)
        self.conv.state_dict()['weight'][:, :, 0, 0] = 0
        self.conv.state_dict()['weight'][:, :, 0, 2] = 0
        self.conv.state_dict()['weight'][:, :, 2, 0] = 0
        self.conv.state_dict()['weight'][:, :, 2, 2] = 0
        # self.conv.state_dict()['bias'][:, :, 0, 0] = 0
        # self.conv.state_dict()['bias'][:, :, 0, 2] = 0
        # self.conv.state_dict()['bias'][:, :, 2, 0] = 0
        # self.conv.state_dict()['bias'][:, :, 2, 2] = 0

                        

    def forward(self, x):
        x = self.conv(x)
        
        return x


def model_laplacian(model, keys):
    for key in keys:
        model.state_dict()[key][:, :, 0, 0] = 0
        model.state_dict()[key][:, :, 0, 2] = 0
        model.state_dict()[key][:, :, 2, 0] = 0
        model.state_dict()[key][:, :, 2, 2] = 0
        # model.state_dict()[key+".bias"][:, :, 0, 0] = 0
        # model.state_dict()[key+".bias"][:, :, 0, 2] = 0
        # model.state_dict()[key+".bias"][:, :, 2, 0] = 0
        # model.state_dict()[key+".bias"][:, :, 2, 2] = 0
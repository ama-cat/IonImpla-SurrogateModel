import torch
from torch import nn
from torch import optim

class DenseBlock(nn.Module):
    def __init__(self, input_channel, k, kernel_size=3, padding=1):
        super().__init__()
        
        self.k = int(k)
        self.input_channel = int(input_channel)
        
        self.conv1 = nn.Conv2d(self.input_channel, self.k, kernel_size=kernel_size, padding=padding) #3
        self.conv2 = nn.Conv2d(self.input_channel+self.k, self.k, kernel_size=kernel_size, padding=padding) #19
        self.conv3 = nn.Conv2d(self.input_channel+self.k*2, self.k, kernel_size=kernel_size, padding=padding) #35
        self.conv4 = nn.Conv2d(self.input_channel+self.k*3, self.k, kernel_size=kernel_size, padding=padding) #51
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x)) #16
        x = torch.cat([x, x1], dim=1) #16+1=17
        x2 = self.relu(self.conv2(x)) #17→16
        x = torch.cat([x, x2], dim=1) #16+17=33
        x3 = self.relu(self.conv3(x)) #33→16
        x = torch.cat([x, x3], dim=1) #16+33=49
        x4 = self.relu(self.conv4(x)) #49→16
        x = torch.cat([x4, x1, x2, x3], dim=1) #16×4=64
        
        return x #64


class ConvUnit(nn.Module):
    #in_channels=256, out_channels=8
    def __init__(self, input_channel, k):
        super().__init__()
        self.k = int(k)
        self.input_channel = int(input_channel)
        self.conv = nn.Conv2d(
            in_channels=input_channel,
            out_channels=self.k,
            kernel_size=3,
            padding=1
        )


    def forward(self, x):
        # x: [batch_size, in_channels=256, 20, 20]
        h = self.conv(x)
        # h: [batch_size, out_channels=8, 6, 6]
        return h



class ForDenseBlock(nn.Module):
    def __init__(self, first_input_channel=1, k=16, iter=4):
        super().__init__()
        self.first_input_channel = first_input_channel #DenseBlockの入力チャンネル数
        self.k = int(k) #成長率
        self.iter = iter #繰り返し数

        #畳み込みを順に定義
        def create_conv_unit(unit_idx):
                unit = ConvUnit(
                    input_channel=self.first_input_channel+self.k*unit_idx,
                    k = self.k
                )
                self.add_module("unit_"+ str(unit_idx), unit)
                return unit
    
        self.conv_units = [create_conv_unit(i) for i in range(self.iter)]
        
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x_list = []
        for conv in self.conv_units:
            x_conv = self.relu(conv(x)) #16
            x = torch.cat([x, x_conv], dim=1) #16+1=17
            x_list.append(x_conv)

        x = torch.cat(x_list, dim=1) #16×4=64

        return x

    
    
    
class TransitionDown(nn.Module):
    
    def __init__(self, input_channel):
        super().__init__()
        
        self.input_channel = input_channel
        
        #1×1畳み込み
        self.conv = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=1)
        #プーリング（サイズ1/2）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        
        return x
    
    
class TransitionUp(nn.Module):
    
    def __init__(self, input_channel):
        super().__init__()
        
        self.input_channel = input_channel
        self.output_channel = input_channel
        
        self.dconv = nn.ConvTranspose2d(self.input_channel, self.output_channel, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
    
    def forward(self, x):
        x = self.relu(self.dconv(x))
        
        return x



Aaaaaaaaaaaaaaaaaaaaaaaaa
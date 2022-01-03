import torch
from torch import nn
from torch import optim

from Block import DenseBlock, TransitionDown, TransitionUp



class FcDenseNet(nn.Module):
    def __init__(self, input_channel, k, kernel_size=3, padding=1):
        super().__init__()
        self.input_channel = int(input_channel)
        self.k = int(k)

        #44→40のサイズ調整
        self.first_conv = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=5)

        #DenseBlock1
        self.dense_block1 = DenseBlock(input_channel=self.input_channel, k=self.k, kernel_size=3, padding=1)
        self.down_trans1 = TransitionDown(input_channel=self.input_channel+self.k*4)

        #DenseBlock2
        self.dense_block2 = DenseBlock(input_channel=self.input_channel+self.k*4, k=self.k, kernel_size=3, padding=1)
        self.down_trans2 = TransitionDown(input_channel=self.input_channel+self.k*4*2)

        #DenseBlock3
        self.dense_block3 = DenseBlock(input_channel=self.input_channel+self.k*4*2, k=self.k, kernel_size=3, padding=1)
        self.down_trans3 = TransitionDown(input_channel=self.input_channel+self.k*4*3)

        #DenseBlock4
        self.dense_block4 = DenseBlock(input_channel=self.input_channel+self.k*4*3, k=self.k, kernel_size=3, padding=1)

        #DenseBlock5
        self.dense_block5 = DenseBlock(input_channel=self.input_channel+self.k*4*4, k=self.k)
        
        
        #DenseBlock6
        self.up_trans3 = TransitionUp(input_channel=self.input_channel+self.k*4*5)
        self.dense_block6 = DenseBlock(input_channel=self.input_channel+self.k*4*5, k=self.k)

        #DenseBlock7
        self.up_trans2 = TransitionUp(input_channel=self.input_channel+self.k*4*4)
        self.dense_block7 = DenseBlock(input_channel=self.input_channel+self.k*4*4, k=self.k)

        #DenseBlock8
        self.up_trans1 = TransitionUp(input_channel=self.input_channel+self.k*4*3)
        self.dense_block8 = DenseBlock(input_channel=self.input_channel+self.k*4*3, k=self.k)

        #40→44のサイズ調整
        self.last_conv = nn.Conv2d(self.input_channel+self.k*4*2, self.input_channel+self.k*4*2, kernel_size=3, padding=3)

        #1×1conv
        self.conv = nn.Conv2d(self.input_channel+self.k*4*2, 1, kernel_size=1)



    def forward(self, x):
        x = self.first_conv(x) #[N, 1, 40, 40]

        h1 = x #[N, 1, 40, 40]
        x = self.dense_block1(x) #[N, 64, 40, 40]
        x = torch.cat([h1, x], dim=1) #[N, 65, 40, 40]
        skip1 = x #[N, 65, 40, 40]
        x = self.down_trans1(x) #[N, 65, 20, 20]

        h2 = x #[N, 65, 20, 20]
        x = self.dense_block2(x) #[N, 64, 20, 20]
        x = torch.cat([h2, x], dim=1) #[N, 129, 20, 20]
        skip2 = x #[N, 129, 20, 20]
        x = self.down_trans2(x) #[N, 129, 10, 10]

        h3 = x #[N, 129, 10, 10]
        x = self.dense_block3(x) #[N, 64, 10, 10]
        x = torch.cat([h3, x], dim=1) #[N, 193, 10, 10]
        skip3 = x #[N, 193, 10, 10]
        x = self.down_trans3(x) #[N, 193, 5, 5]

        h4 = x #[N, 193, 5, 5]
        x = self.dense_block4(x) #[N, 64, 5, 5]
        x = torch.cat([h4, x], dim=1) #[N, 257, 5, 5]
        skip4 = x #[N, 257, 5, 5]

        """Up Sampling"""
        x = self.dense_block5(x) #[N, 64, 5, 5]
        #skip connection
        x = torch.cat([skip4, x], dim=1) #[N, 321, 5, 5]

        x = self.up_trans3(x) #[N, 321, 10, 10]
        x = self.dense_block6(x) #[N, 64, 10, 10]
        #skip connection
        x = torch.cat([skip3, x], dim=1) #[N, 257, 10, 10]

        x = self.up_trans2(x) #[N, 257, 20, 20]
        x = self.dense_block7(x) #[N, 64, 20, 20]
        #skip connection
        x = torch.cat([skip2, x], dim=1) #[N, 193, 20, 20]

        x = self.up_trans1(x) #[N, 193, 40, 40]
        x = self.dense_block8(x) #[N, 64, 40, 40]
        #skip connection
        x = torch.cat([skip1, x], dim=1) #[N, 129, 40, 40]

        x = self.last_conv(x) #[N, 129, 44, 44]

        x = self.conv(x) #[N, 1, 44, 44]
        return x
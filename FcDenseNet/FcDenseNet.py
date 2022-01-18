import torch
from torch import nn
from torch import optim

from model.FcDenseNet.Block import ForDenseBlock, TransitionDown, TransitionUp



class FcDenseNet(nn.Module):
    def __init__(self, input_channel, k=16, kernel_size=3, iter=4):
        super().__init__()
        self.input_channel = int(input_channel)
        self.k = int(k)
        self.iter = int(iter)

        #[3, 101, 201]→[3, 100, 200]のサイズ調整
        
        self.first_conv = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=4, padding=1)


        "down sampling"
        #ForDenseBlock1 
        self.dense_block1 = ForDenseBlock(input_channel=self.input_channel, k=self.k, iter=self.iter) #[3, 100, 200]→[67, 100, 200]
        self.down_trans1 = TransitionDown(input_channel=self.input_channel+self.k*self.iter) #[67, 100, 200]→[67, 50, 100]

        #ForDenseBlock2
        self.dense_block2 = ForDenseBlock(input_channel=self.input_channel+self.k*self.iter, k=self.k, iter=self.iter) #[67, 100, 200]→[131, 50, 100]
        self.down_trans2 = TransitionDown(input_channel=self.input_channel+self.k*self.iter*2) #[131, 50, 100]→[131, 25, 50]

        #ForDenseBlock3
        self.dense_block3 = ForDenseBlock(input_channel=self.input_channel+self.k*self.iter*2, k=self.k, iter=self.iter) #[131, 25, 50]→[195, 25, 50]

        "one D vector concatnate"
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 25*50)

        "up sampling"
        #ForDenseBlock4
        self.dense_block4 = ForDenseBlock(input_channel=self.input_channel+self.k*self.iter*3+1, k=self.k, iter=self.iter) #[196, 25, 50]→[64, 25, 50]
        #skip connection[64, 50, 25]→[259, 50, 25]

        #ForDenseBlock5
        self.up_trans2 = TransitionUp(input_channel=self.input_channel+self.k*self.iter*4) #[259, 25, 50]→[259, 50, 100]
        self.dense_block5 = ForDenseBlock(input_channel=self.input_channel+self.k*self.iter*4, k=self.k, iter=self.iter) #[259, 50, 100]→[64, 50, 100]
        #skip connection [64, 100, 50]→[195, 100, 50]
        
        
        #ForDenseBlock6
        self.up_trans1 = TransitionUp(input_channel=self.input_channel+self.k*self.iter*3) #[195, 50, 100]→[195, 100, 200]
        self.dense_block6 = ForDenseBlock(input_channel=self.input_channel+self.k*self.iter*3, k=self.k, iter=self.iter) #[195, 100, 200]→[64, 100, 200]
        #skip connection [64, 200, 100]→[131, 200, 100]


        #[131, 100, 200]→[64, 101, 201]のサイズ調整
        self.last_conv1 = nn.Conv2d(self.input_channel+self.k*self.iter*2, 64, kernel_size=4, padding=2)

        #1×1conv
        self.last_conv2 = nn.Conv2d(64, 1, kernel_size=1)



    def forward(self, twoD_input, oneD_input):
        x = twoD_input
        batch_size = x.shape[0]

        x = self.first_conv(x) #[N, 3, 200, 100]

        h1 = x #[N, 3, 200, 100]
        x = self.dense_block1(x) #[N, 64, 200, 100]
        x = torch.cat([h1, x], dim=1) #[N, 67, 200, 100]
        skip1 = x #[N, 67, 200, 100]
        x = self.down_trans1(x) #[N, 67, 100, 50]

        h2 = x #[N, 67, 100, 50]
        x = self.dense_block2(x) #[N, 64, 100, 50]
        x = torch.cat([h2, x], dim=1) #[N, 131, 100, 50]
        skip2 = x #[N, 131, 100, 50]
        x = self.down_trans2(x) #[N, 131, 50, 25]

        h3 = x #[N, 131, 50, 25]
        x = self.dense_block3(x) #[N, 64, 50, 25]
        x = torch.cat([h3, x], dim=1) #[N, 195, 50, 25]
        skip3 = x #[N, 195, 50, 25]

        #温度と時間結合
        concat = self.fc1(oneD_input)
        concat = self.fc2(concat)
        concat = concat.reshape(batch_size, -1, 25, 50)
        x = torch.cat((x, concat), dim=1) #二次元情報（temp, time)結合
        #[N, 196, 50, 25]

 
        """Up Sampling"""
        x = self.dense_block4(x) #[N, 64, 50, 25]
        #skip connection
        x = torch.cat([skip3, x], dim=1) #[N, 259, 50, 25]

        x = self.up_trans2(x) #[N, 259, 100, 50]
        x = self.dense_block5(x) #[N, 64, 100, 50]
        #skip connection
        x = torch.cat([skip2, x], dim=1) #[N, 195, 100, 50]

        x = self.up_trans1(x) #[N, 195, 200, 100]
        x = self.dense_block6(x) #[N, 64, 200, 100]
        #skip connection
        x = torch.cat([skip1, x], dim=1) #[N, 131, 200, 100]


        x = self.last_conv1(x) #[N, 64, 201, 101]

        x = self.last_conv2(x) #[N, 1, 201, 101]
        return x
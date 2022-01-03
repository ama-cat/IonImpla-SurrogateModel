import torch
from torch import nn
from torch import optim

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        """-----FCN-----"""
        #[3, 21, 41]
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=4, padding=1, stride=1)
        #[16, 20, 40]
        self.conv1_2 = nn.Conv2d(16, 16, 3)
        #[16, 18, 38]
        
        "pooling"
        #[16, 9, 19]
        
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=1)
        #[32, 8, 18]
        self.conv2_2 = nn.Conv2d(32, 32, 3)
        #[32, 6, 16]
        
        "pooling"
        #[32, 3, 8]
        
        self.conv3 = nn.Conv2d(32, 64, 3)
        #[64, 1, 6]
        
        "squeeze"
        #[64, 6]
        self.conv１d = nn.Conv1d(64, 64, kernel_size=6, padding=0, stride=1)
        #[64, 1]
        "flatten"
        #[64]
        "concate"
        #[64+2]
        
        self.fc1 = nn.Linear(66, 32)
        #[32]
        self.fc2 = nn.Linear(32, 24)
        #[24]
        "reshape"
        #[1, 3, 8]
        
        """-----逆畳み込み-----"""
        self.dconv1 = nn.ConvTranspose2d(1, 32, kernel_size=2, stride=2)
        #[32, 6, 16]
        
        "skip connection(32→64)"
        #[64, 6, 16]
        
        self.conv4_1 = nn.Conv2d(64, 32, kernel_size=3, padding=2)
        #[32, 8, 18]
        self.conv4_2 = nn.Conv2d(32, 32, kernel_size=4, padding=2)
        #[32, 9, 19]
        
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        #[16, 18, 38]
        
        "skip connection(16→32)"
        #[32, 18, 38]
        self.conv5_1 = nn.Conv2d(32, 16, kernel_size=3, padding=2)
        #[16, 20, 40]
        self.conv5_2 = nn.Conv2d(16, 16, kernel_size=4, padding=2)
        #[16, 21, 41]
        
        #1×1畳み込み
        self.conv6 = nn.Conv2d(16, 1, kernel_size=1)
        #[1, 21, 41]
        
        #プーリングと活性化関数
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        #誤差関数と最適化手法
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)
        
    def forward(self, twoD_input, oneD_input):
        h = twoD_input
        "圧縮"
        #[3, 21, 41]
        h = self.relu(self.conv1_1(h))
        #[16, 20, 40]
        output1 = self.relu(self.conv1_2(h))
        #[16, 18, 38]
        
        "skip connection前の画像出力1"
        self.skip1 = output1 #16枚の画像
        
        h = self.pool(output1)
        #[16, 9, 19]
        
        h = self.relu(self.conv2_1(h))
        #[32, 8, 18]
        output2 = self.relu(self.conv2_2(h))
        #[32, 6, 16]
        
        "skip connection前の画像出力2"
        self.skip2 = output2 #32枚の画像
        
        h = self.pool(output2)
        #[32, 3, 8]
        
        h = self.relu(self.conv3(h))
        #[64, 1, 6]
        
        h = torch.squeeze(h)
        #[64, 6]
        
        h = self.relu(self.conv1d(h))
        #[64, 1]
        output3 = torch.squeeze(h)
        #[64]
        
        "一次元結合"
        #[64]
        h = torch.cat((output3, oneD_input), dim=1) #一次元情報（temp, time)結合
        #[66]
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        #[24]
        h = h.reshape(-1, 1, 3, 8)
        #[1, 3, 8]
        
        
        "復元"
        upsample1 = self.relu(self.dconv1(h))
        #[32, 6, 16]
        
        "復元後の画像出力1"
        self.encode1 = upsample1
        
        h = torch.cat((output2, upsample1), dim=1) #チャンネル方向に結合
        #[64, 6, 16]
        
        h = self.relu(self.conv4_1(h))
        #[32, 8, 18]
        "足し合わせ特徴マップの畳み込み後の出力1_1"
        self.encode1_1 = h
        
        h = self.relu(self.conv4_2(h))
        #[32, 9, 19]
        "足し合わせ特徴マップの畳み込み後の出力1_2"
        self.encode1_2 = h
        
        
        upsample2 = self.relu(self.dconv2(h))
        #[16, 18, 38]
        "復元後の画像出力2"
        self.encode2 = upsample2
        
        h = torch.cat((output1, upsample2), dim=1) #チャンネル方向に結合
        #[32, 18, 38]
        
        h = self.relu(self.conv5_1(h))
        #[16, 20, 40]
        "足し合わせ特徴マップの畳み込み後の出力2_1"
        self.encode2_1 = h
        
        h = self.relu(self.conv5_2(h))
        #[16, 21, 41]
        
        "足し合わせ特徴マップの畳み込み後の出力2_2"
        self.encode2_2 = h
        
        h = self.conv6(h)
        #[1, 21, 41]
        
        return h
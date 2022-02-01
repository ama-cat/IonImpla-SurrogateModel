import torch
from torch import nn
from torch import optim
from model.UNET.laplacian import Laplacian, model_laplacian

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        """-----FCN-----"""
        #[3, 101, 201]
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=4, padding=1, stride=1) #[16, 100, 200]
        self.conv1_2 = Laplacian(16, 16) #[16, 98, 198]
        self.bn1 = nn.BatchNorm2d(16)
        

        "pooling" #[16, 49, 99]
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=4, padding=1) #[32, 48, 98]
        self.conv2_2 = Laplacian(32, 32) #[32, 46, 96]
        self.bn2 = nn.BatchNorm2d(32)


        

        "pooling" #[32, 23, 48]
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=(3, 4)) #[64, 21, 45]
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=4, padding=1) #[64, 20, 44]
        self.bn3 = nn.BatchNorm2d(64)



        "pooling" #[64, 10, 22]
        self.conv4_1 = Laplacian(64, 128) #[128, 8, 20]
        self.conv4_2 = Laplacian(128, 128) #[128, 6, 18]
        self.bn4 = nn.BatchNorm2d(128)




        """1次元変換"""
        "pooling" #[128, 3, 9]
        self.conv5_1 = Laplacian(128, 256) #[256, 1, 7]
        self.bn5 = nn.BatchNorm2d(256)

        "squeeze" #[256, 7]

        self.conv1d = nn.Conv1d(256, 256, kernel_size=7, padding=0, stride=1) #[256, 1]
        "flatten" #[256]
        "concate" #[256+2]
        
        self.fc1 = nn.Linear(258, 864) #[864]
        "reshape" #[32, 3, 9]
        
        """-----逆畳み込み-----"""
        self.dconv4 = nn.ConvTranspose2d(32, 128, kernel_size=2, stride=2) #[128, 6, 18]
        

        "skip connection(128→256)" #[256, 6, 18]
        self.conv6_1 = Laplacian(256, 128, padding=2) #[128, 8, 20]
        self.conv6_2 = Laplacian(128, 128, padding=2) #[128, 10, 22]
        self.bn6 = nn.BatchNorm2d(128)
        self.dconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) #[64, 20, 44]
        

        "skip connection(64→128)" #[128, 20, 44]
        self.conv7_1 = nn.Conv2d(128, 64, kernel_size=4, padding=2) #[64, 21, 45]
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=(5, 4), padding=3) #[64, 23, 48]
        self.bn7 = nn.BatchNorm2d(64)
        self.dconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) #[32, 46, 96]


        "skip connection(32→64)" #[64, 46, 96]
        self.conv8_1 = Laplacian(64, 32, padding=2) #[32, 48, 98]
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=4, padding=2) #[32, 49, 99]
        self.bn8 = nn.BatchNorm2d(32)
        self.dconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) #[16, 98, 198]


        "skip connection(16→32)" #[32, 98, 198]
        self.conv9_1 = Laplacian(32, 16, padding=2) #[16, 100, 200]
        self.conv9_2 = nn.Conv2d(16, 16, kernel_size=4, padding=2) #[16, 101, 201]
        self.bn9 = nn.BatchNorm2d(16)


        #1×1畳み込み
        self.conv10 = nn.Conv2d(16, 1, kernel_size=1) #[1, 101, 201]


        
        #プーリングと活性化関数
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        #誤差関数と最適化手法
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)


        
    def forward(self, twoD_input, oneD_input):
        h = twoD_input
        "圧縮"
        #[3, 101, 201]
        h = self.relu(self.conv1_1(h)) #[16, 100, 200]
        output1 = self.relu(self.bn1(self.conv1_2(h))) #[16, 98, 198]


        h = self.pool(output1) #[16, 49, 99]
        h = self.relu(self.conv2_1(h)) #[32, 48, 98]
        output2 = self.relu(self.bn2(self.conv2_2(h))) #[32, 46, 96]


        h = self.pool(output2) #[32, 23, 48]
        h = self.relu(self.conv3_1(h)) #[64, 21, 45]
        output3 = self.relu(self.bn3(self.conv3_2(h))) #[64, 20, 44]


        h = self.pool(output3) #[64, 10, 22]
        h = self.relu(self.conv4_1(h)) #[128, 8, 20]
        output4 = self.relu(self.bn4(self.conv4_2(h))) #[128, 6, 18]



        h = self.pool(output4) #[128, 3, 9]
        h = self.relu(self.bn5(self.conv5_1(h))) #[256, 1, 7]

        """---一次元変換---"""

        h = torch.squeeze(h) #[256, 7]
        
        h = self.relu(self.conv1d(h)) #[256, 1]
        output5 = torch.squeeze(h) #[256]
        
        "一次元結合"
        h = torch.cat((output5, oneD_input), dim=1) #一次元情報（temp, time)結合 [258]
        h = self.relu(self.fc1(h)) #[864]
        h = h.reshape(-1, 32, 3, 9) #[32, 3, 9]

        """------"""
        
        
        "復元"
        upsample4 = self.relu(self.dconv4(h)) #[128, 6, 18]
        
        h = torch.cat((output4, upsample4), dim=1) #チャンネル方向に結合 [256, 6, 18]
        
        h = self.relu(self.conv6_1(h)) #[128, 8, 20]
        h = self.relu(self.bn6(self.conv6_2(h))) #[128, 10, 22]
        
        upsample3 = self.relu(self.dconv3(h)) #[64, 20, 44]

        h = torch.cat((output3, upsample3), dim=1) #チャンネル方向に結合 [128, 20, 44]
        
        h = self.relu(self.conv7_1(h)) #[64, 21, 45]
        h = self.relu(self.bn7(self.conv7_2(h))) #[64, 23, 48]
        
        upsample2 = self.relu(self.dconv2(h)) #[32, 46, 96]

        h = torch.cat((output2, upsample2), dim=1) #チャンネル方向に結合 [64, 46, 96]

        h = self.relu(self.conv8_1(h)) #[32, 47, 97]
        h = self.relu(self.bn8(self.conv8_2(h))) #[32, 49, 99]
        
        upsample1 = self.relu(self.dconv1(h)) #[16, 98, 198]

        h = torch.cat((output1, upsample1), dim=1) #チャンネル方向に結合 [32, 98, 198]

        h = self.relu(self.conv9_1(h)) #[16, 100, 200]
        h = self.relu(self.bn9(self.conv9_2(h))) #[16, 101, 201]

        h = self.relu(self.conv10(h)) #[1, 101, 201]
        
        return h
import torch
from torch import nn



class CustomLoss(nn.Module):
    def __init__(self, k):
        super().__init__()
        # パラメータを設定 
        
        self.k = k #罰則率
        self.criterion = nn.MSELoss()

    def forward(self, y_pred, y_truth):
        '''隣合うピクセルの差を考慮した誤差関数

        Parameters
        ------------------
        y_pred : モデルの出力
        y_truth : シミュレーションデータ
        '''
        img_height = 101
        img_width = 201
        

        left_and_up = y_pred[:, :, :img_height - 1, :img_width - 1] #左上
        left_and_down = y_pred[:, :, 1:, :img_width - 1] #左下
        right_and_up = y_pred[:, :, :img_height - 1, 1:] #右上
        
        return (self.criterion(left_and_up, left_and_down) + self.criterion(left_and_up, right_and_up))*self.k + self.criterion(y_pred, y_truth)
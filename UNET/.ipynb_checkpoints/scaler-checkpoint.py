import torch
import numpy as np

#標準化クラス
class Normalize1D():
    
    def __init__(self, train):
        self.train = train
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0)

    def scaling(self, inputs):
        return (inputs - self.mean) / self.std

    def inverse(self, inputs):
        return self.std * inputs + self.mean
    

class Normalize2D():
    
    def __init__(self, train):
        self.train = train
        self.mean = train.permute([1, 0, 2, 3]).reshape(3, -1).mean(axis=1) #チャンネルごとの平均
        self.std = train.permute([1, 0, 2, 3]).reshape(3, -1).std(axis=1) #チャンネルごとの標準偏差

    def scaling(self, inputs):
        return_data = []
        #チャンネルごとに変換
        for i in range(self.train.shape[1]): #チャンネル数反復
            data = (inputs[:, i, :, :] - self.mean[i]) / self.std[i] #変換
            data = data.unsqueeze(0) #[N, H, W]→[C=1, N, H, W]
            return_data.append(data)
        #結合
        return_data = torch.cat(return_data, axis=0) #C方向
        return_data = return_data.permute([1, 0, 2, 3]) #[C, N, H, W]→[N, C, H, W]
        return return_data

    def inverse(self, inputs):
        return_data = []
        #チャンネルごとに逆変換
        for i in range(self.train.shape[1]): #チャンネル数反復
            data = self.std[i] * inputs[:, i, :, :] + self.mean[i] #逆変換
            data = data.unsqueeze(0) #[N, H, W]→[C=1, N, H, W]
            return_data.append(data)
        #結合
        return_data = torch.cat(return_data, axis=0) #C方向
        return_data = return_data.permute([1, 0, 2, 3]) #[C, N, H, W]→[N, C, H, W]
        
        return return_data
    

#正規化クラス
class Standardize1D():
    
    def __init__(self, train):
        self.train = train
        self.max = train.max(axis=0)[0]

    def scaling(self, inputs):
        return inputs/self.max

    def inverse(self, inputs):
        return inputs*self.max


class Standardize2D():
    
    def __init__(self, train):
        self.train = train
        self.max = train.permute([1, 0, 2, 3]).reshape(3, -1).max(axis=1)[0] #チャンネルごとの最大値
        
    def scaling(self, inputs):
        return_data = []
        #チャンネルごとに変換
        for i in range(self.train.shape[1]): #チャンネル数反復
            data = inputs[:, i, :, :] / self.max[i] #変換
            data = data.unsqueeze(0) #[N, H, W]→[C=1, N, H, W]
            return_data.append(data)
        #結合
        return_data = torch.cat(return_data, axis=0) #C方向
        return_data = return_data.permute([1, 0, 2, 3]) #[C, N, H, W]→[N, C, H, W]
        return return_data
    
    def inverse(self, inputs):
        return_data = []
        #チャンネルごとに逆変換
        for i in range(self.train.shape[1]): #チャンネル数反復
            data = inputs[:, i, :, :]*self.max[i] #逆変換
            data = data.unsqueeze(0) #[N, H, W]→[C=1, N, H, W]
            return_data.append(data)
        #結合
        return_data = torch.cat(return_data, axis=0) #C方向
        return_data = return_data.permute([1, 0, 2, 3]) #[C, N, H, W]→[N, C, H, W]
        return return_data



class NorimalizeDist():
    
    def __init__(self, train):
        self.train = train
        self.mean = train.permute([1, 0, 2, 3]).reshape(3, -1).mean(axis=1) #チャンネルごとの平均
        self.std = train.permute([1, 0, 2, 3]).reshape(3, -1).std(axis=1) #チャンネルごとの標準偏差
        
        

        
        


class LogScaler():
    
    def scaling(self, inputs):
        data = inputs + 1
        self.log_data = np.log10(data)
        
    def inverse(self, outputs):
        data = 10**(outputs)
        self.former_data = data - 1
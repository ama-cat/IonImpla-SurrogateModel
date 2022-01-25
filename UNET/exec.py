"/work/data/sony/"


import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchmetrics import R2Score
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split, Subset, TensorDataset
from joblib import dump, load

#自作python file
from model.UNET.scaler import Standardize1D, Standardize2D, LogScaler
from model.UNET.UNET import Unet



filename = os.path.basename(__file__)

def mape(pred, target):
    return ((pred-target).abs() / (target.abs()+0.01)).mean()


if torch.cuda.is_available():
    gpu = torch.device("cuda") #gpuデバイスオブジェクト作成
    cpu = torch.device("cpu") #cpuデバイスオブジェクト作成
else:
    print("gpu環境が整っていません")

print(os.getcwd())

#tensorboard設定
writer1 = SummaryWriter(log_dir="/work/log/train")
writer2 = SummaryWriter(log_dir="/work/log/valid")


r2score = R2Score().to(gpu)



"""========================="""
#data読み込み（pyファイル読み込み時に自動実行）
data_2d = np.load("0119/data/dist_data.npy")
data_num = len(data_2d)
log_scaler = LogScaler()
log_scaler.scaling(data_2d) #ログデータ生成（self.log_data)
input_2d_data = log_scaler.log_data[:, :3, :, :]
input_temp_data = np.load("0119/data/temperature.npy")[:data_num]
input_time_data = np.load("0119/data/time.npy")[:data_num]
input_1d_data = np.stack([input_temp_data, input_time_data], axis=1)
output_data = log_scaler.log_data[:, 3:, :, :]
print(input_2d_data.shape)
print(input_1d_data.shape)
print(output_data.shape)

#学習とテスト
oneD_x_train, oneD_x_test, twoD_x_train, twoD_x_test, y_train, y_test = train_test_split(input_1d_data, input_2d_data, output_data, test_size=int(data_num*0.1), random_state=0)
#学習と検証
oneD_x_train, oneD_x_val, twoD_x_train, twoD_x_val, y_train, y_val = train_test_split(oneD_x_train, twoD_x_train, y_train, random_state=0, test_size=int(data_num*0.1))


#numpy→tensor
oneD_x_train_tensor = torch.Tensor(oneD_x_train)
twoD_x_train_tensor = torch.Tensor(twoD_x_train)

oneD_x_val_tensor = torch.Tensor(oneD_x_val)
twoD_x_val_tensor = torch.Tensor(twoD_x_val)

oneD_x_test_tensor = torch.Tensor(oneD_x_test)
twoD_x_test_tensor = torch.Tensor(twoD_x_test)

y_train_tensor = torch.Tensor(y_train)
y_val_tensor = torch.Tensor(y_val)
y_test_tensor = torch.Tensor(y_test)


#正規化
scaler1d_in = Standardize1D(oneD_x_train_tensor)
scaler2d_in = Standardize2D(twoD_x_train_tensor)
scaler_out = Standardize2D(y_train_tensor)

oneD_x_train_tensor_scaled = scaler1d_in.scaling(oneD_x_train_tensor)
twoD_x_train_tensor_scaled = scaler2d_in.scaling(twoD_x_train_tensor)

oneD_x_val_tensor_scaled = scaler1d_in.scaling(oneD_x_val_tensor)
twoD_x_val_tensor_scaled = scaler2d_in.scaling(twoD_x_val_tensor)

oneD_x_test_tensor_scaled = scaler1d_in.scaling(oneD_x_test_tensor)
twoD_x_test_tensor_scaled = scaler2d_in.scaling(twoD_x_test_tensor)

y_train_tensor_scaled = scaler_out.scaling(y_train_tensor)
y_val_tensor_scaled = scaler_out.scaling(y_val_tensor)
y_test_tensor_scaled = scaler_out.scaling(y_test_tensor)

ds_train = TensorDataset(twoD_x_train_tensor_scaled, oneD_x_train_tensor_scaled, y_train_tensor_scaled)
ds_val = TensorDataset(twoD_x_val_tensor_scaled, oneD_x_val_tensor_scaled, y_val_tensor_scaled)
ds_test = TensorDataset(twoD_x_test_tensor_scaled, oneD_x_test_tensor_scaled, y_test_tensor_scaled)


test_loader = DataLoader(ds_test, batch_size=4, drop_last=True)
"""============================"""


#訓練関数
def train(model, train_loader, criterion, optimizer):
    #学習時明記コード
    model.train()
    
    ###追記部分1###
    #損失の合計、全体のデータ数を数えるカウンターの初期化
    total_loss = 0
    batch_num = 0
    total_data_len = 0 #結局データサイズになる（mnistだったら60000）
    ### ###
    
    #ミニバッチごとにループさせる, train_loaderの中身を出し切ったら1エポックとなる
    for twoD_input, oneD_input, truth in train_loader: #train_loaderの個数　= データ数/バッチサイズ
        twoD_input, oneD_input, truth = twoD_input.to(gpu), oneD_input.to(gpu), truth.to(gpu)
        output = model(twoD_input, oneD_input) #順伝播 #shape = [batch_size, 1]
        optimizer.zero_grad() #勾配を初期化 
        loss = criterion(output, truth) #損失を計算 数値で算出（batch_size*4の平均）
        loss.backward() #逆伝播で勾配を計算
        optimizer.step() #最適化
        
        ###追記部分2###
        batch_size = len(truth) #バッチサイズ確認
        total_loss += loss.item() #全損失の合計
        total_data_len += batch_size
        
        #r2スコア
        batch_num += 1
        
    #今回のエポックの正答率と損失を求める
    avg_loss = total_loss/batch_num #平均損失算出
    return avg_loss
    ### ###

    
#検証関数
def valid(model, val_loader, criterion, y_val_tensor_scaled, twoD_x_val_tensor_scaled, oneD_x_val_tensor_scaled):
    #検証時明記コード
    model.eval()
    
    ###追記部分1###
    #損失の合計、全体のデータ数を数えるカウンターの初期化
    total_loss = 0
    batch_num = 0 #テストデータ数/バッチサイズ
    total_data_len = 0 #結局データサイズになる（mnistだったら60000）
    ### ###
    
    #ミニバッチごとにループさせる, train_loaderの中身を出し切ったら1エポックとなる
    for twoD_input, oneD_input, truth in val_loader: #train_loaderの個数　= データ数/バッチサイズ
        twoD_input, oneD_input, truth = twoD_input.to(gpu), oneD_input.to(gpu), truth.to(gpu)
        output = model(twoD_input, oneD_input) #順伝播
        loss = criterion(output, truth) #損失を計算
        
        ###追記部分2###
        batch_size = len(truth) #バッチサイズ確認
        total_loss += loss.item() #全損失の合計
        total_data_len += batch_size
        batch_num += 1
        
    #平均損失、スコアを算出
    avg_loss = total_loss/batch_num

    #r2スコア算出
    score = r2score(model(twoD_x_val_tensor_scaled.to(gpu), oneD_x_val_tensor_scaled.to(gpu)).flatten(), y_val_tensor_scaled.to(gpu).flatten())

    return avg_loss, score.item()


    
def tuning(config, epoch, checkpoint_dir=None):

    #tensorboard設定
    writer1 = SummaryWriter(log_dir="/work/log/train")
    writer2 = SummaryWriter(log_dir="/work/log/valid")

    model = Unet()
    model_name = model.__class__.__name__

    model.to(gpu)

    criterion = config["criterion"]
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])


    # バッチ分割
    train_loader = DataLoader(ds_train, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=config["batch_size"], drop_last=False)
    
    
    score_list = [0.0]

    for i in range(epoch):  # データセットに対して複数回ループ
        running_loss = 0.0
        epoch_steps = 0
        #学習
        train_loss = train(model, train_loader, criterion, optimizer)
        
        #検証
        val_loss, val_score = valid(model, val_loader, criterion, y_val_tensor_scaled, twoD_x_val_tensor_scaled, oneD_x_val_tensor_scaled)
        
        #表示
        print(i+1, "train_loss: ", train_loss, "val_loss: ", val_loss, "val_score: ", val_score)
        
        #可視化
        writer1.add_scalar(tag="MSELoss", scalar_value=train_loss, global_step=i+1)
        writer2.add_scalar(tag="MSELoss", scalar_value=val_loss, global_step=i+1)
        writer2.add_scalar(tag="R2score", scalar_value=val_score, global_step=i+1)
        
        if val_score > max(score_list):
            score_list.append(val_score)
            model_path = "0119/code/model.pth"
            torch.save(model.state_dict(), model_path)
            opt_score = val_score
            
    os.rename(model_path, "{0}_{1:.3f}.pth".format(model_name, opt_score))
        
    print("Finished Training")
    writer1.close()
    writer2.close()
    return opt_score
    


        
    
if __name__ == "__main__":
    #例
    config = {"lr": 0.001, "batch_size": 16}
    tuning(config, 30)
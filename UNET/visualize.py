import torch
import matplotlib.pyplot as plt
import numpy as np
from scaler import LogScaler
import data.sony.model.UNET.exec as exec
# from classUnet import Unet


#skip connection1回目の前
def visualize(data, n):
    fig, axes = plt.subplots(4, 4, figsize=(10, 6))
    for i, ax in zip(range(16), axes.ravel()):
        ax.imshow(data.detach().numpy()[n][i])
        ax.set_title(f"{i}", fontsize=20)
        ax.axis("off")
    fig.tight_layout()

    
#最終出力
def last_visualize(n, sim, pred):
    fig, axes=plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(sim[n, 0, :, :], vmin=0, vmax=18)
    axes[0].set_title("Sim", fontsize=30)
    axes[0].axis("off")
    axes[1].imshow(pred.detach().numpy()[n, 0, :, :], vmin=0, vmax=18)
    axes[1].set_title("ML", fontsize=30)
    axes[1].axis("off")
    print(exec.oneD_x_test_tensor[n])
    plt.show()
    
    
#横切りプロット
def horizontal_plot(n, y, sim, pred, low, up):
#     fig, axes=plt.subplots(1, 2, figsize=(20, 5))
    fig = plt.figure(figsize=(8, 4))
    twoD_pred_data = pred.detach().numpy()[n, 0, :, :]
    twoD_sim_data = sim[n, 0, :, :]
    x = np.arange(0, twoD_pred_data.shape[1])
    plt.plot(x, twoD_sim_data[y, :], label="sim", c="r")
    plt.plot(x, twoD_pred_data[y, :], label="ML", linestyle="--", c="cyan")
    plt.ylim(low, up)
    plt.legend()
    
    
#縦切りプロット
def vertical_plot(n, x, sim, pred, low, up):
#     fig, axes=plt.subplots(1, 2, figsize=(7, 5))
    fig = plt.figure(figsize=(5, 4))
    twoD_pred_data = pred.detach().numpy()[n, 0, :, :]
    twoD_sim_data = sim[n, 0, :, :]
    y = np.arange(0, twoD_pred_data.shape[0])
    plt.plot(y, twoD_sim_data[:, x], label="sim", c="r")
    plt.plot(y, twoD_pred_data[:, x], label="ML", linestyle="--", c="cyan")
    plt.ylim(low, up)
    plt.legend()
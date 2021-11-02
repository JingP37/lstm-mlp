# -*- coding: utf-8 -*-
"""
Created on 2020/7

@author: panj
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import time
import copy
import scipy.io as sio


class STA_DNN(nn.Module):
    def __init__(self):
        super(STA_DNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(104, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, 96)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def calc_nmse(pred, target):
    mse = np.sum(((pred - target)/target)**2) / pred.size
    return mse


if __name__=='__main__':
    # Hyper Parameters
    EPOCH = 800
    BATCH_SIZE = 128
    LR = 0.001            # learning rate
    initial = False
    train_test_ratio = 0.8
    r_cnt = 1
    clip =1e-4

    p_index = [5, 19, 32, 46, 57, 71, 84, 98]
    non_empty_index = np.concatenate((range(6, 32), range(33, 59), range(70, 96), range(97, 123)), axis=0)  # 非空子载波索引
    d_index = np.delete(non_empty_index, p_index)

    # 下载数据
    input_data = np.load('../train channel/STADNN_input_30.npy')
    label_data = np.load('../train channel/STADNN_out_30.npy')
    print(input_data.shape)

    # 处理数据
    input_data = input_data[:, non_empty_index]
    label_data = label_data[:, d_index]

    # 数据数量
    nums = int(input_data.shape[0])
    train_nums = int(train_test_ratio * nums)
    val_nums = nums - train_nums
    print('dataset size: ', nums, ', train set size: ', train_nums, ', val set size: ', val_nums)

    # 归一化处理
    scaler = StandardScaler()
    input_data_scaler = scaler.fit_transform(input_data.reshape(-1, 2)).reshape(input_data.shape)
    label_data_scaler = scaler.fit_transform(label_data.reshape(-1, 2)).reshape(label_data.shape)

    # 训练集验证集分配
    train_input = input_data_scaler[:train_nums]
    train_label = label_data_scaler[:train_nums]
    train_input = torch.from_numpy(train_input).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.FloatTensor)

    val_input = input_data_scaler[-val_nums:]
    val_label = label_data_scaler[-val_nums:]
    val_input = torch.from_numpy(val_input).type(torch.FloatTensor)
    val_label = torch.from_numpy(val_label).type(torch.FloatTensor)
    # ----------------------------- load model ---------------------------- #
    dir_name = './model_sta_dnn'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if initial is False:
        # 加载训练数据
        # Data Loader for easy mini-batch return in training
        dataset = Data.TensorDataset(train_input, train_label)
        train_loader = Data.DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8 if torch.cuda.is_available() else 0)
        r_min_err = float('inf')
        for r in range(r_cnt):
            sta_dnn = STA_DNN().to(device)

            optimizer = torch.optim.Adam(sta_dnn.parameters(), lr=LR)
            loss_func = nn.MSELoss()

            LOSS_TRAIN = []
            LOSS_VAL = []
            nmse_train = []
            nmse_val = []
            STEP = 0

            min_err = float('inf')
            time_train = 0

            for epoch in range(EPOCH):
                start = time.time()
                with torch.set_grad_enabled(True):
                    sta_dnn.train()
                    for step, (train_batch, label_batch) in enumerate(train_loader):
                        encoded, decoded = sta_dnn(train_batch.to(device))

                        loss_train = loss_func(decoded, label_batch.to(device))      # mean square error
                        optimizer.zero_grad()               # clear gradients for this training step
                        loss_train.backward()                     # backpropagation, compute gradients
                        nn.utils.clip_grad_norm_(sta_dnn.parameters(), clip)
                        optimizer.step()                    # apply gradients

                        nmse = calc_nmse(decoded.detach().cpu().numpy(), label_batch.detach().cpu().numpy())
                        if step % 1000 == 0:
                            print('Epoch: ', epoch, '| loss_train:', loss_train.item())
                            nmse_train.append(nmse)
                            LOSS_TRAIN.append(loss_train.item())
                time_train += time.time() - start
                with torch.set_grad_enabled(False):
                    sta_dnn.eval()
                    _, decoded = sta_dnn(val_input.to(device))
                    loss_val = loss_func(decoded, val_label.to(device))
                    nmse = calc_nmse(decoded.detach().cpu().numpy(), val_label.detach().cpu().numpy())
                    print('Epoch: ', epoch, '| loss_val: ', loss_val.item())
                    nmse_val.append(nmse)
                    LOSS_VAL.append(loss_val.item())
                    if loss_val < min_err:
                        min_err = loss_val
                        best_model_wts = copy.deepcopy(sta_dnn.state_dict())

            if min_err < r_min_err:
                r_min_err = min_err
                r_best_model_wts = best_model_wts

        sta_dnn.load_state_dict(r_best_model_wts)
        torch.save(sta_dnn.to('cpu'), dir_name + '.pkl')
    else:
        sta_dnn = torch.load(dir_name + '.pkl')

    plt.figure(1)
    x = range(EPOCH)
    plt.semilogy(x, LOSS_TRAIN, 'r-', label_data='loss_train')
    plt.semilogy(x, LOSS_VAL, 'b-', label_data='loss_val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import time
import copy

# AutoEncoder
# AutoEncoder 形式很简单, 分别是 encoder 和 decoder, 压缩和解压, 压缩后得到压缩的特征值, 再从压缩的特征值解压成原数据.


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(104, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU()
        )
        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 104)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def calc_mse(pred, target):
    mse = np.sum((pred - target)**2) / pred.size
    return mse


if __name__ == '__main__':
    # Hyper Parameters
    EPOCH = 40
    BATCH_SIZE = 128
    LR = 0.005            # learning rate
    initial = False
    train_test_ratio = 0.9
    r_cnt = 1
    clip = 1e-4
    non_empty_index = np.concatenate((range(6, 32), range(33, 59), range(70, 96), range(97, 123)), axis=0)  # 非空子载波索引
    train_data_name = '../train channel/AE_dataset.npy'

    # 下载数据
    data = np.load(train_data_name)
    print(data.size)
    scaler = StandardScaler()
    data = data[:, non_empty_index]
    data_scaler = scaler.fit_transform(data.reshape(-1, 2)).reshape(-1, 104)
    nums = int(data_scaler.shape[0])
    train_nums = int(train_test_ratio * nums)
    val_nums = nums - train_nums
    print('dataset size: ', nums, ', train set size: ', train_nums, ', val set size: ', val_nums)

    train_data = data_scaler[:train_nums]
    train_input = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_data).type(torch.FloatTensor)

    val_data = data_scaler[-val_nums:]
    val_input = torch.from_numpy(val_data).type(torch.FloatTensor)
    val_label = torch.from_numpy(val_data).type(torch.FloatTensor)
    # ----------------------------- load model ---------------------------- #
    dir_name = './autoencoder'
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
            num_workers=4 if torch.cuda.is_available() else 0)
        r_min_err = float('inf')
        for r in range(r_cnt):
            autoencoder = AutoEncoder().to(device)

            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            loss_func = nn.MSELoss()

            LOSS_TRAIN = []
            LOSS_VAL = []
            STEP = 0

            min_err = float('inf')
            time_train = 0

            for epoch in range(EPOCH):
                start = time.time()
                with torch.set_grad_enabled(True):
                    scheduler.step()
                    autoencoder.train()
                    for step, (train_batch, label_batch) in enumerate(train_loader):
                        encoded, decoded = autoencoder(train_batch.to(device))

                        loss_train = loss_func(decoded, label_batch.to(device))      # mean square error
                        optimizer.zero_grad()               # clear gradients for this training step
                        loss_train.backward()                     # backpropagation, compute gradients
                        nn.utils.clip_grad_norm_(autoencoder.parameters(), clip)
                        optimizer.step()                    # apply gradients

                        if step % 10000 == 0:
                            print('Epoch: ', epoch, '| train loss: %.4f' % loss_train.item())
                            LOSS_TRAIN.append(loss_train)
                time_train += time.time() - start
                with torch.set_grad_enabled(False):
                    autoencoder.eval()
                    _, decoded_data = autoencoder(val_input.to(device))
                    loss_val = loss_func(decoded_data, val_label.to(device))
                    print('Epoch: ', epoch, '| val loss: ', loss_val.item())
                    LOSS_VAL.append(loss_val)
                    if loss_val < min_err:
                        min_err = loss_val
                        best_model_wts = copy.deepcopy(autoencoder.state_dict())

            if min_err < r_min_err:
                r_min_err = min_err
                r_best_model_wts = best_model_wts

        autoencoder.load_state_dict(r_best_model_wts)
        torch.save(autoencoder.to('cpu'), dir_name + '.pkl')
    else:
        autoencoder = torch.load(dir_name + '.pkl')

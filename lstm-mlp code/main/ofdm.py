# -*- coding: utf-8 -*-
"""
Created on 2019/12/09
Revised in 2021/01/08

@author: panj
"""
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import cmath
import math
import function as fc
from autoencoder import AutoEncoder
from lstm import LSTM_MLP
from STA_DNN import STA_DNN
# ============================================function define========================================================#

# ===========================================main function==========================================================#
# start = time.clock()
# -----------------------------------parameter---------------------------------#
cycle = 1
SNR_dB = [0, 5, 10, 15, 20, 25, 30, 35, 40]
SNR_num = len(SNR_dB)
cp_length = 16
data_sub_num = 48
data_sym_len = 50  # ofdm symbol length in a frame
b_pilot_num = 2
frame_len = data_sym_len + b_pilot_num     # ofdm symbols per frame
frame_num = 960 # simulation frame number
ofdm_sym_num = data_sym_len * frame_num    # the total number of ofdm symbols 
bit_to_sym = np.array([1, 2, 4, 6])
# -------------------------pilot index----------------------------------------#
sub_index = range(64)
c_index = np.array([-21, -7, 7, 21]) + 32   # comb pilot index

e_index = np.concatenate([range(-32, -26), [0], range(27, 32)], axis=0) + 32    # empty symbol index
other_index = np.concatenate([c_index, e_index], axis=0)
v_index = np.delete(sub_index, e_index)  # the index of subcarrier removing empty subcarriers (valid subcarriers)
d_index = np.delete(sub_index, other_index)   # data symbol index
d2_index = np.concatenate([d_index, d_index+64], axis=0)    # index of the data subcarriers for real and imaginary part
long_training = np.array([0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 1+0j, 1+0j, -1+0j, -1+0j, 1+0j, 1+0j, -1+0j, 1+0j, -1+0j, 1+0j,
                          1+0j, 1+0j, 1+0j, 1+0j, 1+0j, -1+0j, -1+0j, 1+0j, 1+0j, -1+0j, 1+0j, -1+0j, 1+0j, 1+0j, 1+0j, 1+0j,
                          0+0j, 1+0j, -1+0j, -1+0j, 1+0j, 1+0j, -1+0j, 1+0j, -1+0j, 1+0j, -1+0j, -1+0j, -1+0j, -1+0j, -1+0j, 1+0j,
                          1+0j, -1+0j, -1+0j, 1+0j, -1+0j, 1+0j, -1+0j, 1+0j, 1+0j, 1+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j],
                         dtype='complex64')
comb_pilot = np.array([1+0j, 1+0j, 1+0j, -1+0j], dtype='complex64')
polor_para = np.array([1,1,1,1, -1,-1,-1,1, -1,-1,-1,-1, 1,1,-1,1, -1,-1,1,1, -1,1,1,-1, 1,1,1,1, 1,1,-1,1,
1,1,-1,1, 1,-1,-1,1, 1,1,-1,1, -1,-1,-1,1, -1,1,-1,-1, 1,-1,-1,1, 1,1,1,1, -1,-1,1,1,
-1,-1,1,-1, 1,-1,1,1, -1,-1,-1,1, 1,-1,-1,-1, -1,1,-1,-1, 1,-1,1,1, 1,1,-1,1, -1,1,-1,1,
-1,-1,-1,-1, -1,1,-1,1, 1,-1,1,-1, 1,1,1,-1, -1,1,-1,-1, -1,1,1,1, -1,-1,-1,-1, -1,-1,-1])

p_index = [5, 19, 32, 46, 57, 71, 84, 98]   # pilot subcarrier index for 104 subcarriers
output_index = np.delete(range(104), p_index)  # data subcarrier index for 104 subcarriers

# ============================ channel load and dealing==================================#
# name = 'ht_VTVEO_1200'
name = 'ht_test_20_41.7_41.7'
# name = 'ht_test_20_20_41.7'
channel_file = '../test channel/' + name + '.mat'
print('test channel : ' + name)
matlab_data = sio.loadmat(channel_file)
ht = matlab_data['h']
hf = np.fft.fft(ht[:frame_num * frame_len], n=64, axis=1)    # channel true value

# ============================= name of the trained neural network ==================================== #
lstm_mlp_name = 'lstm_mlp_dataset_input'
lstm_mlp_name_30 = 'lstm_mlp_dataset_input_30'
sta_dnn_name = 'model_sta_dnn'
ae_name = 'autoencoder'

# ================================initialization============================================#
[hf_d_refer,
 hf_sta,
 hf_cdp,
 hf_ae,
 hf_dl,
 hf_dl_30,
 hf_sta_dnn] = [np.zeros((frame_num, data_sym_len, len(d_index)), dtype="complex64") for i in range(7)]

[ber_refer,
 ber_sta,
 ber_cdp,
 ber_ae,
 ber_dl,
 ber_dl_30,
 ber_sta_dnn] = [np.zeros((4, SNR_num, cycle)) for j in range(7)]

# ======================================simulation process=============================================================#
for n in range(cycle):
    for modu_way in range(2, 4):  # 0: bpsk, 1: qpsk , 2: 16qam , 3: 64qam
        # ===============================transmitter===================================================================#
        # generate signal randomly
        signal_bit = fc.generate_signal(ofdm_sym_num, data_sub_num * bit_to_sym[modu_way])
        # map bit to symbol depending on modulation way
        ofdm_sym_48 = fc.map(signal_bit, modu_way)
        # insert comb pilot
        ofdm_sym_64 = fc.insert_comb_pilot(ofdm_sym_48.reshape(frame_num, data_sym_len, -1),
                                           comb_pilot, polor_para, c_index, d_index)
        # ifft
        ofdm_modulation_out = np.fft.ifft(ofdm_sym_64, n=64, axis=2)
        # insert Cyclic prefix
        ofdm_cp_out = fc.insert_cp(ofdm_modulation_out, cp_length)
        # insert block pilot
        trans_sig = fc.insert_block_pilot(ofdm_cp_out, data_sym_len, long_training, frame_num)

        # =============================passing time-varying channel====================================================#
        y = fc.add_channel(trans_sig, ht[:trans_sig.shape[0], :])
        hf_frame = hf[:frame_num * frame_len, :].reshape(frame_num, frame_len, 64)  # channel true value
        hf_bp_refer, hf_d_refer = hf_frame[:, :b_pilot_num, :], hf_frame[:, b_pilot_num:, :]
        print(modu_way)
        for m in range(SNR_num):
            y_wgn = fc.wgn(y, SNR_dB[m])
            # y_wgn = y   #无高斯白噪声
        # ================================receiver=================================================================#
            # cut block pilot
            y_frame = y_wgn.reshape(frame_num, frame_len, -1)  # receiver signal in frequency
            y_bp, y_d = y_frame[:, :b_pilot_num, :], y_frame[:, b_pilot_num:, :]
            # remove cyclic prefix
            y_bp_cutcp = y_bp.reshape(frame_num, -1)[:, cp_length * 2:].reshape(frame_num, 2, -1)
            y_d_cutcp = y_d[:, :, cp_length:]
            # fft
            yf_d = np.fft.fft(y_d_cutcp, n=64, axis=2)
            yf_bp = np.fft.fft(y_bp_cutcp, n=64, axis=2)
            # pilot symbol channel estimation (LS algorithm)
            hf_BP_ls = fc.block_pilot_estimation(yf_bp, long_training, frame_num, b_pilot_num, v_index)
            hf_CP_ls = np.zeros(hf_d_refer.shape, dtype='complex64')
            hf_CP_ls[:, :, c_index] = fc.comb_pilot_estimation(yf_d[:, :, c_index], comb_pilot, polor_para, frame_num,
                                                               data_sym_len)
            hf_P_ls = np.concatenate((hf_BP_ls, hf_CP_ls), axis=1)  # pilot symbol channel estimation using LS
            # truth value demodulation (ideal value)
            xf_D_refer = (yf_d[:, :, d_index] / hf_d_refer[:, :, d_index]).reshape(-1, 48)
            xf_refer_bit = fc.demap(xf_D_refer, modu_way)
            ber_refer[modu_way, m, n] = fc.cal_ber(xf_refer_bit, signal_bit)

            # Spectral Temporal Averaging
            ber_sta[modu_way, m, n], hf_sta = fc.sta(hf_P_ls, yf_d, d_index, c_index, modu_way, signal_bit,
                                                     frame_num, data_sym_len)
            # Constructed Data Pilot
            ber_cdp[modu_way, m, n], hf_cdp = fc.cdp(hf_BP_ls, yf_d, yf_bp, d_index, modu_way, signal_bit,
                                                     long_training, frame_num, data_sym_len)
            # Autoencoder
            ber_ae[modu_way, m, n], hf_ae = fc.ae(hf_P_ls, yf_d, d_index, modu_way, signal_bit,
                                                  v_index, c_index, ae_name)
            # # Proposed method (DPA + LSTM + DNN)
            ber_dl[modu_way, m, n], hf_dl = fc.dl(hf_P_ls, yf_d, d_index, modu_way, signal_bit,
                                                  v_index, c_index, lstm_mlp_name)
            # the Proposed network trained in 30dB
            ber_dl_30[modu_way, m, n], hf_dl_30 = fc.dl(hf_P_ls, yf_d, d_index, modu_way, signal_bit,
                                                        v_index, c_index, lstm_mlp_name_30)
            # # STA_DNN
            ber_sta_dnn[modu_way, m, n], hf_sta_dnn = fc.sta_dnn(hf_P_ls, yf_d, d_index, modu_way, signal_bit,
                                                                 v_index, c_index, sta_dnn_name)

            print(m)
# end = time.clock()
sio.savemat('ber_' + name + '_0121.mat', {'ber_STA': ber_sta, 'ber_CDP': ber_cdp, 'ber_AE': ber_ae, 'ber_LSTM': ber_dl,
                                     'ber_LSTM_30dB': ber_dl_30,
                                     'ber_STA_DNN': ber_sta_dnn,
                                     # 'ber_lmmse': ber_lmmse,
                                     'ber_ideal': ber_refer})
# print(end-start)


plt.figure(1)
x = range(0, 45, 5)
plt.semilogy(x, np.mean(ber_cdp[2, :, :], axis=1), 'b-*', label='CDP')
plt.semilogy(x, np.mean(ber_sta[2, :, :], axis=1), 'm-*', label='sta')
plt.semilogy(x, np.mean(ber_ae[2, :, :], axis=1), 'g-*', label='AE-CE')
plt.semilogy(x, np.mean(ber_dl[2, :, :], axis=1), 'y-*', label='proposed')
plt.semilogy(x, np.mean(ber_dl_30[2, :, :], axis=1), 'r-*', label='proposed_30')
plt.semilogy(x, np.mean(ber_sta_dnn[2, :, :], axis=1), 'c-*', label='sta_dnn')
plt.semilogy(x, np.mean(ber_refer[2, :, :], axis=1), 'k-^', label='True value')
plt.xlim(0, 40)
plt.xlabel('SNR(dB)')
plt.ylabel('ber')
plt.legend()
plt.grid()

plt.figure(2)
x = range(0, 45, 5)
plt.semilogy(x, np.mean(ber_cdp[3, :, :], axis=1), 'b-*', label='CDP')
plt.semilogy(x, np.mean(ber_sta[3, :, :], axis=1), 'm-*', label='sta')
plt.semilogy(x, np.mean(ber_ae[3, :, :], axis=1), 'g-*', label='AE-CE')
plt.semilogy(x, np.mean(ber_dl[3, :, :], axis=1), 'y-*', label='proposed')
plt.semilogy(x, np.mean(ber_dl_30[3, :, :], axis=1), 'r-*', label='proposed_30')
plt.semilogy(x, np.mean(ber_sta_dnn[3, :, :], axis=1), 'c-*', label='sta_dnn')
plt.semilogy(x, np.mean(ber_refer[3, :, :], axis=1), 'k-^', label='True value')
plt.xlim(0, 40)
plt.xlabel('SNR(dB)')
plt.ylabel('ber')
plt.legend()
plt.grid()
#
# # # plt.savefig('./h_20_25_30.jpg')
plt.show()
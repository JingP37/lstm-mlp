# -*- coding: utf-8 -*-
"""
Created on 2019/12/09

@author: panj
"""
import torch
from sklearn.preprocessing import StandardScaler
import math
import numpy as np


def generate_signal(m, n):
    output = np.random.randint(0, 2, size=[m, n])    #np.random.randint（a,b） 的取值范围为[a，b）之间的整数
    return output


def map(signal_bit, modu_way):
    '''
    :param signal_bit: the bit signal ,shape = (ofdm_sym_num, data_sub_num*bit_to_sym[modu_way])
    :param modu_way:  0:bpsk, 1:qpsk, 2:16qam, 3:64qam
    :return: output , pilot_symbol
             output = signal_symbol, shape =(ofdm_sym_num, data_sub_num)
    '''

    if modu_way == 0:
        output = map_bpsk(signal_bit)
    elif modu_way == 1:
        output = map_qpsk(signal_bit)
    elif modu_way == 2:
        output = map_16qam(signal_bit)
    elif modu_way == 3:
        output = map_64qam(signal_bit)
    else:
        print('the input of modu_way is error')
        output = 1
    return output


def map_bpsk(signal_bit):
    output = np.empty_like(signal_bit, dtype="complex64")
    for m in range(signal_bit.shape[0]):
        for n in range(signal_bit.shape[1]):
            if signal_bit[m, n] == 0:
                output[m, n] = -1 + 0j
            else:
                output[m, n] = 1 + 0j
    return output


def map_qpsk(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1] / 2)
    x = signal_bit.reshape(c, d, 2)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :]
            if (a == [0, 0]).all():
                output[m, n] = complex(-math.sqrt(2)/2, -math.sqrt(2)/2)
            elif (a == [0, 1]).all():
                output[m, n] = complex(-math.sqrt(2)/2, math.sqrt(2)/2)
            elif (a == [1, 1]).all():
                output[m, n] = complex(math.sqrt(2) / 2, math.sqrt(2) / 2)
            else:
                output[m, n] = complex(math.sqrt(2) / 2, -math.sqrt(2) / 2)
    return output


def map_16qam(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1]/4)
    x = signal_bit.reshape(c, d, 4)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :2]
            if (a == [0, 0]).all():
                real = -3
            elif (a == [0, 1]).all():
                real = -1
            elif (a == [1, 1]).all():
                real = 1
            else:
                real = 3
            b = x[m, n, 2:]
            if (b == [0, 0]).all():
                imag = -3
            elif (b == [0, 1]).all():
                imag = -1
            elif (b == [1, 1]).all():
                imag = 1
            else:
                imag = 3
            output[m, n] = complex(real, imag)/math.sqrt(10)
    return output


def map_64qam(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1]/6)
    x = signal_bit.reshape(c, d, 6)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :3]
            if (a == [0, 0, 0]).all():
                real = -7
            elif (a == [0, 0, 1]).all():
                real = -5
            elif (a == [0, 1, 1]).all():
                real = -3
            elif (a == [0, 1, 0]).all():
                real = -1
            elif (a == [1, 0, 0]).all():
                real = 7
            elif (a == [1, 0, 1]).all():
                real = 5
            elif (a == [1, 1, 1]).all():
                real = 3
            else:
                real = 1
            b = x[m, n, 3:]
            if (b == [0, 0, 0]).all():
                imag = -7
            elif (b == [0, 0, 1]).all():
                imag = -5
            elif (b == [0, 1, 1]).all():
                imag = -3
            elif (b == [0, 1, 0]).all():
                imag = -1
            elif (b == [1, 0, 0]).all():
                imag = 7
            elif (b == [1, 0, 1]).all():
                imag = 5
            elif (b == [1, 1, 1]).all():
                imag = 3
            else:
                imag = 1
            output[m, n] = complex(real, imag)/math.sqrt(84)
    return output


def demap(signal_symbol, modu_way):
    '''
    :param signal_symbol: the symbol signal ,shape = (ofdm_sym_num, data_sub_num)
    :param modu_way:  0:bpsk, 1:qpsk, 2:16qam, 3:64qam
    :return: output
             output = signal_bit, shape =(ofdm_sym_num, data_sub_num*bit_to_sym[modu_way])
    '''
    if signal_symbol.ndim == 1:
        signal_symbol = signal_symbol[np.newaxis, :]
    if modu_way == 0:
        output = demap_bpsk(signal_symbol)
    elif modu_way == 1:
        output = demap_qpsk(signal_symbol)
    elif modu_way == 2:
        output = demap_16qam(signal_symbol)
    elif modu_way == 3:
        output = demap_64qam(signal_symbol)
    else:
        print('the input of modu_way is error')
        output = 1
    return output


def demap_bpsk(x):
    output = np.empty_like(x, dtype="int")
    for m in range(x.shape[0]):
        for n in range(x.shape[1]):
            if x[m, n].real >= 0:
                output[m, n] = 1
            else:
                output[m, n] = 0
    return output


def demap_qpsk(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 2), dtype="int")
    for m in range(c):
        for n in range(d):
            a = x[m, n].real
            b = x[m, n].imag
            if (a <= 0) & (b <= 0):
                output[m, n, :] = [0, 0]
            elif (a <= 0) & (b > 0):
                output[m, n, :] = [0, 1]
            elif (a > 0) & (b > 0):
                output[m, n, :] = [1, 1]
            else:
                output[m, n, :] = [1, 0]
    output = output.reshape(c, int(2*d))
    return output


def demap_16qam(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 4), dtype="int")
    for m in range(c):
        for n in range(d):
            a = math.sqrt(10)*x[m, n].real
            if a <= -2:
                output[m, n, :2] = [0, 0]
            elif (a <= 0) & (a > -2):
                output[m, n, :2] = [0, 1]
            elif (a <= 2) & (a > 0):
                output[m, n, :2] = [1, 1]
            else:
                output[m, n, :2] = [1, 0]
            b = math.sqrt(10)*x[m, n].imag
            if b <= -2:
                output[m, n, 2:] = [0, 0]
            elif (b <= 0) & (b > -2):
                output[m, n, 2:] = [0, 1]
            elif (b <= 2) & (b > 0):
                output[m, n, 2:] = [1, 1]
            else:
                output[m, n, 2:] = [1, 0]
    output = output.reshape((c, int(4*d)))
    return output


def demap_64qam(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 6), dtype="int")
    for m in range(c):
        for n in range(d):
            a = math.sqrt(84)*x[m, n].real
            if a <= -6:
                output[m, n, :3] = [0, 0, 0]
            elif (a > -6) & (a <= -4):
                output[m, n, :3] = [0, 0, 1]
            elif (a > -4) & (a <= -2):
                output[m, n, :3] = [0, 1, 1]
            elif (a > -2) & (a <= 0):
                output[m, n, :3] = [0, 1, 0]
            elif (a > 0) & (a <= 2):
                output[m, n, :3] = [1, 1, 0]
            elif (a > 2) & (a <= 4):
                output[m, n, :3] = [1, 1, 1]
            elif (a > 4) & (a <= 6):
                output[m, n, :3] = [1, 0, 1]
            else:
                output[m, n, :3] = [1, 0, 0]
            b = math.sqrt(84) * x[m, n].imag
            if b <= -6:
                output[m, n, 3:] = [0, 0, 0]
            elif (b > -6) & (b <= -4):
                output[m, n, 3:] = [0, 0, 1]
            elif (b > -4) & (b <= -2):
                output[m, n, 3:] = [0, 1, 1]
            elif (b > -2) & (b <= 0):
                output[m, n, 3:] = [0, 1, 0]
            elif (b > 0) & (b <= 2):
                output[m, n, 3:] = [1, 1, 0]
            elif (b > 2) & (b <= 4):
                output[m, n, 3:] = [1, 1, 1]
            elif (b > 4) & (b <= 6):
                output[m, n, 3:] = [1, 0, 1]
            else:
                output[m, n, 3:] = [1, 0, 0]
    output = output.reshape(c, int(6*d))
    return output


def insert_comb_pilot(ofdm_sym_48, comb_pilot, polor_para, c_index, d_index):
    ofdm_sym_64 = np.zeros((ofdm_sym_48.shape[0], ofdm_sym_48.shape[1], 64), dtype="complex64")
    for i in range(ofdm_sym_48.shape[0]):
        for j in range(ofdm_sym_48.shape[1]):
            ofdm_sym_64[i, j, c_index] = comb_pilot * polor_para[j]
            ofdm_sym_64[i, j, d_index] = ofdm_sym_48[i, j, :]
    return ofdm_sym_64


def insert_block_pilot(input, data_sym_len, long_training,frame_num):
    h_long_training = np.fft.ifft(long_training, 64)
    b_pilot = np.concatenate((np.array(h_long_training)[32:], h_long_training, h_long_training), axis=0).reshape(2, -1)
    block_pilot = np.array([b_pilot] * frame_num)

    a = input.reshape(frame_num, data_sym_len, -1)
    output = np.concatenate((block_pilot, a), axis=1)

    return output.reshape(frame_num*(data_sym_len+2),-1)


def insert_cp(ofdm_modulation_out, cp_length):
    output = np.concatenate((ofdm_modulation_out[:, :, -cp_length:], ofdm_modulation_out), axis=2)
    return output


def add_channel(input, ht):
    output = np.zeros(input.shape, dtype="complex64")
    for m in range(input.shape[0]):
        for n in range(input.shape[1]):
            for p in range(ht.shape[1]):
                if n-p >= 0:
                    output[m, n] = output[m, n] + ht[m, p]*input[m, n-p]
                else:
                    break
    return output


def wgn(x,  snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(abs(x)**2)/x.size
    npower = xpower / snr
    noise_real = 2*(np.random.rand(x.shape[0], x.shape[1])-0.5*np.ones_like(x))
    noise_imag = 2*(np.random.rand(x.shape[0], x.shape[1])-0.5*np.ones_like(x))
    noise = noise_real+1j*noise_imag
    unit_power = np.sum(abs(noise)**2)/noise.size
    noise = noise/math.sqrt(unit_power)*math.sqrt(npower)
    # npower_final = np.sum(abs(noise) ** 2) / noise.size
    return x+noise


def block_pilot_estimation(yf, xf, frame_num, b_pilot_num, v_index):
    hf_bp = np.zeros(yf.shape, dtype="complex64")
    xf_frame = np.tile(xf, (frame_num, b_pilot_num, 1))
    hf_bp[:, :, v_index] = yf[:, :, v_index]/xf_frame[:, :, v_index]
    return hf_bp


def comb_pilot_estimation(yf_cp, comb_pilot, polor_para, frame_num, data_sym_len):
    hf_cp = np.zeros(yf_cp.shape, dtype="complex64")
    for i in range(frame_num):
        for j in range(data_sym_len):
            hf_cp[i, j, :] = yf_cp[i, j, :]/comb_pilot*polor_para[j]
    return hf_cp


def linear_interp(input, index, y):
    h = np.empty(y.shape, dtype="complex64")
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                if k <= index[1]:
                    h[i, j, k] = (input[i, j, index[0]]-input[i, j, index[1]])/(index[0]-index[1])*(k - index[0]) + input[i, j, index[0]]
                elif k <= index[2] & k > index[1]:
                    h[i, j, k] = (input[i, j, index[1]] - input[i, j, index[2]]) / (index[1] - index[2]) * (k - index[1]) + input[i, j, index[1]]
                else:
                    h[i, j, k] = (input[i, j, index[2]] - input[i, j, index[3]]) / (index[2] - index[3]) * (k - index[2]) + input[i, j, index[2]]
    return h, y/h


def cal_ber(x, y):
    ber = 0
    x1 = x.reshape(-1)
    y1 = y.reshape(-1)
    error_num = 0
    for index in range(len(x1)):
        if x1[index] != y1[index]:
            error_num = error_num + 1
    ber = error_num/len(x1)
    return ber


def linear(hf_CP_ls, yf_D, d_index, c_index, modu_way, signal_bit):
    hf_linear, xf_linear = linear_interp(hf_CP_ls, c_index, yf_D)
    xf_linear_bit = demap(xf_linear.reshape(-1, 64)[:, d_index], modu_way)  #demap
    ber_linear = cal_ber(xf_linear_bit, signal_bit)    #calculate ber
    return ber_linear, hf_linear


def DD(hf_b, yf_d, d_index, modu_way, x_bit_true):
    hf_D = np.zeros((yf_d.shape[0], yf_d.shape[1], d_index.size), dtype="complex128")
    sf = np.zeros((1, d_index.size), dtype="complex128")
    frame_num = yf_d.shape[0]
    frame_len = yf_d.shape[1]
    # 每一帧的初始信道由帧头长训练符号信道估计得到
    for i in range(frame_num):
        hf = np.mean(hf_b[i, :, d_index], axis=1)
        for j in range(frame_len):
            hf_D[i, j, :] = hf
            sf[:, :] = yf_d[i, j, d_index]/hf
            x = demap(sf, modu_way)
            xf = map(x, modu_way)
            hf = yf_d[i, j, d_index]/xf
            if (i == 0) & (j == 0):
                x_bit = x
            else:
                x_bit = np.concatenate((x_bit, x), axis=0)
    ber = cal_ber(x_bit, x_bit_true)
    return ber, hf_D


def sta(hf_p_ls, yf_d, d_index, c_index, modu_way, x_bit_true, frame_num, data_sym_len):
    hf_sta = np.zeros(shape=yf_d.shape, dtype="complex64")
    hf = np.zeros(yf_d.shape[2], dtype="complex64")
    hf_update = np.zeros(64, dtype="complex64")
    x_bit = []
    for i in range(frame_num):
        hf[d_index] = np.mean(hf_p_ls[i, :2, :][:, d_index], axis=0)
        for j in range(data_sym_len):
            sf = yf_d[i, j, d_index] / hf[d_index]
            x = demap(sf, modu_way)
            x_bit += [x]
            xf = map(x, modu_way)
            hf[d_index] = yf_d[i, j, d_index] / xf  # DPA procedure
            hf[c_index] = hf_p_ls[i, j + 2, c_index]
            for k in range(yf_d.shape[2]):
                a, b = 0, 0
                for m in range(-2, 3, 1):  # 由于边界和空符号的影响需要分段考虑
                    if k + m >= 6 and k + m < 59 and k + m != 32:
                        a = a + hf[k + m]
                        b = b + 1   # 计数符号，计加了几次
                if b != 0:
                    hf_update[k] = a / b
            hf[d_index] = 1 / 2 * hf_update[d_index] + 1 / 2 * hf[d_index]
            hf_sta[i, j, d_index] = hf[d_index]
    ber = cal_ber(np.array(x_bit), x_bit_true)
    return ber, hf_sta[:, :, d_index]


def cdp(hf_b, yf_d, yf_bp, d_index, modu_way, x_bit_true, long_training, frame_num, data_sym_len):
    hf_cdp = np.zeros((frame_num, data_sym_len, len(d_index)), dtype="complex64")
    xf_i = np.zeros((len(d_index)), dtype="complex64")
    x_bit = []
    xf_pre = []
    for i in range(frame_num):
        hf = hf_b[i, -1, d_index].squeeze()
        for j in range(data_sym_len):
            hf_cdp[i, j, :] = hf
            sf = (yf_d[i, j, d_index]/hf).squeeze()
            # DPA 过程
            x = demap(sf, modu_way)
            xf = np.squeeze(map(x, modu_way))
            hf_i = (yf_d[i, j, d_index]/xf).squeeze()
            # 存储每次解调后的值
            x_bit += [x]
            # 用该时刻的信道去均衡上一时刻的接收信号，并解调
            if j == 0:
                xf_pre = long_training[d_index]
                s_i = (yf_bp[i, -1, d_index]/hf_i).squeeze()
                for k in range(len(d_index)):
                    xf_i[k] = 1 + 0j if s_i[k].real > 0 else -1 + 0j
            else:
                s_i = (yf_d[i, j - 1, d_index]/hf_i).squeeze()
                x_i = demap(s_i, modu_way)
                xf_i = np.squeeze(map(x_i, modu_way))
            for k in range(len(d_index)):
                if (xf_i[k] == xf_pre[k]).all():
                    hf[k] = hf_i[k]
            xf_pre = xf
    ber = cal_ber(np.array(x_bit), x_bit_true)
    return ber, hf_cdp


def ae(hf_p, yf_d, d_index, modu_way, x_bit_true, v_index, c_index, dir_name):
    frame_num = yf_d.shape[0]
    frame_len = yf_d.shape[1]
    hf_ae = np.zeros((frame_num, frame_len, d_index.size), dtype="complex64")
    sf = np.zeros((1, d_index.size), dtype="complex64")
    hf_input = np.zeros(64, dtype="complex64")
    p_index = [5, 19, 32, 46, 57, 71, 84, 98]
    output_index = np.delete(range(104), p_index)
    device = torch.device("cpu")
    AutoEncoder = torch.load(dir_name + '.pkl').to(device)
    scaler = StandardScaler()

    for i in range(frame_num):
        hf = np.mean(hf_p[i, :2, d_index], axis=1)
        for j in range(frame_len):
            hf_ae[i, j, :] = hf
            sf = yf_d[i, j, d_index]/hf
            x = demap(sf, modu_way)
            xf = map(x, modu_way)
            hf = yf_d[i, j, d_index]/xf        #DPA procedure
            hf_input[d_index] = hf
            hf_input[c_index] = hf_p[i, j + 2, c_index]
            input = np.concatenate((hf_input[v_index].real, hf_input[v_index].imag), axis=0)
            # ----------------实例化------------------#
            input1 = scaler.fit_transform(input.reshape(-1, 2)).reshape(input.shape)
            input2 = torch.from_numpy(input1).type(torch.FloatTensor)
            _, output = AutoEncoder(input2.to(device))
            out = scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1, 2)).reshape(output.shape)
            out = out[output_index]
            hf = out[:48]+1j*out[48:]
            if (i == 0) & (j == 0):
                x_bit = x
            else:
                x_bit = np.concatenate((x_bit, x), axis=0)
    ber = cal_ber(x_bit, x_bit_true)
    return ber, hf_ae


def dl(hf_p, yf_d, d_index, modu_way, x_bit_true, v_index, c_index, dir_name):
    hf_DL = np.zeros((yf_d.shape[0], yf_d.shape[1], d_index.size), dtype="complex64")
    p_index = [5, 19, 32, 46, 57, 71, 84, 98]
    output_index = np.delete(range(104), p_index)
    sf = np.zeros((1, d_index.size), dtype="complex64")
    x_bit = []
    device = torch.device("cpu")
    NET = torch.load(dir_name + '.pkl').to(device)
    scaler = StandardScaler()

    for i in range(yf_d.shape[0]):
        hf = np.mean(hf_p[i, :2, :].squeeze(), axis=0)
        hn, cn = None, None
        for j in range(yf_d.shape[1]):
            hf_input = hf
            input = np.concatenate((hf_input[v_index].real,
                                    hf_input[v_index].imag,
                                    hf_p[i, j + 2, c_index].real,
                                    hf_p[i, j + 2, c_index].imag), axis=0)
            # ----------------实例化------------------#
            input1 = scaler.fit_transform(input.reshape(-1, 2)).reshape(input.shape)
            input2 = torch.from_numpy(input1).type(torch.FloatTensor).unsqueeze(0)

            output, hn, cn = NET(input2.to(device), hn, cn)
            out = scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1, 2)).reshape(output.shape)
            # out = out[:,  output_index]
            hf_out = out[:, :48] + 1j * out[:, 48:]
            hf_DL[i, j, :] = hf_out
            sf = yf_d[i, j, d_index]/hf_out
            x = demap(sf, modu_way)
            xf = map(x, modu_way)
            hf_out = yf_d[i, j, d_index] / xf
            hf[c_index] = hf_p[i, j + 2, c_index]
            hf[d_index] = hf_out

            if (i == 0) & (j == 0):
                x_bit = x
            else:
                x_bit = np.concatenate((x_bit, x), axis=0)

    ber = cal_ber(x_bit, x_bit_true)
    return ber, hf_DL


def sta_dnn(hf_p_ls, yf_d, d_index, modu_way, x_bit_true, v_index, c_index, dir_name):
    frame_num = yf_d.shape[0]
    frame_len = yf_d.shape[1]
    hf_sta_dnn = np.zeros((frame_num, frame_len, 64), dtype="complex64")
    sf = np.zeros((1, d_index.size), dtype="complex64")
    hf_input = np.zeros(64, dtype="complex64")
    hf = np.zeros(yf_d.shape[2], dtype="complex64")
    hf_update = np.zeros(64, dtype="complex64")

    device = torch.device("cpu")
    NET4 = torch.load(dir_name + '.pkl').to(device)
    scaler = StandardScaler()
    x_bit = []

    for i in range(frame_num):
        hf[d_index] = np.mean(hf_p_ls[i, :2, d_index], axis=1)
        for j in range(frame_len):
            hf_sta_dnn[i, j, :] = hf
            sf[:, :] = yf_d[i, j, d_index] / hf[d_index]
            x = demap(sf, modu_way)
            xf = map(x, modu_way)
            if (i == 0) & (j == 0):
                x_bit = x
            else:
                x_bit = np.concatenate((x_bit, x), axis=0)
            hf[d_index] = yf_d[i, j, d_index] / xf
            hf[c_index] = hf_p_ls[i, j + 2, c_index]
            for k in range(yf_d.shape[2]):
                a, b = 0, 0
                for m in range(-2, 3, 1):  # 由于边界和空符号的影响需要分段考虑
                    if k + m >= 6 and k + m < 59 and k + m != 32:
                        a = a + hf[k + m]
                        b = b + 1  # 计数符号，计加了几次
                if b != 0:
                    hf_update[k] = a / b
            hf = 1 / 2 * hf_update + 1 / 2 * hf
            input = np.concatenate((hf[v_index].real, hf[v_index].imag), axis=0)
            # ----------------实例化------------------#
            input1 = scaler.fit_transform(input.reshape(-1, 2)).reshape(input.shape)
            input2 = torch.from_numpy(input1).type(torch.FloatTensor)
            _, output = NET4(input2.to(device))
            out = scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1, 2)).reshape(output.shape)
            hf[d_index] = out[:48] + 1j * out[48:]

    ber = cal_ber(x_bit, x_bit_true)
    return ber, hf_sta_dnn


# def lmmse(hf_d_refer, hf_ls, yf_d, v_index, d_index, c_index, modu_way, signal_bit, ofdm_sym_64, SNR_dB):
#     hf_d = hf_d_refer.reshape(-1, 64)
#     yf_d = yf_d.reshape(-1, 64)
#     hf_ls = hf_ls.reshape(-1, 64)
#     ofdm_sym = ofdm_sym_64.reshape(-1, 64)
#     sf = np.zeros((1, 64), dtype="complex64")
#     hf_lmmse = np.zeros_like(hf_d)
#     SNR = 10 ** (SNR_dB / 10)
#
#     if modu_way == 0 :
#         beta = 1
#     elif modu_way == 1:
#         beta = 1
#     elif modu_way == 2:
#         beta = 17/9
#     elif modu_way == 3:
#         beta = 1289/480
#
#     for i in range(hf_d.shape[0]):
#         # auto_corr = np.transpose(np.mat(hf_d[i, c_index])) * np.transpose(np.mat(hf_d[i, c_index]).H)
#         # cross_corr = np.transpose(np.mat(hf_d[i, v_index])) * np.transpose(np.mat(hf_ls[i, c_index]).H)
#         # hf_lmmse = np.transpose(cross_corr * np.linalg.pinv(auto_corr) * np.transpose(np.mat(hf_ls[i, c_index])).A)
#
#         auto_corr = np.transpose(np.mat(hf_d[i, c_index])) * np.transpose(np.mat(hf_d[i, c_index]).H)
#         cross_corr = np.transpose(np.mat(hf_d[i, v_index])) * np.transpose(np.mat(hf_d[i, c_index]).H)
#         hf_lmmse = np.transpose(cross_corr * np.linalg.pinv(auto_corr + beta/SNR * np.eye(4)) * np.transpose(np.mat(hf_ls[i, c_index])).A)
#
#         sf[:, v_index] = yf_d[i, v_index] / hf_lmmse
#         x = demap(sf[:, d_index], modu_way)
#         if i == 0:
#             x_bit = x
#         else:
#             x_bit = np.concatenate((x_bit, x), axis=0)
#
#     ber = cal_ber(x_bit, signal_bit)
#     return ber, hf_lmmse





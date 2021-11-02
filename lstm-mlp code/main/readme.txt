The file "ofdm" is the main program for the simulation. And file "autoencoder" or "lstm" or "STA_DNN" can be used for training and testing.

该文件夹中，ofdm文件用于OFDM系统的仿真，其中用到了多种信道估计方法解调，具体看代码；
其余"autoencoder","STA_DNN"和"lstm"文件均为各信道估计算法中的神经网络代码，具体可看文献。
“lstm”文件中包含的网络即为文献中的LSTM-MLP网络。里面部分代码用于网络训练。

lstm_mlp_dataset_input.pkl是已训练好的LSTM-MLP模型，其中包含了相应的参数
lstm_mlp_dataset_input_30.pkl中的网络结构和LSTM-MLP模型一致，但其训练的环境中信道含有30dB的噪声，
其余model_sta_dnn.pkl和autoencoder.pkl为相应的网络在同一数据集下训练的模型。
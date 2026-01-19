import os
import numpy as np
print('----------------------')
data_path_1 = os.path.join('./raw_data/PEMS04_10/PEMS04_10.npz')  # 完整的
data_1 = np.load(data_path_1)['data'][:, :]
nan_1 = np.sum(np.isnan(data_1))
print('raw_data/PEMS04_10/PEMS04_10.npz数据集个数：',data_1.size)
print('raw_data/PEMS04_10/PEMS04_10.npz数据集形状：',data_1.shape)
print('raw_data/PEMS04_10/PEMS04_10.npz数据集中的nan值个数：',nan_1)

print('----------------------')
data_path_2 = os.path.join('./raw_data/PEMS04_30/PEMS04_30.npz')  # 完整的
data_2 = np.load(data_path_2)['data'][:, :]
nan_2 = np.sum(np.isnan(data_2))
print('raw_data/PEMS04_30/PEMS04_30.npz数据集个数：',data_2.size)
print('raw_data/PEMS04_30/PEMS04_30.npz数据集形状：',data_2.shape)
print('raw_data/PEMS04_30/PEMS04_30.npz数据集中的nan值个数：',nan_2)

print('----------------------')
data_path_3 = os.path.join('./raw_data/PEMS04/PEMS04.npz')
data_3 = np.load(data_path_3)['data'][:, :]
nan_3 = np.sum(np.isnan(data_3))
print('raw_data/PEMS04/PEMS04.npz数据集个数：',data_3.size)
print('raw_data/PEMS04/PEMS04.npz数据集形状：',data_3.shape)
print('raw_data/PEMS04/PEMS04.npz数据集中的nan值个数：',nan_3)

print('----------------------')
data_path_4 = os.path.join('raw_data/PEMS08_30/PEMS08_30.npz')  # 完整的
data_4 = np.load(data_path_4)['data'][:, :]
nan_4 = np.sum(np.isnan(data_4))
print('raw_data/PEMS08_30/PEMS08_30.npz数据集个数：',data_4.size)
print('raw_data/PEMS08_30/PEMS08_30.npz数据集形状：',data_4.shape)
print('raw_data/PEMS08_30/PEMS08_30.npz数据集中的nan值个数：',nan_4)

print('----------------------')
data_path_5 = os.path.join('raw_data/PEMS08_10/PEMS08_10.npz')  # 完整的
data_5 = np.load(data_path_5)['data'][:, :]
nan_5 = np.sum(np.isnan(data_5))
print('raw_data/PEMS08_10/PEMS08_10.npz数据集个数：',data_4.size)
print('raw_data/PEMS08_10/PEMS08_10.npz数据集形状：',data_4.shape)
print('raw_data/PEMS08_10/PEMS08_10.npz数据集中的nan值个数：',nan_4)

print('----------------------')
data_path_6 = os.path.join('./raw_data/PEMS08/PEMS08.npz')
data_6 = np.load(data_path_6)['data'][:, :]
nan_6 = np.sum(np.isnan(data_6))
print('raw_data/PEMS08/PEMS08.npz数据集个数：',data_6.size)
print('raw_data/PEMS08/PEMS08.npz数据集形状：',data_6.shape)
print('raw_data/PEMS08/PEMS08.npz数据集中的nan值个数：',nan_6)

print('----------------------')
print('*******************')
data_path_7_1 = os.path.join('./miss_nan/PEMS04_miss_nan_10.npz')
data_7_1 = np.load(data_path_7_1)['data'][:, :]
nan_7_1 = np.sum(np.isnan(data_7_1))
print('miss_nan/PEMS04_miss_nan_10.npz数据集个数：',data_7_1.size)
print('miss_nan/PEMS04_miss_nan_10.npz数据集形状：',data_7_1.shape)
print('miss_nan/PEMS04_miss_nan_10.npz数据集中的nan值个数：',nan_7_1)

print('*******************')
data_path_7_2 = os.path.join('./miss_nan/PEMS04_miss_nan_30.npz')
data_7_2 = np.load(data_path_7_2)['data'][:, :]
nan_7_2 = np.sum(np.isnan(data_7_2))
print('miss_nan/PEMS04_miss_nan_30.npz数据集个数：',data_7_2.size)
print('miss_nan/PEMS04_miss_nan_30.npz数据集形状：',data_7_2.shape)
print('miss_nan/PEMS04_miss_nan_30.npz数据集中的nan值个数：',nan_7_2)

print('*******************')
data_path_7_3 = os.path.join('./miss_nan/PEMS08_miss_nan_10.npz')
data_7_3 = np.load(data_path_7_3)['data'][:, :]
nan_7_3 = np.sum(np.isnan(data_7_3))
print('miss_nan/PEMS08_miss_nan_10.npz数据集个数：',data_7_3.size)
print('miss_nan/PEMS08_miss_nan_10.npz数据集形状：',data_7_3.shape)
print('miss_nan/PEMS08_miss_nan_10.npz数据集中的nan值个数：',nan_7_3)

print('*******************')
data_path_7_4 = os.path.join('./miss_nan/PEMS08_miss_nan_30.npz')
data_7_4 = np.load(data_path_7_4)['data'][:, :]
nan_7_4 = np.sum(np.isnan(data_7_4))
print('miss_nan/PEMS08_miss_nan_30.npz数据集个数：',data_7_4.size)
print('miss_nan/PEMS08_miss_nan_30.npz数据集形状：',data_7_4.shape)
print('miss_nan/PEMS08_miss_nan_30.npz数据集中的nan值个数：',nan_7_4)

print('----------------------')
data_path_8 = os.path.join('./missing_data_generate/PEMS04_missing_10.npz')
data_8 = np.load(data_path_8)['data'][:, :]
nan_8 = np.sum(np.isnan(data_8))
print('missing_data_generate/PEMS04_missing_10.npz数据集个数：',data_8.size)
print('missing_data_generate/PEMS04_missing_10.npz数据集形状：',data_8.shape)
print('missing_data_generate/PEMS04_missing_10.npz数据集中的nan值个数：',nan_8)

print('----------------------')
data_path_8_1 = os.path.join('./missing_data_generate/PEMS04_missing_30.npz')
data_8_1 = np.load(data_path_8_1)['data'][:, :]
nan_8_1 = np.sum(np.isnan(data_8_1))
print('missing_data_generate/PEMS04_missing_30.npz数据集个数：',data_8_1.size)
print('missing_data_generate/PEMS04_missing_30.npz数据集形状：',data_8_1.shape)
print('missing_data_generate/PEMS04_missing_30.npz数据集中的nan值个数：',nan_8_1)


print('----------------------')
data_path_9 = os.path.join('./Data/PEMSD4/pems04.npz')
data_9 = np.load(data_path_9)['data'][:, :]
nan_9 = np.sum(np.isnan(data_9))
print('Data/PEMSD4/pems04.npz数据集个数：',data_9.size)
print('Data/PEMSD4/pems04.npz数据集形状：',data_9.shape)
print('Data/PEMSD4/pems04.npz数据集中的nan值个数：',nan_9)

print('----------------------')
data_path_10 = os.path.join('./Data/PEMSD8/pems08.npz')
data_10 = np.load(data_path_10)['data'][:, :]
nan_10 = np.sum(np.isnan(data_10))
print('Data/PEMSD8/pems08.npz数据集个数：',data_10.size)
print('Data/PEMSD8/pems08.npz数据集形状：',data_10.shape)
print('Data/PEMSD8/pems08.npz数据集中的nan值个数：',nan_10)

print('----------------------')
data_path_11 = os.path.join('./Data/PEMSD4/pems04_missing.npz')
data_11 = np.load(data_path_11)['data'][:, :]
nan_11 = np.sum(np.isnan(data_11))
print('Data/PEMSD4/pems04_missing.npz数据集个数：',data_11.size)
print('Data/PEMSD4/pems04_missing.npz数据集形状：',data_11.shape)
print('Data/PEMSD4/pems04_missing.npz数据集中的nan值个数：',nan_11)

print('----------------------')
data_path_12 = os.path.join('./Data/PEMSD8/pems08_missing.npz')
data_12 = np.load(data_path_12)['data'][:, :]
nan_12 = np.sum(np.isnan(data_12))
print('Data/PEMSD8/pems08_missing.npz数据集个数：',data_12.size)
print('Data/PEMSD8/pems08_missing.npz数据集形状：',data_12.shape)
print('Data/PEMSD8/pems08_missing.npz数据集中的nan值个数：',nan_12)

print('--------------------------------------')
data04_origin = os.path.join('./origin_data/PEMS04.npz')  # 完整的
data04 = np.load(data04_origin)['data'][:, :]
nan_13=np.sum(np.isnan(data04))
print('最原始的04数据集个数：',data04.size)
print('最原始的04数据集形状：',data04.shape)
print('最原始的04数据集中的nan值个数：',nan_13)

print('--------------------------------------')
data08_origin = os.path.join('./origin_data/PEMS08.npz')  # 完整的
data08 = np.load(data08_origin)['data'][:, :]
nan_14=np.sum(np.isnan(data04))
print('最原始的08数据集个数：',data08.size)
print('最原始的08数据集形状：',data08.shape)
print('最原始的04数据集中的nan值个数：',nan_14)

print('--------------------------------------')
data04_dtw_path = os.path.join('./raw_data/PEMS04/dtw.npy')
data04_dtw = np.load(data04_dtw_path)
print('rawdata/PEMS04/dtw.npy文件：',data04_dtw)
print(data04_dtw.shape)  # (307,307)

print('--------------------------------------')
data08_dtw_path = os.path.join('./raw_data/PEMS08/dtw.npy')
data08_dtw = np.load(data08_dtw_path)
print('rawdata/PEMS08/dtw.npy文件：',data08_dtw)
print(data08_dtw.shape)  # (170,170)

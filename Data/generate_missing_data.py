import os
import numpy as np

#
data_path = 'PEMSD4/pems04.npz'
data = np.load(data_path)  # only the first dimension, traffic flow data
data = data['data'][:, :, 0]
p = 0.7
n0 = data.shape[0]
n1 = data.shape[1]
num = int(n0 * n1 * p)
for i in range(num):
    r = np.random.randint(0, n0)
    c = np.random.randint(0, n1)
    data[r][c] = np.nan

save_file = "PEMSD4/pems04_missing_70.npz"
np.savez(save_file, data=data)

missing_data = np.load(save_file)['data']
# origin_data=np.load(file)['data'][:,:,0]
# # data, scaler = normalize_dataset(missing_data, 'std', False)
# # np.save("PEMSD4/norm_pems04_missing.npy", data)
#
# # 填充
# # data = np.expand_dims(missing_data, axis=-1)
# import pandas as pd
# df_data=pd.DataFrame(missing_data)
# df_data.interpolate(inplace=True)
# df_data.fillna(method="bfill",inplace=True)
# data=df_data.values
# dis=np.linalg.norm(origin_data-data)    # 20421.08533419777








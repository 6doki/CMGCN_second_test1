import numpy as np
from GMF import GMF

import sys

print(sys.path)
model = GMF(8, 12, 10, 1)

origin_data = np.load("../Data/PEMSD4/pems04.npz")['data'][:, :, 0]
comp_data = np.load("../Data/PEMSD4/pems04_comp_data.npz")['data']

mse = np.sqrt((np.mean((origin_data - comp_data) ** 2)))
mae = np.mean(np.abs(origin_data - comp_data))
neg = np.isnan(comp_data).astype(int)
neg_count = np.count_nonzero(neg)  # 非零个数=nan值个数
count_0 = np.sum(np.where(comp_data, 0, 1))
count_n0 = np.count_nonzero(comp_data)

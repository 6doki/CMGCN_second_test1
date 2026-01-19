import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# 假设 N = 3, steps = 4，A 和 A_dtw 是简单的示例矩阵
N = 3
steps = 4

# 定义空间图 A 和 时间图 A_dtw
A = np.array([[1, 0.5, 0.3],
              [0.5, 1, 0.2],
              [0.3, 0.2, 1]])

A_dtw = np.array([[1, 0.1, 0.05],
                  [0.1, 1, 0.08],
                  [0.05, 0.08, 1]])

# 初始化 (N * steps, N * steps) 的全零矩阵
adj = np.zeros([N * steps, N * steps])

# 填充矩阵，空间图与时间图融合
for i in range(steps):
    if (i == 1) or (i == 2):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A  # 填充 A
    else:
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw  # 填充 A_dtw

# 增加时间步之间的连接
for i in range(N):
    for k in range(steps - 1):
        adj[k * N + i, (k + 1) * N + i] = 1
        adj[(k + 1) * N + i, k * N + i] = 1

# 可视化加入时间步之间节点连接后的矩阵
plt.figure(figsize=(6, 6))
plt.imshow(adj, cmap="Blues", interpolation="none")
plt.colorbar(label="Connection Strength")
plt.title("Adjacency Matrix with Temporal Connections")
plt.savefig('/home/xuke/zzh/CMGCN_1001/b.png')
print("OK!")
# plt.show()
# print(mpl.get_backend())
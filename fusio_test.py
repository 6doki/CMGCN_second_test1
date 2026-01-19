import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

def visualize_adj_matrix(adj):
    G = nx.from_numpy_matrix(adj)  # 从邻接矩阵构建图
    pos = nx.spring_layout(G)  # 使用弹簧布局
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title('Visualization of the Constructed Adjacency Matrix')
    # plt.show()
    plt.savefig('/home/xuke/zzh/CMGCN_1001/fusion_test.png')
    print("fusion_test OK!")

def construct_adj_fusion(A, A_dtw, steps):
    N = len(A)
    adj = np.zeros([N * steps] * 2)  # 创建一个新的零矩阵[N * steps, N * steps]

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A  # 填充子矩阵
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    adj[3 * N: 4 * N, 0: N] = A_dtw  # 建立长距离依赖连接
    adj[0: N, 3 * N: 4 * N] = A_dtw

    adj[2 * N: 3 * N, 0: N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[0: N, 2 * N: 3 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]

    for i in range(len(adj)):  # 添加自连接
        adj[i, i] = 1

    return adj

# 示例邻接矩阵
N = 4
steps = 4
A = np.array([[0, 1, 0, 0],
              [1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0]])
A_dtw = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0]])

adj = construct_adj_fusion(A, A_dtw, steps)
visualize_adj_matrix(adj)

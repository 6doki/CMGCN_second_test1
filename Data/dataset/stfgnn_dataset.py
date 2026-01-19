import os
import pandas as pd
import numpy as np
import pickle
import torch
import time
from torch.autograd import Variable

from .multi_step_dataset import MultiStepDataset


def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    data:输入的数据，一个包含时间序列的数组或者矩阵
    ntr:处理后返回的数据的时间步长
    N：节点数量
    '''
    # data=data.as_matrix()
    '''
    将data数据整形为三维矩阵:
    (1) 形状 (天数,每天的时间步数,节点数)
    (2) -1表示自动推断天数
    (3) 288表示一天的时间步
    (4) 取前ntr天的数据
    '''
    data = np.reshape(data, [-1, 288, N])  # （91,288,358）
    return data[0:ntr]  # （54,288,358）


'''对输入的数据进行标准化处理'''
def normalize(a):
    '''
    np.mean函数计算输入数据a在每一行上的平均值，axis=1表示沿着每一行计算
    keepdims=True表示保持原始数据的维度，返回的mu是一个与输入a形状相同的二维数组
    '''
    mu = np.mean(a, axis=1, keepdims=True)
    '''
    np.std函数计算输入数据a在每一行上的标准差，axis=1表示沿着每一行计算
    keepdims=True表示保持原始数据的维度，返回的std是一个与输入a形状相同的二维数组
    '''
    std = np.std(a, axis=1, keepdims=True)
    '''
    返回标准化后的数据，即将输入数据a的每个元素减去对应行的平均值mu，再除以对应行的标准差std
    返回的数据形状与输入a相同，即为一个二维数组
    '''
    return (a - mu) / std


def compute_dtw(a, b, order=1, Ts=12, normal=True):
    # order:用于计算dtw距离时的范式的阶数，默认为1，表示使用曼哈顿距离(绝对值和)，当order=2时，就是使用欧几里得距离
    # Ts:时间窗口的大小，用于限制DTW搜索的范围(也就是论文中的蓝色区域宽度L，蓝色区域为)
    if normal: #归一化，通常是为了消除序列间幅度差异的影响，确保它们的比较更有公平性
        a = normalize(a)  # （54,288）
        b = normalize(b)
    T0 = a.shape[1]  # 288
    d = np.reshape(a, [-1, 1, T0]) - np.reshape(b, [-1, T0, 1])  # dist matrix: (54, 288, 288)
    d = np.linalg.norm(d, axis=0, ord=order)  # 范式1max(sum(abs(x), axis=0))（绝对值和的最大值）: (288,288)
    D = np.zeros([T0, T0])  # dtw matrix: (288,288)
    for i in range(T0):
        for j in range(max(0, i - Ts), min(T0, i + Ts + 1)):
            if (i == 0) and (j == 0):
                D[i, j] = d[i, j] ** order
                continue
            if (i == 0):
                D[i, j] = d[i, j] ** order + D[i, j - 1]
                continue
            if (j == 0):
                D[i, j] = d[i, j] ** order + D[i - 1, j]
                continue
            if (j == i - Ts):
                D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i - 1, j])
                continue
            if (j == i + Ts):
                D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i, j - 1])
                continue
            D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[-1, -1] ** (1.0 / order)  # value


def construct_adj_fusion(A, A_dtw, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

    ----------
    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''

    '''
    该融合机制综合了时间依赖和长距离依赖的影响：
    1，对角块表示同一时间步内的节点连接，即每个时间步内的节点之间的关系。
    2，相邻时间步之间的块表示跨时间步的节点连接，这展示了相邻时间步之间的时间依赖性。
    3，最后四行赋值代码引入了长距离依赖，它们在非连续时间步之间建立了新的连接
    '''
    N = len(A)
    adj = np.zeros([N * steps] * 2)  # 创建一个新的零矩阵[N * steps,N * steps]，其中step=4

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A     # (N:2N,N:2N)  (2N:3N, 2N:3N)
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw     # (0:N, 0:N)   (3N:4N, 3N:4N)
    print('融合过程1完成！！！')
    # '''
    for i in range(N):
        for k in range(steps - 1):
            '''
            k * N + i                            表示时间步k中的节点i
            (k + 1) * N + i                      表示时间步k+1中相同的节点i
            adj[k * N + i, (k + 1) * N + i] = 1  表示节点i在相邻时间步有连接
            '''
            adj[k * N + i, (k + 1) * N + i] = 1
            # 保持矩阵对称性质
            adj[(k + 1) * N + i, k * N + i] = 1
    print('融合过程2完成！！！')
    # '''
    '''
    以下两行代码：在邻接矩阵中为时间步之间的长距离依赖创建连接，具体是时间步0和时间步3之间的节点建立连接
    '''
    adj[3 * N: 4 * N, 0:  N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N] # 连接时间步3到时间步0
    adj[0: N, 3 * N: 4 * N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N]  # 连接时间步0到时间步3
    print('融合过程3完成！！！')
    '''
    对称！！！！！！！
    将时间步0和时间步1之间的邻接关系（即adj[0 * N: 1 * N, 1 * N: 2 * N]）复制到时间步2和时间步0之间的邻接区域（即adj[2 * N: 3 * N, 0: N]）
    将时间步0和时间步1之间的邻接关系（即adj[0 * N: 1 * N, 1 * N: 2 * N]）复制到时间步0和时间步2之间的邻接区域（即adj[0: N, 2 * N: 3 * N]）
    '''
    adj[2 * N: 3 * N, 0: N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[0: N, 2 * N: 3 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    print('融合过程4完成！！！')
    '''
    对称！！！！！！！！
    将时间步0和时间步1之间的邻接关系复制到时间步1和时间步3之间的邻接区域
    将时间步0和时间步1之间的邻接关系复制到时间步3和时间步1之间的邻接区域
    '''
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    print('融合过程5完成！！！')

    for i in range(len(adj)):  # 为每个节点添加自身连接
        adj[i, i] = 1
    print('融合过程6完成！！！')
    return adj  # (4N,4N)

def construct_adj_fusion_k(AS, AT, step):
    """
    构造时空融合邻接矩阵，维度为 3N × 3N。

    参数：
        AS (np.ndarray): 空间邻接矩阵，维度 N × N。
        AT (np.ndarray): 时间相似性矩阵，维度 N × N。
        time_connects (bool): 是否在非对角位置添加时间连接（单位矩阵）。

    返回：
        adj_fusion (np.ndarray): 时空融合邻接矩阵，维度 3N × 3N。
    """
    N = AS.shape[0]  # 节点数
    adj_fusion = np.zeros((step * N, step * N))  # 初始化 3N × 3N 的零矩阵

    # ------------------------------------------------
    # 1. 填充四个角落的时间图 AT
    # ------------------------------------------------
    # 左上角 (0,0)
    adj_fusion[0:N, 0:N] = AT
    # 右上角 (0,2)
    adj_fusion[0:N, 2 * N:3 * N] = AT
    # 左下角 (2,0)
    adj_fusion[2 * N:3 * N, 0:N] = AT
    # 右下角 (2,2)
    adj_fusion[2 * N:3 * N, 2 * N:3 * N] = AT
    print("新融合过程1完成！！！")

    # ------------------------------------------------
    # 2. 填充中心块的空间图 AS (1,1)
    # ------------------------------------------------
    adj_fusion[N:2 * N, N:2 * N] = AS
    print("新融合过程2完成！！！")

    # ------------------------------------------------
    # 3. 填充其他位置的时间连接图（单位矩阵）
    # ------------------------------------------------
        # (0,1)、 (1,0)、 (1,2)、 (2,1) 块填充单位矩阵
    adj_fusion[0:N, N:2 * N] = np.eye(N)  # 块 (0,1)
    adj_fusion[N:2 * N, 0:N] = np.eye(N)  # 块 (1,0)
    adj_fusion[N:2 * N, 2 * N:3 * N] = np.eye(N)  # 块 (1,2)
    adj_fusion[2 * N:3 * N, N:2 * N] = np.eye(N)  # 块 (2,1)
    print("新融合过程3完成！！！")
    return adj_fusion


'''
STFGNNDataset：用于构建时空图卷积神经网络（STFGNN）的数据集
（1）数据加载
（2）时间相似性矩阵的构建
（3）邻接矩阵构建
（4）获取数据特征
'''
class STFGNNDataset(MultiStepDataset):
    '''
    空间图：继承于父类MultiStepDataset
    时间图：DTW生成
    '''
    def __init__(self, config):
        super().__init__(config)  # 确保继承自MultiStepDataset的初始化操作得到执行
        self.strides = self.config.get("strides", 4) # 步长
        self.order = self.config.get("order", 1) # 计算DTW距离的时候的选取哪种距离（1：曼哈顿距离 2：欧式距离）
        self.lag = self.config.get("lag", 12) # 模型滞后期（表示时间上的延迟） -----------估计用不上？？？！！！！
        self.period = self.config.get("period", 288) # 一天288个时间步 -----------好像没用上？？？！！！
        self.sparsity = self.config.get("sparsity", 0.01) # 稀疏度，邻接矩阵中非零元素占比 -----------好像没用上？？？！！！
        self.train_rate = self.config.get("train_rate", 0.6) # 训练数据比例
        self.adj_percent=self.config.get("adj_percent", 0.01) # 邻接矩阵稀疏度，节点数*稀疏度=top=k （与该节点最相似的前top个节点）
        self.adj_mx = torch.FloatTensor(self._construct_adj())  # SG和TG融合后的graph：(4N,4N)
        # self.adj_mx = torch.FloatTensor(self._construct_adj())
        # self.adj_mx = torch.randn((1432, 1432))

    # 构建节点之间的邻接矩阵（节点i与其最相似的top个节点）:(N,N) ---------> 时间相似性图
    def _construct_dtw(self):
        print("------------------进入STFGNNDataset类_construct_dtw方法：-------------------")
        # 如果已经有相似性图数据文件，则直接加载。否则计算新的相似性图
        if os.path.exists(self.dtw_file):  # 直接导入
            w_adj = np.load(self.dtw_file)
        else:
            '''
            三维数组：(时间步总数,节点数量,节点特征)
            data[0]:时间步总数
            data[1]:节点数量
            data[2]:节点特征数量
            '''
            data = self.rawdat[:, :, 0]  # 取数据集中的所有时间步的每个节点的第一个节点特征  data:(时间步总数,节点数) 二维数据
            total_day = data.shape[0] / 288  # 时间步总数/每天的时间步数量=总的天数:91 days
            train_day = int(total_day * 0.6)  # 将数据中的60%用于训练 train day: 54 test day: 37
            n_route = data.shape[1]  # 节点数量 358
            xtr = gen_data(data, train_day, n_route)  # (天数,每天的时间步数,节点数量)（54,288,358）只取前train_day天的数据
            # print(np.shape(xtr))

            T0 = 288     # 每天的时间步
            T = 12       # 后续处理使用的时间步长
            N = n_route  # 节点数量
            d = np.zeros([N, N]) # 初始化相似度矩阵d

            for i in range(N):
                for j in range(i + 1, N):
                    '''
                    xtr[:, :, i]的含义是提取节点i的所有时序数据，形状是(天数，每天的时间步数)
                    '''
                    d[i, j] = compute_dtw(xtr[:, :, i], xtr[:, :, j])  # 计算节点i和节点j的DTW动态规整距离

            dtw = d + d.T  # 使矩阵对称
            print("节点之间的相似性矩阵（时间图）已经生成！")

            n = dtw.shape[0]                # 相似性矩阵节点个数
            w_adj = np.zeros([n, n])        # 邻接矩阵：存储节点之间的连接关系
            adj_percent = self.adj_percent  # 确定邻接矩阵的密度,即每个节点与其最近的top个节点连接,节点*密度=top=k,也就是论文提到的k值(找出与某个节点相似性较高的前K个节点)
            print("时间图邻接矩阵密度adj_percent:",adj_percent)
            top = int(n * adj_percent)
            print("top:", adj_percent)

            for i in range(dtw.shape[0]):  # 每个节点与topk个节点的关系设置为1
                '''
                dtw[i,:] 表示第i个节点与所有其他节点之间的相似度，返回一个一维数组，长度为N
                argsort():对dtw[i,:]升序排序，返回相似度从低到高的节点索引，越小的值表示节点越相似
                dtw[i, :].argsort()[0:top] 选出相似性最高的前top个节点的索引
                '''
                a = dtw[i, :].argsort()[0:top]
                for j in range(top):
                    w_adj[i, a[j]] = 1      # 节点i与其最相似的top个节点建立连接

            for i in range(n):  # 对称
                for j in range(n):
                    if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] == 0):
                        w_adj[i][j] = 1
                    if (i == j):
                        w_adj[i][j] = 1

        print("时间图self.dtw已经生成！！！")

        np.save(self.dtw_file, w_adj)   # w_adj --> self.dtw_file （w_adj保存在文件中，可以直接加载，无需重新计算）
        print("w_adj矩阵中总的节点数量(w_adj.shape[0]): ", w_adj.shape[0])

        n = w_adj.shape[0]
        # n = self.dtw_file.shape[0]
        print("时间图矩阵的稀疏度: ", len(w_adj.nonzero()[0]) / (n * n))  # 计算稀疏度
        self.dtw = w_adj                # self.dtw_file --> self.dtw 赋值给类属性，方便调用（不用每次都加载文件）
        print("self.dtw矩阵中总的节点数量(self.dtw.shape[0]): ", self.dtw.shape[0])

        print("------------------退出STFGNNDataset类_construct_dtw方法：-------------------")

    # 构建融合后的graph： （4N,4N)
    def _construct_adj(self):
        """
        构建local 时空图
        :param A: np.ndarray, adjacency matrix, shape is (N, N)
        :param steps: 选择几个时间步来构建图
        :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
        """

        print("------------------进入STFGNNDataset类_construct_adj方法：-------------------")

        self._construct_dtw()  # 构建self.dtw: (N,N)  时间相似性图
        print("第一步：调用STFGNNDataset类_construct_dtw方法 生成时间相似图")
        print("self.adj_mx.shape",self.adj_mx.shape)
        print("self.dtw.shape", self.dtw.shape)
        adj_mx = construct_adj_fusion(self.adj_mx, self.dtw, self.strides)  # (4N,4N)
        # adj_mx = construct_adj_fusion(self.adj_mx, self.adj_mx, self.strides)  # (4N,4N)
        # adj_mx = construct_adj_fusion_k(self.adj_mx, self.dtw, self.strides)  # (3N,3N)
        print("The shape of localized adjacency matrix: {}".format(adj_mx.shape), flush=True)
        print("融合之后的大矩阵：",adj_mx)
        print("------------------退出STFGNNDataset类_construct_adj方法：-------------------")
        return adj_mx


    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据
        这些DataLoader是用来迭代访问数据集的工具
        Returns:
            tuple: tuple contains:
                train_dataloader:
                eval_dataloader:
                test_dataloader:
        """
        # 加载数据集

        return self.data["train_loader"], self.data["valid_loader"], self.data["test_loader"]

    def get_data_feature(self):
        """
        返回数据集特征（一个字典），子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {
            "scaler": self.data["scaler"], # 数据缩放器 对原始数据进行标准化和归一化操作的工具。
            "adj_mx": self.adj_mx, # 融合图
            "num_batches": self.data['num_batches'] # 训练过程一次模型训练需要多少个批次的数据
        }

        return feature

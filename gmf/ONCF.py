import torch.nn as nn
import torch
from torch.nn import init
import numpy as np


class ONCF(nn.Module):
    def __init__(self, embedding_size, n_times, n_nodes, n_output=1):  # n_times和n_nodes表示时间点和节点的总数
        super(ONCF, self).__init__()
        self.embedding_size = embedding_size
        self.n_times = n_times
        self.n_nodes = n_nodes
        self.n_output = n_output

        # 矩阵分解的初始目标矩阵
        self.time_embedding_layer = nn.Embedding(self.n_times, self.embedding_size)  # 时间的嵌入层
        self.node_embedding_layer = nn.Embedding(self.n_nodes, self.embedding_size)  # 节点的嵌入层
        self._set_normalInit(has_bias=False)

        # cnn setting
        self.channel_size = 16  # 32
        self.kernel_size = 2
        self.strides = 2
        '''四层二维卷积'''
        self.cnn = nn.Sequential(
            # batch_size * 1 * 256 * 256
            # nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides),
            # nn.ReLU(),
            # nn.Dropout2d(p=0.2, inplace=False),
            # batch_size * 1 * 128 * 128
            # nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides),
            # nn.ReLU(),
            # nn.Dropout2d(p=0.2, inplace=False),
            # batch_size * 1 * 64 * 64
            # nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides),
            # nn.ReLU(),
            # nn.Dropout2d(p=0.2, inplace=False),
            # batch_size * 32 * 32 * 32
            # nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            # nn.ReLU(),
            # nn.Dropout2d(p=0.2, inplace=False),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            nn.Dropout2d(p=0.2, inplace=False),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            nn.Dropout2d(p=0.2, inplace=False),
            # batch_size * 32 * 2 * 2
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            nn.Dropout2d(p=0.2, inplace=False),
            # batch_size * 32 * 1 * 1
        )
# '''
#         #两个二维扩张卷积层，扩展卷积的dilation参数设置为2，增加感受野，捕获更大范围的特征
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, self.channel_size, kernel_size=3, dilation=2, padding=2),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.2, inplace=False),
#             nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, dilation=2, padding=2),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.2, inplace=False),
#         )
# '''
        # fully-connected layer, used to predict
        self.fc = nn.Linear(self.channel_size, self.n_output)

        # self.fc = nn.Linear(self.embedding_size * self.embedding_size, self.n_output)

        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.channel_size*((self.embedding_size // (2 ** 5)) ** 2), 1)
        # )
        # self._set_uniformInit(has_bias=True,parameter=np.sqrt(self.embedding_size))

    # dropout

    #         self.drop_prop = 0.5
    #         self.dropout = nn.Dropout(drop_prop)

    def forward(self, x1, x2):  #

        # convert float to int
        # x1 = list(map(int, x1))
        # x2 = list(map(int, x2))

        x1 = self.time_embedding_layer(x1) # (n,64)
        x2 = self.node_embedding_layer(x2) # (n,64)

        # inner product
        # prediction = torch.sum(torch.mul(x1, x2), dim=1)
        # outer product
        # interaction_map = torch.ger(x1, x2) # ger is 1d
        # interaction_map = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1))   # (n,64,1)    (n,1,64)  -> (n,64,64)
        interaction_map = torch.einsum('ni,nj->nij', x1, x2)  # 计算外积
        interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))  # (n,1,64,64)

        # cnn 经过多层卷积和激活函数，数据最终被压缩到1*1的空间维度，但是通道数从1增加到了16
        feature_map = self.cnn(interaction_map)  # output: batch_size * 16 * 1 * 1
        feature_vec = feature_map.view((-1, self.channel_size))  # 重塑为(n,32)

        # fc
        # feature_vec=interaction_map.view((-1,self.embedding_size*self.embedding_size))
        prediction = self.fc(feature_vec)  # 经过全连接层之后为(n,1)
        prediction = torch.relu(prediction)  # 激活函数之后为(n,1)
        prediction = prediction.view((-1))  # 重塑为一维张量(n,),即包含n个预测值的一维张量

        return prediction

    def _set_normalInit(self, parameter=[0, 1], has_bias=True):  # 正态分布
        init.normal_(self.time_embedding_layer.weight, mean=parameter[0], std=parameter[1])
        init.normal_(self.node_embedding_layer.weight, mean=parameter[0], std=parameter[1])
        if has_bias:
            init.normal_(self.time_embedding_layer.bias, mean=parameter[0], std=parameter[1])
            init.normal_(self.node_embedding_layer.bias, mean=parameter[0], std=parameter[1])

    def _set_uniformInit(self, parameter=5, has_bias=True):  # 均匀分布
        init.uniform_(self.fc.weight, a=-parameter, b=parameter)
        if has_bias:
            init.uniform_(self.fc.bias, a=-parameter, b=parameter)


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))

        #         print('loss:', loss)
        return loss

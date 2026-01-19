import torch.nn as nn
import torch
from torch.nn import init
import numpy as np

# 时间和空间向量嵌入之后，通过卷积层提取特征，然后通过线性层进行预测
class GMF(nn.Module):
    def __init__(self, n_factor, n_times, n_samples, n_output=1):
        super(GMF, self).__init__()
        self.n_factor = n_factor    # 嵌入向量的维度
        self.n_times = n_times      # 时间向量的数量
        self.n_samples = n_samples  # 样本向量的数量（空间？）
        self.n_output = n_output    # 输出维度：1 表示预测的目标变量数量

        # embedding layer
        self.gmf_time_embedding_layer = nn.Embedding(self.n_times, self.n_factor)      # 输入维度n_times  输出维度n_factor
        self.gmf_sample_embedding_layer = nn.Embedding(self.n_samples, self.n_factor)  # 输入维度n_samples  输出维度n_factor
        self._set_normalInit(has_bias=False)
        # output layer
        self.channel_size=16
        '''多个nn.conv1d提取空间特征'''
        self.cnn=nn.Sequential(
            # batch_size * channel_size * 16
            nn.Conv1d(1,self.channel_size,2,stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.2,inplace=False),
            # batch_size * channel_size * 8
            nn.Conv1d(self.channel_size,self.channel_size,2,stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.2,inplace=False),
            # batch_size * channel_size * 4
            nn.Conv1d(self.channel_size,self.channel_size,2,stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.2,inplace=False),
            # batch_size * channel_size * 2
            nn.Conv1d(self.channel_size,self.channel_size,2,stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.2,inplace=False),
            # batch_size * channel_size * 1

        )
        self.predict = nn.Linear(self.channel_size, self.n_output)  # 输入channel_size 输出n_output
        self._set_uniformInit(has_bias=True, parameter=np.sqrt(n_factor))

    def forward(self, x1, x2):
        x1 = self.gmf_time_embedding_layer(x1)  # n,k     # input_length=1
        print('时间嵌入x1.size:',x1.size())
        x2 = self.gmf_sample_embedding_layer(x2)  # n,k
        print('样本嵌入x2 size:',x2.size())
        x=torch.mul(x1,x2) # 两个嵌入向量逐元素相乘
        print('时间向量与空间向量逐元素相乘 x=mul(x1,x2) size:',x.size())
        x=x.view((-1,1,self.n_factor))  # n,1,n_factor 准备送入1D卷积层 其中batch_size=n
        print('变换x,将x重塑为三维张量之后x：',x.size())
        # x = torch.flatten(torch.mul(x1, x2), start_dim=1)  # element-wise multiply
        # print(torch.mul(x1, x2).size())

        x=self.cnn(x)
        print('经过CNN之后，x的形状x.size:',x.size())
        x=x.view((-1,self.channel_size))
        # print(x.size())   # batch_size * embedding_size
        output = torch.relu(self.predict(x))
        return output

    def _set_normalInit(self, parameter=[0, 1], has_bias=True):  # 正态分布
        init.normal_(self.gmf_time_embedding_layer.weight, mean=parameter[0], std=parameter[1])
        init.normal_(self.gmf_sample_embedding_layer.weight, mean=parameter[0], std=parameter[1])
        if has_bias:
            init.normal_(self.gmf_time_embedding_layer.bias, mean=parameter[0], std=parameter[1])
            init.normal_(self.gmf_sample_embedding_layer.bias, mean=parameter[0], std=parameter[1])

    def _set_uniformInit(self, parameter=5, has_bias=True):  # 均匀分布
        init.uniform_(self.predict.weight, a=-parameter, b=parameter)
        if has_bias:
            init.uniform_(self.predict.bias, a=-parameter, b=parameter)

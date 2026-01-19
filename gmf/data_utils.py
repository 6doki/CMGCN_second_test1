import torch.utils.data as data
import numpy as np


# 在构建pytorch数据加载器的时候使用该类，方便将自定义数据集封装为Dataset对象，然后利用pytorch的DataLoader类来批量加载数据，并且进行后续的训练或预测

class GMFdata(data.Dataset):
    '''
    __init__方法：将传入的三个数据集time、samples和flows保存为类的成员变量
    '''
    def __init__(self, times, samples, flows):
        super(GMFdata, self).__init__()
        self.times = times
        self.samples = samples
        self.flows = flows
    '''
    __len__方法：返回数据集time的长度，也就是第一个维度的大小，即数据集中样本的数量
    '''
    def __len__(self):
        return self.times.shape[0]
    '''
    __getitem__方法：根据传入的索引item返回对应的times、samples和flows
    '''
    def __getitem__(self, item):
        return self.times[item], self.samples[item], self.flows[item]

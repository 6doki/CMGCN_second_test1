import os
import numpy as np


def load_st_dataset(dataset, miss_ratio):
    print("进入load_dataset.py中load_st_dataset方法：")
    '''
    功能：根据给定的数据集名称和缺失率加载相应的数据文件(Data文件夹下的npz文件)
    :param dataset: 数据集名称
    :param miss_ratio: 缺失率
    :return:
    '''
    # output B, N, D
    if dataset == 'PEMSD4':
        if miss_ratio == 50:
            # data_path = os.path.join('../Data/PEMSD4/pems04_missing_50.npz')
            data_path = os.path.join('./Data/PEMSD4/pems04_missing_50.npz')
        else:
            # data_path = os.path.join('../Data/PEMSD4/pems04_missing.npz')
            data_path = os.path.join('./Data/PEMSD4/pems04_missing.npz')
        data = np.load(data_path)['data'][:, :]  # only the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        if miss_ratio == 50:
            # data_path = os.path.join('../Data/PEMSD8/pems08_missing_50.npz')
            data_path = os.path.join('./Data/PEMSD8/pems08_missing_50.npz')
        else:
            data_path = os.path.join('./Data/PEMSD8/pems08_missing.npz')
            # data_path = os.path.join('../Data/PEMSD8/pems08_missing.npz')
        data = np.load(data_path)['data'][:, :]  # only the first dimension, traffic flow data
    else:
        raise ValueError
    # 形状、最大值、最小值、平均值和中位数
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

def load_st_dataset_mytest(dataset, miss_ratio):
    print("进入load_dataset.py中load_st_dataset_mytest方法：")
    '''
    功能：根据给定的数据集名称和缺失率加载相应的数据文件
    :param dataset: 数据集名称
    :param miss_ratio: 缺失率
    :return:
    '''
    # output B, N, D
    if dataset == 'PEMSD4':
        if miss_ratio == 10:
            # data_path = os.path.join('../Data/PEMSD4/pems04_missing_50.npz')
            # data_path = os.path.join('./missing_data_generate/PEMS04_missing_10.npz')
            data_path = os.path.join('./raw_data/PEMS04_10/PEMS04_10.npz')
        else:
            # data_path = os.path.join('../Data/PEMSD4/pems04_missing.npz')
            # data_path = os.path.join('./missing_data_generate/PEMS04_missing_30.npz')
            data_path = os.path.join('./raw_data/PEMS04_30/PEMS04_30.npz')
        data = np.load(data_path)['data'][:, :]  # only the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        if miss_ratio == 10:
            # data_path = os.path.join('../Data/PEMSD8/pems08_missing_50.npz')
            # data_path = os.path.join('./missing_data_generate/PEMS08_missing_10.npz')
            data_path = os.path.join('./raw_data/PEMS08_10/PEMS08_10.npz')
        else:
            # data_path = os.path.join('./missing_data_generate/PEMS08_missing_30.npz')
            # data_path = os.path.join('../Data/PEMSD8/pems08_missing.npz')
            data_path = os.path.join('./raw_data/PEMS08_30/PEMS08_30.npz')
        data = np.load(data_path)['data'][:, :]  # only the first dimension, traffic flow data
    else:
        raise ValueError
    # 形状、最大值、最小值、平均值和中位数
    print('已经加载 %s 数据集，其形状shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

def load_st_dataset_mytest_test(dataset, miss_ratio):
    print("进入load_dataset.py中load_st_dataset_mytest方法：")
    '''
    功能：根据给定的数据集名称和缺失率加载相应的数据文件
    :param dataset: 数据集名称
    :param miss_ratio: 缺失率
    :return:
    '''
    # output B, N, D
    if dataset == 'PEMSD4':
        if miss_ratio == 10:
            # data_path = os.path.join('../Data/PEMSD4/pems04_missing_50.npz')
            data_path = os.path.join('./miss_nan/PEMS04_miss_nan_10.npz')
            # data_path = os.path.join('./raw_data/PEMS04_10/PEMS04_10.npz')
        else:
            # data_path = os.path.join('../Data/PEMSD4/pems04_missing.npz')
            data_path = os.path.join('./miss_nan/PEMS04_miss_nan_30.npz')
            # data_path = os.path.join('./raw_data/PEMS04_30/PEMS04_30.npz')
        data = np.load(data_path)['data'][:, :]  # only the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        if miss_ratio == 10:
            # data_path = os.path.join('../Data/PEMSD8/pems08_missing_50.npz')
            data_path = os.path.join('./miss_nan/PEMS08_miss_nan_10.npz')
            # data_path = os.path.join('./raw_data/PEMS08_10/PEMS08_10.npz')
        else:
            # data_path = os.path.join('./missing_data_generate/PEMS08_missing_30.npz')
            data_path = os.path.join('./miss_nan/PEMS08_miss_nan_30.npz')
            # data_path = os.path.join('./raw_data/PEMS08_30/PEMS08_30.npz')
        data = np.load(data_path)['data'][:, :]  # only the first dimension, traffic flow data
    else:
        raise ValueError
    # 形状、最大值、最小值、平均值和中位数
    print('已经加载 %s 数据集，其形状shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

if __name__ == '__main__':
    # dataset = load_st_dataset('PEMSD8',30)
    dataset_410 = load_st_dataset_mytest('PEMSD4', 10)
    dataset_430 = load_st_dataset_mytest('PEMSD4', 30)
    dataset_810 = load_st_dataset_mytest('PEMSD8', 10)
    dataset_830 = load_st_dataset_mytest('PEMSD8', 30)

    dataset_nan_410 = load_st_dataset_mytest_test('PEMSD4', 10)
    dataset_nan_430 = load_st_dataset_mytest_test('PEMSD4', 30)
    dataset_nan_810 = load_st_dataset_mytest_test('PEMSD8', 10)
    dataset_nan_830 = load_st_dataset_mytest_test('PEMSD8', 30)

    print('测试410数据集是否存在nan：')
    # nan1 = np.isnan(dataset_410)  # 检查数据集中是否存在”nan“值，并将结果存储在变量nan中
    # nan_count1 = np.sum(np.isnan(dataset_410))
    # print(nan_count1)
    print(dataset_410.size)
    zero1=dataset_410==0
    zero_count_1=np.sum(zero1)
    print(zero_count_1)

    print('测试430数据集是否存在nan：')
    nan2 = np.isnan(dataset_430)
    nan_count2 = np.sum(np.isnan(dataset_430))
    print(nan_count2)

    print('测试810数据集是否存在nan：')
    nan3 = np.isnan(dataset_810)
    nan_count3 = np.sum(np.isnan(dataset_810))
    print(nan_count3)

    print('测试830数据集是否存在nan：')
    nan4 = np.isnan(dataset_830)
    nan_count4 = np.sum(np.isnan(dataset_830))
    print(nan_count4)

    # print('测试4数据集是否存在nan：')
    # data_path_4=os.path.join('./raw_data/PEMS04/PEMS04.npz')
    # dataset_4 = np.load(data_path_4)['data'][:, :]
    # nan5=np.isnan(dataset_4)
    # nan_count5 = np.sum(np.isnan(dataset_4))
    # print(nan_count5)
    #
    # print('测试PEMS04_missing_10数据集是否存在nan：')
    # data_path_410=os.path.join('./missing_data_generate/PEMS04_missing_10.npz')
    # dataset_missing_410 = np.load(data_path_410)['data']
    # # nan6=np.isnan(dataset_missing_410)
    # # nan_count6 = np.sum(np.isnan(dataset_missing_410))
    # # print(nan_count6)
    # print(dataset_missing_410.size)
    # zero_6=dataset_missing_410==0
    # zero_count_6=np.sum(zero_6)
    # print(zero_count_6)

    print("测试新生成的PEMS04_miss_nan_10数据集是否存在nan: ")
    new_nan1_count=np.sum(np.isnan(dataset_nan_410))
    print(new_nan1_count)

    print("测试新生成的PEMS04_miss_nan_30数据集是否存在nan: ")
    new_nan2_count=np.sum(np.isnan(dataset_nan_430))
    print(new_nan2_count)

    print("测试新生成的PEMS08_miss_nan_10数据集是否存在nan: ")
    new_nan3_count=np.sum(np.isnan(dataset_nan_810))
    print(new_nan3_count)

    print("测试新生成的PEMS08_miss_nan_30数据集是否存在nan: ")
    new_nan4_count=np.sum(np.isnan(dataset_nan_830))
    print(new_nan4_count)
import numpy as np
import random

# 加载原始 npz 文件
file_path = "./raw_data/PEMS04/PEMS04.npz"
data = np.load(file_path)

# 缺失值处理函数
def add_missing_values(data, missing_percentage):
    """
    对数据添加缺失值。
    :param data: 原始数据数组
    :param missing_percentage: 缺失值百分比（0-100）
    :return: 添加缺失值后的数组
    """
    modified_data = data.copy()  # 防止修改原始数据
    total_elements = modified_data.size # 计算数据的总元素数量
    print('数据集 %s 的总元素数量'%data, total_elements)
    num_missing = int(total_elements * (missing_percentage / 100)) # 计算需要添加的缺失值的数量
    print('数据集 %s 需要添加的缺失值元素数量'%data, num_missing)

    # 随机选择索引添加缺失值
    indices = list(np.ndindex(modified_data.shape)) # 生成一个所有索引的迭代器，用于遍历数组的每个元素的位置。
    random.shuffle(indices) # 随机打乱索引列表
    missing_indices = indices[:num_missing] # 选择前 num_missing 个索引作为缺失值的位置

    for idx in missing_indices:
        modified_data[idx] = 0  # 添加缺失值
    return modified_data

# 保存处理后的数据
def save_processed_data(file_name, processed_data):
    """
    保存处理后的数据到新的 npz 文件。
    :param file_name: 文件名
    :param processed_data: 处理后的数据字典
    """
    np.savez(file_name, **processed_data)

# 添加缺失值并保存文件
processed_data_10 = {}
processed_data_30 = {}

missing_percentages = [10, 30]
for key in data.files:
    array = data[key]
    processed_data_10[key] = add_missing_values(array, 10)
    processed_data_30[key] = add_missing_values(array, 30)

# 保存处理后的文件
save_processed_data("./missing_data_generate/PEMS04_missing_10.npz", processed_data_10)
save_processed_data("./missing_data_generate/PEMS04_missing_30.npz", processed_data_30)

print("已生成包含缺失值的文件：PEMS04_missing_10.npz 和 PEMS04_missing_30.npz")

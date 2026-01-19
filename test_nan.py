import numpy as np

# 加载 npz 文件
file_path = "./missing_data_generate/PEMS08_missing_10.npz"
data = np.load(file_path)

# 查看文件中的内容
print("文件中的键值：", data.files)

# 遍历每个键值对应的数据，检查是否有 NaN
for key in data.files:
    array = data[key]
    if np.isnan(array).any():
        print(f"键值 '{key}' 对应的数据中存在 NaN 值。")
    else:
        print(f"键值 '{key}' 对应的数据中没有 NaN 值。")

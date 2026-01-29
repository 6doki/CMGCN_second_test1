import numpy as np
import os
from tslearn.clustering import KShape

# ================= 配置区域 =================
DATA_PATH = 'model_DAGCN/data/PEMS04/PEMS04.npz'
OUTPUT_PATH = 'model_DAGCN/data/PEMS04/patterns.npy'
NUM_PATTERNS = 16
WINDOW_SIZE = 12
# [修改] 采样率改小，或者我们下面直接限制总数
SAMPLE_RATE = 0.01
MAX_SAMPLES = 20000  # [新增] 限制最大样本数，防止卡死


# ===========================================

def generate_patterns():
    print("----------数据准备（聚类）------------")
    print(f"正在加载数据: {DATA_PATH} ...")

    # 根据文件格式加载
    try:
        data = np.load(DATA_PATH)['data']
    except:
        # 兼容 PEMS04 可能的存储格式
        data = np.load(DATA_PATH)
        if isinstance(data, dict):  # 如果是字典但key不是data
            keys = list(data.keys())
            data = data[keys[0]]

    # 只取流量特征
    if data.ndim == 3:
        flow_data = data[:, :, 0]
    else:
        flow_data = data

    print(f"原始数据形状: {flow_data.shape}")

    segments = []
    T, N = flow_data.shape

    # [修改] 为了速度，只取前 50 个节点的数据进行采样就足够了
    # 交通模式是通用的，不需要所有节点参与训练
    limit_nodes = min(N, 50)
    print(f"仅使用前 {limit_nodes} 个节点进行采样...")

    stride = 1
    for n in range(limit_nodes):
        series = flow_data[:, n]
        # 简单归一化 (Z-Score) 防止死值
        std_val = np.std(series)
        if std_val < 1e-5: continue  # 跳过死节点

        series = (series - np.mean(series)) / std_val

        for t in range(0, T - WINDOW_SIZE, stride):
            if np.random.rand() < SAMPLE_RATE:
                segments.append(series[t: t + WINDOW_SIZE])
                # [新增] 如果攒够了数据就提前退出
                if len(segments) >= MAX_SAMPLES:
                    break
        if len(segments) >= MAX_SAMPLES:
            break

    segments = np.array(segments)
    # 确保维度正确 (N, T, 1)
    if segments.ndim == 2:
        segments = segments[..., np.newaxis]

    print(f"最终用于聚类的片段数量: {segments.shape}")

    if segments.shape[0] < NUM_PATTERNS:
        raise ValueError(f"样本数 ({segments.shape[0]}) 少于聚类数 ({NUM_PATTERNS})，请调大 SAMPLE_RATE")

    # 使用 k-Shape 聚类
    print(f"开始 k-Shape 聚类 (k={NUM_PATTERNS})...")
    # [修改] 减少 n_init 以加快速度
    ks = KShape(n_clusters=NUM_PATTERNS, verbose=True, random_state=42, n_init=1)
    ks.fit(segments)

    patterns = ks.cluster_centers_.squeeze()
    print(f"聚类完成! Patterns 形状: {patterns.shape}")

    np.save(OUTPUT_PATH, patterns)
    print(f"Patterns 已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_patterns()
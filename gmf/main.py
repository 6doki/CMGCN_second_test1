import random
import argparse
import torch
import numpy as np
from GMF import GMF
import data_utils
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import time
from ONCF import ONCF
from load_dataset import load_st_dataset_mytest
from load_dataset import load_st_dataset_mytest_test
from load_dataset import load_st_dataset
import os
import sys

# sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
'''
数据补全：对缺失数据进行预测和填补
1，加载数据
2，生成训练、验证和测试数据集
3，定义模型并进行训练
4，用训练好的模型对缺失的数据进行预测和填补
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Run gmf.")
    parser.add_argument('--device', default="cuda:0", type=str, help="indices of GPUs")
    parser.add_argument('--dataset', nargs='?', default=None, help='Choose a dataset.')
    parser.add_argument('--miss_ratio', default=None, help="dataset miss ratio")
    parser.add_argument('--val_ratio', default=0.0, help='Proportion of validation sets')
    parser.add_argument('--val_flag', default=False, help='set validation dataset or not')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--n_factors', type=int, default=16, help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]', help="Regularization for time and sample embeddings.")
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')  # 0.001
    parser.add_argument('--optimizer', nargs='?', default='adam',help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1, help='Show performance per X iterations')
    parser.add_argument('--save_model', type=int, default=1, help='Whether to save the trained model.')
    return parser.parse_args()


# generate the train set and test set of times, samples and flow
'''
生成时间序列预测任务的数据集合，假设输入的 data 是一个二维的 NumPy 数组，表示一个带有缺失值的数据矩阵，其中每行代表不同时间点的数据，
每列代表不同样本的观测值。目标是根据是否有缺失值，将数据拆分成训练集、验证集和测试集，并处理缺失值。

参数：
data:(m,n) 二维的numpy数组，其中 m 表示时间点的数量，n 表示样本数
val_ratio:用于划分验证集的比例，表示从训练数据中划分多少比例作为验证集

data:
          样本1索引  样本2索引  ...  样本n索引
时间点1索引[ flow                          ]
时间点2索引[                               ]
...
时间点m索引[                               ]

'''
def generate_data(data, val_ratio):
    print("进入gmf/main.py中的generate_data方法：")
    m, n = data.shape
    times_train = [] # 存储训练集的时间点索引
    times_test = [] # 存储测试集的时间点索引
    samples_train = [] # 存储训练集的样本索引
    samples_test = [] # 存储测试集的样本索引
    flows_train = []  # train: value exist 存储训练集中观测值
    flows_test = []  # test: value miss 存储测试集中观测值，对于缺失的值，用0代替

    for i in range(m):
        for j in range(n):
            flow = data[i, j] # 提取每一行每一列的数据
            if ~np.isnan(flow):     # exist 对于每一个数据点，如果存在，则加入训练集
                times_train.append(i)
                samples_train.append(j)
                flows_train.append(flow)
            else:                   # missing 对于每一个数据点，如果缺失，就加入测试集
                times_test.append(i)
                samples_test.append(j)
                flows_test.append(0) # 用0补充

    # 封装成两个列表：训练、测试
    # 这两个列表都包含三个元素，每个元素都是一个一维数组，一维数组的长度是训练集的个数
    traindata = [np.array(times_train), np.array(samples_train), np.array(flows_train)]
    testdata = [np.array(times_test), np.array(samples_test), np.array(flows_test)]

    n_train = int((1 - val_ratio) * traindata[0].shape[0])
    valdata = [td[n_train:] for td in traindata]
    traindata = [td[:n_train] for td in traindata]
    print(
        "在缺失值数据填补模块中：Generate data shaped: \ntraindata--{}, valdata--{}, testdata--{}".format(traindata[0].shape, valdata[0].shape,
                                                                                  testdata[0].shape))
    return traindata, valdata, testdata

'''
作用：将训练集、验证集和测试集转换为 PyTorch 的 DataLoader 对象。DataLoader 是 PyTorch 中用于加载数据的工具
data:输入的数据，是一个包含三个部分的列表，分别是时间（times）、样本索引（samples）和观测值（flows）
batch_size：一次加载多少个样本
shuffle：布尔值，表示是否在加载数据时打乱顺序（一般在训练时设置为 True，在验证或测试时通常设置为 False）
'''
def dataloade(data, batch_size, shuffle):
    print("进入gmf/main.py中的dataloade方法：")
    # cuda = True if torch.cuda.is_available() else False
    # TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    times, samples, flows = data
    # 转为张量
    times, samples, flows = torch.tensor(times).to(torch.int64), torch.tensor(samples).to(torch.int64), torch.FloatTensor(flows)
    times, samples, flows = times.to("cuda:0"), samples.to("cuda:0"), flows.to("cuda:0")
    data = data_utils.GMFdata(times,samples,flows)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle) # 创建一个数据加载器
    return dataloader

'''
参数：
trainloader: 用于训练的 DataLoader，提供训练数据的批次。
valloader: 用于验证的 DataLoader，提供验证数据的批次。
model: 待训练的模型。
loss_fn: 损失函数，用于计算模型预测值与真实值之间的误差
optimizer: 优化器，更新模型的参数（如 Adam, SGD 等）
epochs: 训练的总轮数（epoch）
n_train: 训练集的样本数量（用于计算训练集的平均损失）
n_val: 验证集的样本数量（用于计算验证集的平均损失）
args: 包含其他配置参数的字典，可能包括训练设置、是否启用验证、数据集等
'''
def train(trainloader, valloader, model, loss_fn, optimizer, epochs, n_train, n_val, args):
    print("进入gmf/main.py中的train方法：")
    train_loss_list = [] # 记录每个 epoch 训练损失的列表
    if args.val_flag:
        val_loss_list = []
    best_model = None # 保存当前验证集或训练集损失最小的模型参数
    best_loss = float('inf') # 当前验证集或训练集的最佳损失值，初始化为无穷大
    not_improved_count = 0 # 用于实现早停（early stopping），记录验证损失在连续多少轮内没有改进。
    t0 = time.time() # 记录训练开始的时间
    for epo in range(epochs):
        train_total_loss = 0 # 用于统计当前 epoch 的总训练损失
        # train
        model.train() # 模型设置为训练模式
        t1 = time.time() # 记录当前 epoch 开始时的时间。
        for _, data in enumerate(trainloader): # 遍历训练数据集，每次从 trainloader 中获取一个 batch
            times, samples, flows = data
            optimizer.zero_grad() # 清空梯度
            prediction = model(times, samples) # 前向传播
            loss = loss_fn(prediction.squeeze(), flows) # 计算当前batch的损失
            loss.backward() # 反向传播
            optimizer.step() # 更新模型参数
            train_total_loss += loss.item() # 累加当前batch的损失
        train_epoch_loss = train_total_loss / n_train  # 训练集平均损失
        print("{}:".format(time.asctime(time.localtime(time.time()))), end="")
        print("train epoch {} loss: {}, time:{:.4f}seconds".format(epo + 1, train_epoch_loss, (time.time() - t1)))
        # valida
        t2 = time.time()
        if args.val_flag:
            val_total_loss = epoch_validate(valloader, model, loss_fn) # 调用函数计算验证集的总损失
            #  epoch loss
            val_epoch_loss = val_total_loss / n_val # 验证集平均损失
            print("valida epoch {} loss: {:.4f}, time:{:.1f}mins".format(epo + 1, val_epoch_loss,
                                                                         (time.time() - t2) / 60))
        train_loss_list.append(train_epoch_loss)
        if args.val_flag: # 启用验证集
            val_loss_list.append(val_epoch_loss)
            '''
            如果启用了验证集，且当前验证损失比之前的最优损失还小，则更新最优损失，并保存当前模型参数
            '''
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_model = copy.deepcopy(model.state_dict())
            else:
                not_improved_count += 1
            loss_img(val_loss_list, "Valida")
        else: # 没启用验证集的话 就用训练集
            if train_epoch_loss < best_loss:
                best_loss = train_epoch_loss
                not_improved_count = 0
                best_model = copy.deepcopy(model.state_dict())
                print("最好模型！！!")
            else:
                not_improved_count += 1
        if not_improved_count == 15:
            break
    print("Total time:{:.1f}mins".format((time.time() - t0) / 60))


    '''
    根据不同的数据集（如 PEMSD4 和 PEMSD8）和缺失率设置保存模型的路径
    '''
    if args.dataset == "PEMSD4":
        if args.miss_ratio == 30:
            # path = "../Data/PEMSD4/pems04_best_model_30.pth"
            path = "./Data/PEMSD4/pems04_best_model_30.pth"
        else:
            # path = "../Data/PEMSD4/pems04_best_model.pth"
            path = "./Data/PEMSD4/pems04_best_model.pth"
    elif args.dataset == "PEMSD8":
        if args.miss_ratio == 30:
            # path = "../Data/PEMSD8/pems08_best_model_30.pth"
            path = "./Data/PEMSD8/pems08_best_model_30.pth"
        else:
            # path = "./Data/PEMSD8/pems08_best_model.pth"
            path = "./Data/PEMSD8/pems08_best_model.pth"
    # torch.save(best_model, path)
    loss_img(train_loss_list, "Train")
    return best_model


# 收敛性图
def loss_img(loss_list, train_or_val):
    print("进入gmf/main.py中的loss_img方法：")
    ep = np.arange(1, len(loss_list) + 1)
    plt.plot(ep, loss_list, label="loss/epoch")
    plt.title("{} MSELoss".format(train_or_val))
    plt.xlabel("epoch")
    plt.ylabel("{} loss".format(train_or_val))
    plt.show()


def epoch_validate(valloader, model, loss_fn):
    print("进入gmf/main.py中的epoch_validate方法：")
    model.eval()
    val_total_loss = 0
    with torch.no_grad():
        for _, data in enumerate(valloader):
            times, samples, flows = data
            prediction = model(times, samples)
            val_loss = loss_fn(prediction.squeeze(), flows)
            val_total_loss += val_loss.item()
    return val_total_loss


def test(valloader, model, best_model):
    print("进入gmf/main.py中的test方法：")
    model.load_state_dict(best_model)
    model.eval()
    predictions = []
    # flows_list = []
    t0 = time.time()
    num = 0
    dif_value = 0
    dif_sqr = 0
    with torch.no_grad():
        for _, data in enumerate(valloader):
            times, samples, flows = data
            pred = model(times, samples)
            predictions.append(pred)
            # flows_list.append(flows)
            num += len(flows)
            dif_value += torch.sum(torch.abs(pred - flows)).item()
            dif_sqr += torch.sum((pred - flows) ** 2).item()
    t1 = time.time()
    predictions = torch.squeeze(torch.cat(predictions, dim=0).float())
    print("prediction:", predictions)
    # flows_list = torch.cat(flows_list, dim=0).float()
    # mae = torch.mean(torch.abs(predictions - flows_list))
    # mse = torch.mean((predictions - flows_list) ** 2)
    # mape = torch.mean(np.abs(predictions - flows_list) / flows_list) * 100
    mae = dif_value / num
    rmse = (dif_sqr / num) ** 0.5
    print(
        "{}:mae:{}, rmse:{}, Time:{:.1f}mins".format(time.asctime(time.localtime(time.time())), mae,
                                                     rmse, (t1 - t0) / 60))

'''
数据填充第一步：主要是负责生成预测结果（填充值）
该函数用于在测试集上进行预测并返回填充后的数据:
1,加载最佳模型的权重并进行预测
2,在测试集上进行推理（不计算梯度），得到填充后的预测值
3,将预测结果从 GPU 转移到 CPU，并转换为 numpy 数组，返回填充后的数据
'''
def predict(testloader, model, best_model):
    model.load_state_dict(best_model)
    model.eval()
    predictions = []
    t0 = time.time()
    with torch.no_grad():
        for _, data in enumerate(testloader):
            times, samples, _ = data
            pred = model(times, samples)
            # if pred < 0:
            #     pred = 0
            predictions.append(pred)
    print("{:.1f}mins".format((time.time() - t0) / 60))
    predictions = torch.cat(predictions, dim=0).squeeze()
    predictions = predictions.to("cpu").numpy()
    return predictions

'''
数据填充第二步：将生成的填充值填充到原始数据中，完成数据填充任务
该函数用于将预测结果填充到缺失的数据中:
1,将填充的预测值插回原始数据中的相应位置，完成数据的填充任务
2,返回填充后的数据
'''
def data_comp(data, testdata, prediction):
    test_time, test_sample, _ = testdata
    for i in range(len(prediction)):
        x = test_time[i]
        y = test_sample[i]
        data[x, y] = prediction[i]
    return data


# main抽象填充算法，使其在预测主函数中调用一个函数即可实现数据填充功能
def main(data, dataset, miss_ratio, seed):
    print("main填充函数：")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # load args
    args = parse_args()
    args.dataset = dataset
    args.miss_ratio = miss_ratio
    # if args.dataset == "PEMSD4":
    #     # root_path = "../Data/PEMSD4/pems04"
    #     root_path = "./Data/PEMSD4/pems04"
    # elif dataset == "PEMSD8":
    #     # root_path = "../Data/PEMSD8/pems08"
    #     root_path = "./Data/PEMSD8/pems08"

    # load and generate data
    print("data shape:",data.shape)
    n_times, n_samples= data.shape  # 这是多少？？？
    print("Generate data...")

    traindata, valdata, testdata = generate_data(data, args.val_ratio)  # [times,samples,flows]
    # dataset load
    trainloader = dataloade(traindata, args.batch_size, shuffle=True)
    if args.val_flag:
        valloader = dataloade(valdata, args.batch_size, shuffle=True)
    else:
        valloader = None
    testloader = dataloade(testdata, args.batch_size, shuffle=False)
    n_train, n_val, n_test = traindata[0].shape[0], valdata[0].shape[0], testdata[0].shape[0]
    #  model
    print("Create model...")
    # model = GMF(n_factor=args.n_factors, n_times=n_times, n_samples=n_samples)
    model = ONCF(embedding_size=args.n_factors, n_times=n_times, n_nodes=n_samples)
    model = model.to(args.device)
    loss_fn = nn.MSELoss().to(args.device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    print("Train...")
    # train & validate
    best_model = train(trainloader, valloader, model, loss_fn, optimizer, args.epochs, n_train, n_val, args)
    # load best model
    # if args.miss_ratio == 50:
    #     path = root_path + "_best_model_50.pth"
    # else:
    #     path = root_path + "_best_model.pth"
    # best_model = torch.load(path)
    # test
    if args.val_flag:
        print("Test...")
        test(valloader, model, best_model)
    # predict
    print("Predict...")
    prediction = predict(testloader, model, best_model)
    # data completion
    print("Complete data...")
    miss_data = copy.deepcopy(data)
    print("缺失数据副本miss_data形状：",miss_data.shape)
    comp_data = data_comp(miss_data, testdata, prediction)
    print("数据填充完成！完成后的数据形状是：",comp_data.shape)
    # if args.miss_ratio == 50:
    #     np.savez(root_path + "_comp_data_50.npz", data=comp_data)
    # else:
    #     np.savez(root_path + "_comp_data.npz", data=comp_data)
    print(args.dataset, args.miss_ratio)
    return comp_data


if __name__ == '__main__':
    print("进入gmf/main.py中的main方法：")
    dataset = "PEMSD4"
    miss_ratio = 10
    # data = load_st_dataset(dataset, miss_ratio) # 根据给定的数据集和缺失率加载对应的文件
    data = load_st_dataset_mytest_test(dataset, miss_ratio)
    mask=np.isnan(data)
    comp_data = main(data, dataset, miss_ratio,seed=10)
    print("当前工作文件：",os.getcwd())
    if dataset == "PEMSD4":
        origin_data = np.load("./raw_data/PEMS04/PEMS04.npz")['data'][:, :, 0]
    elif dataset == "PEMSD8":
        origin_data = np.load("./raw_data/PEMS08/PEMS08.npz")['data'][:, :, 0]

    origin_data=origin_data[mask]
    comp_data=comp_data[mask]

    relative_error=np.linalg.norm(np.abs(origin_data-comp_data))/np.linalg.norm(origin_data)
    mae = np.mean(np.abs(origin_data - comp_data))
    rmse = np.sqrt(np.mean((origin_data - comp_data) ** 2))

    print("relative_error: {}, mae: {}, rmse: {}".format(relative_error, mae, rmse))

    # # 保存 comp_data 为 .npz 文件
    # np.savez('./comp_data_04_10.npz', comp_data=comp_data)
    # print("comp_data 已保存为 './comp_data.npz'")


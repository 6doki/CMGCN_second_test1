import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码


# plt.figure(figsize=(13, 10))  # figsize：宽和高，单位是英尺
fig = plt.figure()
y = -0.4

# 构造x轴刻度标签、数据
""""(a) PeMSD4(MR=10) """
labels = list(range(1, 13))
CCGCRN_MAE = [18.31, 18.71, 19.10, 19.45, 19.78, 20.12, 20.48, 20.80, 21.09, 21.40, 21.81, 22.37]
CCGCRN_RMSE = [29.48, 30.2, 30.88, 31.46, 32.02, 32.58, 33.15, 33.68, 34.18, 34.69, 35.28, 36.04]
CMGCN_MAE = [17.62, 18.25, 18.75, 19.07, 19.48, 19.78, 20.15, 20.50, 20.81, 21.09, 21.48, 21.98]
CMGCN_RMSE = [28.69, 29.8, 30.65, 31.15, 31.94, 32.32, 32.87, 33.41, 33.91, 34.29, 34.82, 35.44]

x = np.arange(len(labels))  # x轴刻度标签位置
width = 0.1  # 柱子的宽度

ax1 = fig.add_subplot(221)  # subplot可以规划figure划分为n个子图，但每条subplot命令只会创建一个子图. i(行数) j(列数) n(第几个)
# ax11 = ax1.twinx()
plot11 = ax1.plot(x, CCGCRN_MAE, color="coral", label="CCGCRN_MAE")
plot12 = ax1.plot(x, CCGCRN_RMSE, color="cornflowerblue", label="CCGCRN_RMSE")
plot13 = ax1.plot(x, CMGCN_MAE, color="gold", label="CMGCN_MAE")
plot14 = ax1.plot(x, CMGCN_RMSE, color="grey", label="CMGCN_RMSE")
ax1.set_ylabel('MAE/RMSE')
# ax11.set_ylabel('RMSE')
ax1.set_xlabel('Horizon')
# ax1.set_ylim(30.0, 33.6)
# ax11.set_ylim(0.130, 0.155)
plt.xticks(x, labels=labels)
ax1.set_title('(a) MAE/RMSE on PeMSD4(MR=10)', y=y)
# x轴刻度标签位置不进行计算

""" (b) PeMSD8(MR=10)"""
CCGCRN_MAE = [14.87, 15.20, 15.59, 15.98, 16.39, 16.78, 17.14, 17.44, 17.71, 18.01, 18.46, 19.07]
CCGCRN_RMSE = [23.39, 24.04, 24.69, 25.37, 26.01, 26.63, 27.18, 27.64, 28.06, 28.53, 29.18, 30.07]
CMGCN_MAE = [14.64, 15.20, 15.64, 15.97, 16.35, 16.76, 16.93, 17.21, 17.45, 17.71, 18.05, 18.56]
CMGCN_RMSE = [22.76, 23.81, 24.50, 25.12, 25.80, 26.46, 26.82, 27.26, 27.63, 27.98, 28.45, 29.1]
ax2 = fig.add_subplot(222)
# ax21 = ax2.twinx()
plot21 = ax2.plot(x, CCGCRN_MAE, color="coral", label="CCGCRN_MAE")
plot22 = ax2.plot(x, CCGCRN_RMSE, color="cornflowerblue", label="CCGCRN_RMSE")
plot23 = ax2.plot(x, CMGCN_MAE, color="gold", label="CMGCN_MAE")
plot24 = ax2.plot(x, CMGCN_RMSE, color="grey", label="CMGCN_RMSE")
ax2.set_ylabel('MAE/RMSE')
# ax21.set_ylabel('RMSE')
ax2.set_xlabel('Horizon')
# ax2.set_ylim(30.0, 33.6)
# ax21.set_ylim(0.130, 0.155)
plt.xticks(x, labels=labels)
ax2.set_title('(b) MAE/RMSE on PeMSD8(MR=10)', y=y)

"""(c) PeMSD4(MR=30)"""
CCGCRN_MAE = [18.48, 18.74, 19.05, 19.32, 19.58, 19.87, 20.18, 20.44, 20.70, 20.98, 21.36, 21.87]
CCGCRN_RMSE = [30.07, 30.54, 31.04, 31.48, 31.91, 32.35, 32.80, 33.21, 33.61, 34.03, 34.55, 35.21]
CMGCN_MAE = [17.34, 17.82, 18.19, 18.49, 18.59, 18.85, 19.11, 19.40, 19.71, 20.06, 20.36, 20.79]
CMGCN_RMSE = [28.64, 29.42, 30.11, 30.67, 30.71, 31.11, 31.52, 32.08, 32.51, 33.46, 33.43, 34.07]
ax3 = fig.add_subplot(223)
# ax31 = ax3.twinx()
plot31 = ax3.plot(x, CCGCRN_MAE, color="coral", label="CCGCRN_MAE")
plot32 = ax3.plot(x, CCGCRN_RMSE, color="cornflowerblue", label="CCGCRN_RMSE")
plot33 = ax3.plot(x, CMGCN_MAE, color="gold", label="CMGCN_MAE")
plot34 = ax3.plot(x, CMGCN_RMSE, color="grey", label="CMGCN_RMSE")
ax3.set_ylabel('MAE/RMSE')
# ax31.set_ylabel('RMSE')
ax3.set_xlabel('Horizon')
# ax3.set_ylim(30.0, 33.6)
# ax31.set_ylim(0.130, 0.155)
plt.xticks(x, labels=labels)
ax3.set_title('(c) MAE/RMSE on PeMSD4(MR=30)', y=y)

"""(d) PeMSD8(MR=30)"""
CCGCRN_MAE = [14.86, 15.16, 15.54, 15.92, 16.26, 16.56, 16.90, 17.26, 17.57, 17.84, 18.19, 18.72]
CCGCRN_RMSE = [23.66, 24.18, 24.80, 25.40, 25.91, 26.37, 26.87, 27.39, 27.86, 28.28, 28.81, 29.53]
CMGCN_MAE = [15.31, 15.72, 16.01, 16.20, 16.41, 16.69, 16.89, 17.20, 17.38, 17.59, 17.86, 18.22]
CMGCN_RMSE = [23.83, 24.49, 24.95, 25.34, 25.71, 26.12, 26.44, 26.94, 27.22, 27.54, 27.94, 28.45]
ax4 = fig.add_subplot(224)
# ax41 = ax4.twinx()
plot41 = ax4.plot(x, CCGCRN_MAE, color="coral", label="CCGCRN_MAE")
plot42 = ax4.plot(x, CCGCRN_RMSE, color="cornflowerblue", label="CCGCRN_RMSE")
plot43 = ax4.plot(x, CMGCN_MAE, color="gold", label="CMGCN_MAE")
plot44 = ax4.plot(x, CMGCN_RMSE, color="grey", label="CMGCN_RMSE")
ax4.set_ylabel('MAE/RMSE')
# ax41.set_ylabel('RMSE')
ax4.set_xlabel('Horizon')
# ax4.set_ylim(30.0, 33.6)
# ax41.set_ylim(0.130, 0.155)
plt.xticks(x, labels=labels)
ax4.set_title('(d) MAE/RMSE on PeMSD8(MR=30)', y=y)

pppp = [plot11[0], plot12[0], plot13[0], plot14[0]]
labs = [l.get_label() for l in pppp]
fig.legend(pppp, labs, loc='upper center', ncol=5)
plt.show()

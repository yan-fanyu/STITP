# _*_ coding: utf-8 _*_
"""
Time:     2022/9/4 21:15
Author:   Yan Fanyu
Version:  V 0.1
File:     main.py
Describe: Github link: https://github.com/YanFanYu2001
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt			# 导入模块

N = 90000

kind_num = 3

# 1. 导入数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2、数据处理 先将原始数据格式转为np数组，然后再转为DNN网络的tensor格式
train = np.array(train)
test = np.array(test)
train_input = torch.FloatTensor(train[:, :-1])
train_label = torch.LongTensor(train[:, -1])
test_input = torch.FloatTensor(test[:, :-1])
test_label = torch.LongTensor(test[:, -1])



# 2. 定义 BP 神经网络
# 使用 四层 前向 神经网络进行训练，input dimension = 16; hidden dimension = [ 500, 100, 50, 10]; output dimension = 3
class Net(torch.nn.Module):
    def __init__(self, input, output):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input, output)
        )


    # 前向反馈
    def forward(self, x):
        x = self.layer(x)
        return x




# 3. 定义优化器损失函数
# #n_feature:输入的特征维度, n_hiddenb:神经元个数, n_output:输出的类别个数
net = Net(16, kind_num)




# 优化器选用随机梯度下降方式
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

# 对于多分类一般采用的交叉熵损失函数,
loss_func = torch.nn.CrossEntropyLoss()

# 损失函数的数组
Loss = []
start_time = time.time()
# 4. 训练数据
for it in range(N):
    train_out = net(train_input)                 # 输入input,输出out
    loss = loss_func(train_out, train_label)     # 输出与label对比
    if it%100==0:
        elapsed = time.time() - start_time
        Loss.append(loss.detach().numpy())
        print('It: %d, loss: %.4f, Time: %.2f' % (it, loss.detach().numpy(), elapsed, ) +', finsh: {:.2%}'.format(it/N))
        start_time = time.time()
    optimizer.zero_grad()   # 梯度清零
    loss.backward()         # 前馈操作s
    optimizer.step()        # 使用梯度优化器




# 打印损失数组
print('Loss = ', Loss)

# 5. 得出结果
train_out = net(test_input) #out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
prediction = torch.max(train_out, 1)[1] # 返回index  0返回原值
pred_label = prediction.data.numpy()

real_label = test_label.data.numpy()


## 6.衡量准确率

# 各类别数目
num3 = [0, 0, 0]

# 各类别正确数目
a = [0, 0, 0]

# 各类别正确率
accuracy3 = [0, 0, 0]

for i in range(0, len(real_label)):
    num3[real_label[i]] += 1
    if pred_label[i] == real_label[i]:
        a[real_label[i]] += 1

for i in range(0, kind_num):
    accuracy3[i] = a[i] / num3[i]

print(accuracy3)

# 计算准确性
accuracy = sum(pred_label == real_label) / float(real_label.size)



# 打印预测的准确性
print("准确率 = ",accuracy)

# 训练集 序号 数组
x = np.linspace(1, 60, 60)
n = np.linspace(1, len(Loss), len(Loss))

n = n*100

# 画图显示中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1、损失迭代图
plt.plot(n, Loss)
plt.title('损失迭代图')
plt.xlabel('迭代次数/100')
plt.ylabel('损失')
plt.show()

# 2、预测与实际图对比
plt.scatter(x, pred_label, c='blue')
plt.scatter(x, real_label, c='red')
plt.show()

# 3、三类预测准确性柱状图

x_data = [f"第{i}类准确性" for i in range(0, kind_num)]
y_data = accuracy3

# 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 画图，plt.bar()可以画柱状图
for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i])

# 设置图片名称
plt.title("个类分类准确性")
# 设置x轴标签名
plt.xlabel("类别")
# 设置y轴标签名
plt.ylabel("准确率")
# 显示
plt.show()
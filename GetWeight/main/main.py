# _*_ coding: utf-8 _*_
"""
Time:     2022/9/4 21:15
Author:   Yan Fanyu
Version:  V 0.1
File:     main.py
Describe: Github link: https://github.com/YanFanYu2001
"""


import pandas as pd
from matplotlib import pyplot as plt

# 读入数据集
data = pd.read_csv("data.csv")

#得到行数和列数
[m,n]=data.shape

# 原始数据序列
data = data.values.astype('float')

# 数据的标准化
for i in range(0, n):
    maxi = max(data[:, i])
    mini = min(data[:, i])
    data[:, i] = (data[:, i] - mini) / (maxi - mini)

# 写入文件
df = pd.DataFrame(data) # 组成一个csv
df.to_csv("./标准化矩阵.csv", index=False)


# 求差值序列
delta = abs((data - data[:, 0].reshape(len(data), 1))[:, 1:])


# 写入文件
df = pd.DataFrame(delta) # 组成一个csv
df.to_csv("./差值矩阵.csv", index=False)

# 差值矩阵的最大值与最小值
max_delta = delta.max()
min_delta = delta.min()

# 分辨系数ro通常为 0.5
rho = 0.5

# 求关联系数xi

Xi = (min_delta + rho * max_delta) / (delta + rho * max_delta)
gamma = Xi.mean(axis=0)


# 写入文件
df = pd.DataFrame(gamma)
df.to_csv("./关联系数.csv", index=False)

print('关联系数 = ', gamma)

# 关联系数按照降序排序
sort_y = sorted(gamma, reverse=True)
ordered_list = sorted(range(len(gamma)), key=lambda k: gamma[k], reverse=True)

# 按照关联性降序排序生成柱状图
x_data = [f"X{ordered_list[i]}" for i in range(0, len(ordered_list))]
y_data = sort_y

# 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 画柱状图
for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i])

# 设置图片名称
plt.title("各个特征与类别关联度分析")
# 设置x轴标签名
plt.xlabel("特征标签")
# 设置y轴标签名
plt.ylabel("关联度")
# 显示
plt.show()


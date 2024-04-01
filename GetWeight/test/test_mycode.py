import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 读入数据集
data = pd.read_csv("../main/data.csv")

[m,n]=data.shape #得到行数和列数

data = np.array(data)

X0 = []
X1 = []
X2 = []

for i in range(0, m):
    if data[i, -1] == 1:
        X0.append(data[i, :])
    elif data[i, -1] == 2:
        X1.append(data[i, :])
    else:
        X2.append(data[i, :])


# 原始数据序列
data = data.astype('float')





# 数据的标准化
for i in range(0, n):
    maxi = max(data[:, i])
    mini = min(data[:, i])
    data[:, i] = (data[:, i] - mini) / (maxi - mini)



# 组内方差 和 组间方差
s1 = []
s2 = []
s = []



# 每个特征 组内方差有三个
for i in range(0, n-1):
    s1.append(np.var(X0[i]) + np.var(X1[i]) + np.var(X2[i]))
    a = [np.mean(X0[i]), np.mean(X1[i]), np.mean(X2[i])]
    s2.append(np.var(a))

s = [s1, s2]
s = np.array(s)

# 方差的标准化
for i in range(0, 2):
    maxi = max(s[i, :])
    mini = min(s[i, :])
    s[i, :] = (s[i, :] - mini) / (maxi - mini)


s = s[1, :] - s[0, :]

print(s)


# 按照关联性降序排序生成柱状图

x_data = [f"X{i}" for i in range(0, 16)]
y_data = s

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

X = []


print(len(X0[15]))
print(len(X1[15]))
print(len(X2[15]))



plt.scatter(data[:, 10], data[:, -1])


plt.show()


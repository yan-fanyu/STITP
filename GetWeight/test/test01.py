#导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 解决图标题中文乱码问题
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#导入数据
data=pd.read_csv('train.csv')
# print(data)
#提取变量名 x1 -- x7

label_need=data.keys()[1:]

print(label_need)

data1=data[label_need].values


[m,n]=data1.shape #得到行数和列数

data2=data1

data3=data2

ymin=0.002
ymax=1
for j in range(0,n):
    d_max=max(data2[:,j])
    d_min=min(data2[:,j])
    data3[:,j]=(ymax-ymin)*(data2[:,j]-d_min)/(d_max-d_min)+ymin
print(data3)
# 绘制 x1,x4,x5,x6,x7 的折线图
t=range(0, 400)
plt.plot(t,data3[:,0],'*-',c='red')
for i in range(4):
    plt.plot(t,data3[:,2+i],'.-')
plt.xlabel('year')
plt.legend(['x1','x4','x5','x6','x7'])
plt.title('灰色关联分析')

# 得到其他列和参考列相等的绝对值
for i in range(3,7):
    data3[:,i]=np.abs(data3[:,i]-data3[:,0])

#得到绝对值矩阵的全局最大值和最小值
data4=data3[:,3:7]

d_max=np.max(data4)
d_min=np.min(data4)

a=0.5 #定义分辨系数

# 计算灰色关联矩阵
data4=(d_min+a*d_max)/(data4+a*d_max)
xishu=np.mean(data4, axis=0)
print(' x4,x5,x6,x7 与 x1之间的灰色关联度分别为：')
print(xishu)

plt.show()
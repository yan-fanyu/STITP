import pandas as pd
x=pd.read_csv('../main/data.csv')
x=x.iloc[:,1:].T
# 1、数据均值化处理
x_mean=x.mean(axis=1)
for i in range(x.index.size):
    x.iloc[i,:] = x.iloc[i,:]/x_mean[i]

# 2、提取参考队列和比较队列
ck=x.iloc[0,:]
cp=x.iloc[1:,:]

# 比较队列与参考队列相减
t=pd.DataFrame()
for j in range(cp.index.size):
    temp=pd.Series(cp.iloc[j,:]-ck)

#求最大差和最小差m
mmax=t.abs().max().max()
mmin=t.abs().min().min()
rho=0.5

#3、求关联系数

ksi=((mmin+rho*mmax)/(abs(t)+rho*mmax))

#4、求关联度
r=ksi.sum(axis=1)/ksi.columns.size

#5、关联度排序，得到结果r3>r2>r1
result=r.sort_values(ascending=False)
print(result)
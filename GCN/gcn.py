# _*_ coding: utf-8 _*_
"""
Time:     2022/9/4 21:15
Author:   Yan Fanyu
Version:  V 0.1
File:     main.py
Describe: Github link: https://github.com/YanFanYu2001
"""


import dgl
import dgl.nn as dglnn
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import generate_mask_tensor
from dgl.data import DGLDataset
import torch
import numpy as np

# 读取数据集
data = pd.read_csv('data.csv')
data = np.array(data)


# 34个节点，78条边 无权无向图
def build_network():
    # 起点集合
    src = data[0, :]
    # 终点集合
    dst = data[1,:]

    # 起始节点和目标节点交换是为了得到一个无向图
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    return dgl.graph((u, v))


def _sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

# 继承 DGLDataset 类
class MyDataset(DGLDataset):
    def __init__(self):
        super(MyDataset, self).__init__(name='dataset_name')

    def process(self):
        g = G
        train_mask = _sample_mask(train_set, g.number_of_nodes())#[true, true, fasle, ...,false][34] 如果在训练集里， 就是true
        val_mask = _sample_mask(validation_set, g.number_of_nodes()) #评估集
        test_mask = _sample_mask(test_set, g.number_of_nodes()) #测试集

        # 把各个集合赋值给模型
        g.ndata['train_mask'] = generate_mask_tensor(train_mask)
        g.ndata['val_mask'] = generate_mask_tensor(val_mask)
        g.ndata['test_mask'] = generate_mask_tensor(test_mask)
        g.ndata['label'] = torch.tensor(labels) # 节点的标签

        # 节点的特征个数 可以自行定义
        features = 5

        # 节点的特征
        g.ndata['feat'] = torch.randn(g.number_of_nodes(), features) # 每个节点features个特征

        print(g.ndata)

        self._num_labels = int(torch.max(labels).item() + 1)
        self._labels = labels
        self._g = g

    def __getitem__(self, idx):
        return self._g


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型"mean"代表返回平均值
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h




#计算分类正确的节点数 / 总共的节点数，即分类的准确率
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1) # 返回每个节点最可能的类型 [type1, type2, ..., type.]
        print("实际标签", end="")
        print(labels)
        print("预测标签", end="")
        print(indices)
        correct = torch.sum(indices == labels)
        real = labels
        pred = indices
        return correct.item() * 1.0 / len(labels), real, pred



if __name__ == "__main__":
    # 标签数量
    class_num = 4
    G = build_network()
    # 训练节点的索引
    train_set = np.array(range(int(G.number_of_nodes() * 0.6)))
    # 评估节点的索引
    validation_set = train_set + int(G.number_of_nodes() * 0.2)
    # 验证节点的索引
    test_set = np.array([i for i in np.array(range(G.number_of_nodes())) if (i not in train_set) and (i not in validation_set)])

    labels = torch.randint(0, class_num, (G.number_of_nodes(),))  # 标签，分为 class_num 类

    dataset = MyDataset()

    graph = dataset[0]
    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    n_features = node_features.shape[1]  # 大小为10，表示每个节点有十个特征
    n_labels = int(node_labels.max().item() + 1)  # 标签的个数，大小为2， 即所有的节点被分为几类

    # 建立模型
    model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
    # 使用Adam优化器进行优化
    opt = torch.optim.Adam(model.parameters())


    # 训练100次
    for epoch in range(50):
        model.train()
        # 使用所有节点(全图)进行前向传播计算
        logits = model(graph, node_features)
        # 计算损失值
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        # 计算验证集的准确度
        acc, real, pred = evaluate(model, graph, node_features, node_labels, valid_mask)
        # 进行反向传播计算
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("损失 = ", end="")
        print(loss.item())
        print("正确率 = ", end="")
        print(acc)
        print("-----------------------------------------------------------")

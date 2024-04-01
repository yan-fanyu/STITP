import networkx as nx
import numpy as np

import util


class FastNewman:
    def __init__(self, path):
        self.G = util.load_graph(path)
        # G = nx.read_gml('dolphins.gml')
        self.A = nx.to_numpy_array(self.G)  # 邻接矩阵
        self.num_node = len(self.A)  # 点数
        self.num_edge = sum(sum(self.A))  # 边数
        self.c = {}  # 记录所有Q值对应的社团分布

        # def merge_community(self, iter_num, detaQ, e, b):
    #     # 一起合并容易出bug  查询的结果I在遍历过程中 可能在已经前面某次作为J被合并了
    #     # 比如某次是[ 3, 11] [11, 54] 第一轮迭代中11被合并 第二轮54合并到旧的11中  会导致后面被删除 导致节点消失  需要将54合并到现在11所在位置  比较麻烦 不如一个个合并
    #     b_num = sum([len(i) for i in b])
    #     det_max = np.amax(detaQ)
    #
    #     (I, J) = np.where(detaQ == det_max)
    #     print((I, J) )
    #     # 由于先遍历的I I中可能有多个相同值  所以合并时候因应该将J合并到I中
    #     # 如果将I合并到J中 后续删除删不到
    #     for m in range(len(I)):
    #         # 确保J还未被合并
    #         if J.tolist().index(J[m]) == m:
    #             # 将第J合并到I 然后将J清零
    #             e[I[m], :] = e[J[m], :] + e[I[m], :]
    #             e[J[m], :] = 0
    #             e[:, I[m]] = e[:, J[m]] + e[:, I[m]]
    #             e[:, J[m]] = 0
    #             b[I[m]] = b[I[m]] + b[J[m]]
    #
    #     e = np.delete(e, J, axis=0)
    #     e = np.delete(e, J, axis=1)
    #     J = sorted(list(set(J)), reverse=True)
    #     for j in J:
    #         b.remove(b[j])  # 删除第J组社团，（都合并到I组中了）
    #     b_num2 = sum([len(i) for i in b])
    #     if b_num2 != b_num:
    #         print("111")
    #     self.c[iter_num] = b.copy()
    #     return e, b

    def merge_community(self, iter_num, detaQ, e, b):
        # 一个个合并
        (I, J) = np.where(detaQ == np.amax(detaQ))
        # 由于先遍历的I I中可能有多个相同值  所以合并时候因应该将J合并到I中
        # 如果将I合并到J中 后续删除删不到
        e[I[0], :] = e[J[0], :] + e[I[0], :]
        e[J[0], :] = 0
        e[:, I[0]] = e[:, J[0]] + e[:, I[0]]
        e[:, J[0]] = 0
        b[I[0]] = b[I[0]] + b[J[0]]

        e = np.delete(e, J[0], axis=0)
        e = np.delete(e, J[0], axis=1)
        b.remove(b[J[0]])  # 删除第J组社团，（都合并到I组中了）
        self.c[iter_num] = b.copy()
        return e, b

    def Run_FN(self):
        e = self.A / self.num_edge  # 社区i,j连边数量占总的边的比例
        a = np.sum(e, axis=0)  # e的列和，表示与社区i中节点相连的边占总边数的比例
        b = [[i] for i in range(self.num_node)]  # 本轮迭代的社团分布
        Q = []
        iter_num = 0
        while len(e) > 1:
            num_com = len(e)
            detaQ = -np.power(10, 9) * np.ones((self.num_node, self.num_node))  # detaQ可能为负数，初始设为负无穷
            for i in range(num_com - 1):
                for j in range(i + 1, num_com):
                    if e[i, j] != 0:
                        detaQ[i, j] = 2 * (e[i, j] - a[i] * a[j])
            if np.sum(detaQ + np.power(10, 9)) == 0:
                break

            e, b = self.merge_community(iter_num, detaQ, e, b)

            a = np.sum(e, axis=0)
            # 计算Q值
            Qt = 0.0
            for n in range(len(e)):
                Qt += e[n, n] - a[n] * a[n]
            Q.append(Qt)
            iter_num += 1
        max_Q, community = self.get_community(Q)
        return max_Q, community

    def get_community(self, Q):
        max_k = np.argmax(Q)
        community = self.c[max_k]
        return Q[max_k], community


if __name__ == '__main__':
    fast = FastNewman("test.txt")

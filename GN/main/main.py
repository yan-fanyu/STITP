# _*_ coding: utf-8 _*_
"""
Time:     2022/9/4 21:15
Author:   Yan Fanyu
Version:  V 0.1
File:     main.py
Describe: Github link: https://github.com/YanFanYu2001
"""

# 首先导入包
import networkx as nx
import matplotlib.pyplot as plt
import util

# GN 算法 类
class GN(object):
	def __init__(self, G):
		self._G_cloned = util.clone_graph(G)
		self._G = G
		self._partition = [[n for n in G.nodes()]]
		self._max_Q = 0.0

	# GN算法
	def execute(self):
		while len(self._G.edges()) > 0:
			# 1.计算所有边的edge betweenness(边介数)
			edge = max(nx.edge_betweenness_centrality(self._G).items(), key = lambda item:item[1])[0]
			# 2.移去edge betweenness(边介数)最大的边
			self._G.remove_edge(edge[0], edge[1])
			# 获得移去边后的子连通图
			components = [list(c) for c in list(nx.connected_components(self._G))]
			if len(components) != len(self._partition):
				# 3.计算 Q 值，即模块度的值
				cur_Q = util.cal_Q(components, self._G_cloned)
				if (cur_Q > self._max_Q):
					self._max_Q = cur_Q
					self._partition = components
		return self._partition

# 可视化划分结果(显示用的，不重要，可以不看)
def showCommunity(G, partition, pos):
	# 划分在同一个社区的用一个符号表示，不同社区之间的边用黑色粗体
	cluster = {}
	labels = {}
	for index,item in enumerate(partition):
		for nodeID in item:
			labels[nodeID] = str(nodeID) #设置可视化label
			cluster[nodeID] = index #节点分区号

	# 可视化节点
	colors = ['r','g','b','green','orange','olive','purple','cyan', 'g', 'r', 'b', 'r', 'b', 'r']
	shapes = ['o','o','o','o','o','o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '8', 's', 'p']
	for index,item in enumerate(partition):
		nx.draw_networkx_nodes(G, pos, nodelist = item, 
			node_color = colors[index],
			node_shape = shapes[index],
			node_size = 200,
			alpha = 1)

	# 可视化边
	edges = {len(partition):[]}
	for link in G.edges():
		# cluster间的link
		if cluster[link[0]] != cluster[link[1]]:
			edges[len(partition)].append(link)
		else:
			# cluster内的link
			if cluster[link[0]] not in edges:
				edges[cluster[link[0]]] = [link]
			else:
				edges[cluster[link[0]]].append(link)

	for index,edgelist in enumerate(edges.values()):
		# cluster内
		if index < len(partition):
			nx.draw_networkx_edges(G, pos, 
				edgelist = edgelist,
				width = 1, alpha = 0.8, edge_color = colors[index])
		else:
			# cluster间
			nx.draw_networkx_edges(G, pos, 
				edgelist = edgelist,
				width = 3, alpha = 0.8, edge_color = colors[index])

	# 可视化label
	nx.draw_networkx_labels(G, pos, labels, font_size = 12)

	plt.axis('off')
	plt.show()


if __name__ == '__main__':
	G = util.load_graph("football-x.txt")
	pos = nx.spring_layout(G)
	nx.draw(G, pos, with_labels = True, font_weight = 'bold')
	plt.show()

	# 调用 GN算法，对社区进行分类
	algo = GN(G)
	partition = algo.execute()
	print(partition)

	# 可视化结果
	showCommunity(algo._G_cloned, partition, pos)



	print(pos)


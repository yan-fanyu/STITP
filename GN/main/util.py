#coding=utf-8
import networkx as nx

# 加载网络
x1 = []
x2 = []
def load_graph(path):
	G = nx.Graph()
	with open(path) as text:
		for line in text:
			vertices = line.strip().split(" ")
			v_i = int(vertices[0])
			v_j = int(vertices[1])
			x1.append(v_i)
			x2.append(v_j)
			G.add_edge(v_i, v_j)
	#print(x1)
	#print(x2)
	return G

# 克隆
def clone_graph(G):
	cloned_graph = nx.Graph(G)
	return cloned_graph

# 计算Q值
def cal_Q(partition, G):
	m = len(list(G.edges()))   # m 代表边的个数
	a = []
	e = []

	# 详见 https://blog.csdn.net/qq_31510519/article/details/85325275

	# 计算每个社区的a值
	print(partition)
	for community in partition:
		t = 0
		for node in community:
			t += len(list(G.neighbors(node)))  # 计算节点的度
		a.append(t / float(2 * m))             #

	# 计算每个社区的e值
	for community in partition:
		t = 0
		for i in range(len(community)):
			for j in range(len(community)):
				if i != j:
					if G.has_edge(community[i], community[j]):
						t += 1
		e.append(t / float(2 * m))

	# 计算Q
	q = 0
	for ei, ai in zip(e, a):
		q += (ei - ai ** 2)  # q 等于 sum(ei - ai^2)
	return q
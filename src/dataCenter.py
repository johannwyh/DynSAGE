import sys
import os

from collections import defaultdict
import numpy as np
from copy import deepcopy

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, config, transductive=False, node_per_class=20, cheat=False):
		super(DataCenter, self).__init__()
		self.config = config
		self.transductive = transductive
		self.node_per_class = node_per_class
		self.cheat_graph = cheat

	def normalize_features(self, mx):
		"""Row-normalize sparse matrix"""
		rowsum = mx.sum(1)
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.
		r_mat_inv = np.diag(r_inv)
		mx = r_mat_inv.dot(mx)
		return mx

	def load_dataSet(self, dataSet='cora'):
		if dataSet == 'cora':
			cora_content_file = self.config['file_path.cora_content']
			cora_cite_file = self.config['file_path.cora_cite']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			label_map = {} # map label to Label_ID
			with open(cora_content_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					feat_data.append([float(x) for x in info[1:-1]])
					node_map[info[0]] = i
					if not info[-1] in label_map:
						label_map[info[-1]] = len(label_map)
					labels.append(label_map[info[-1]])
			feat_data = np.asarray(feat_data)
			#feat_data = self.normalize_features(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			if self.cheat_graph:
				num_class, num_nodes = len(np.unique(labels)), feat_data.shape[0]
				node_cluster = [[] for i in range(num_class)]
				for i in range(num_nodes):
					node_cluster[labels[i]].append(i)
				for i in range(num_class):
					for u in node_cluster[i]:
						for v in node_cluster[i]:
							adj_lists[u].add(v)
			else:
				with open(cora_cite_file) as fp:
					for i,line in enumerate(fp):
						info = line.strip().split()
						assert len(info) == 2
						paper1 = node_map[info[0]]
						paper2 = node_map[info[1]]
						adj_lists[paper1].add(paper2)
						adj_lists[paper2].add(paper1)

			assert len(feat_data) == len(labels) == len(adj_lists)
			n_nodes = feat_data.shape[0]
			if not self.transductive:
				test_indexs, val_indexs, train_indexs, all_indexs = self._split_data(feat_data.shape[0])
			else:
				test_indexs, val_indexs, train_indexs, all_indexs = self._split_data_gcn(feat_data.shape[0], labels, self.node_per_class)

		elif dataSet == 'pubmed':
			pubmed_content_file = self.config['file_path.pubmed_paper']
			pubmed_cite_file = self.config['file_path.pubmed_cites']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			with open(pubmed_content_file) as fp:
				fp.readline()
				feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
				for i, line in enumerate(fp):
					info = line.split("\t")
					node_map[info[0]] = i
					labels.append(int(info[1].split("=")[1])-1)
					tmp_list = np.zeros(len(feat_map)-2)
					for word_info in info[2:-1]:
						word_info = word_info.split("=")
						tmp_list[feat_map[word_info[0]]] = float(word_info[1])
					feat_data.append(tmp_list)
			
			feat_data = np.asarray(feat_data)
			#feat_data = self.normalize_features(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			if self.cheat_graph:
				num_class, num_nodes = len(np.unique(labels)), feat_data.shape[0]
				node_cluster = [[] for i in range(num_class)]
				for i in range(num_nodes):
					node_cluster[labels[i]].append(i)
				for i in range(num_class):
					for u in node_cluster[i]:
						for v in node_cluster[i]:
							adj_lists[u].add(v)
			else:
				with open(pubmed_cite_file) as fp:
					fp.readline()
					fp.readline()
					for line in fp:
						info = line.strip().split("\t")
						paper1 = node_map[info[1].split(":")[1]]
						paper2 = node_map[info[-1].split(":")[1]]
						adj_lists[paper1].add(paper2)
						adj_lists[paper2].add(paper1)
			
			assert len(feat_data) == len(labels) == len(adj_lists)
			n_nodes = feat_data.shape[0]
			if not self.transductive:
				test_indexs, val_indexs, train_indexs, all_indexs = self._split_data(feat_data.shape[0])
			else:
				test_indexs, val_indexs, train_indexs, all_indexs = self._split_data_gcn(feat_data.shape[0], labels, self.node_per_class)

		elif dataSet == 'cocit':
			cocit_label_file = self.config['file_path.cocit_label']
			cocit_edge_file = self.config['file_path.cocit_edge']

			label_content = []
			with open(cocit_label_file) as fp:
				lines = fp.readlines()
				for line in lines:
					line = line.strip()
					if line == "":
						continue
					node, label = list(map(int, line.split("\t")))
					label_content.append((node, label))
			label_content = sorted(label_content, key=lambda x : x[0])
			labels = [x[1] for x in label_content]
			labels = np.asarray(labels, dtype=np.int64)

			n_nodes = len(labels)
			
			feat_data = None
			
			adj_lists = defaultdict(set)
			if self.cheat_graph:
				num_class, num_nodes = len(np.unique(labels)), n_nodes
				node_cluster = [[] for i in range(num_class)]
				for i in range(num_nodes):
					node_cluster[labels[i]].append(i)
				for i in range(num_class):
					for u in node_cluster[i]:
						for v in node_cluster[i]:
							adj_lists[u].add(v)
			else:
				with open(cocit_edge_file) as fp:
					lines = fp.readlines()
					for line in lines:
						line = line.strip()
						if line == "":
							continue
						u, v = list(map(int, line.split("\t")))
						adj_lists[u].add(v)
						adj_lists[v].add(u)
			
			assert len(labels) == len(adj_lists)
			if not self.transductive:
				test_indexs, val_indexs, train_indexs, all_indexs = self._split_data(n_nodes)
			else:
				test_indexs, val_indexs, train_indexs, all_indexs = self._split_data_gcn(n_nodes, labels, self.node_per_class)
		
		setattr(self, dataSet+'_n_nodes', n_nodes)
		setattr(self, dataSet+'_test', test_indexs)
		setattr(self, dataSet+'_val', val_indexs)
		setattr(self, dataSet+'_train', train_indexs)
		setattr(self, dataSet+'_train_exp', train_indexs)
		setattr(self, dataSet+'_all', all_indexs)

		setattr(self, dataSet+'_feats', feat_data)
		setattr(self, dataSet+'_labels', labels)
		setattr(self, dataSet+'_train_labels', deepcopy(labels))
		setattr(self, dataSet+'_adj_lists', adj_lists)

	def _split_data(self, num_nodes, test_split = 3, val_split = 6):
		rand_indices = np.random.permutation(num_nodes)
		all_indexs = np.array(list(range(num_nodes)))

		test_size = num_nodes // test_split
		val_size = num_nodes // val_split
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[:test_size]
		val_indexs = rand_indices[test_size:(test_size+val_size)]
		train_indexs = rand_indices[(test_size+val_size):]
		
		return test_indexs, val_indexs, train_indexs, all_indexs

	def _split_data_gcn(self, num_nodes, labels, node_per_class=20, val_size=500, test_size=1000):
		rand_indices = np.random.permutation(num_nodes)
		all_indexs = np.array(list(range(num_nodes)))
		val_indexs = rand_indices[:val_size]
		test_indexs = rand_indices[val_size:val_size+test_size]

		remains = np.ones(np.unique(labels).shape, dtype=np.int)
		remains *= node_per_class

		training_set = []
		while np.sum(remains) > 0:
			ind = np.random.randint(0, num_nodes)
			if ind not in val_indexs and ind not in test_indexs and remains[labels[ind]] > 0:
				training_set.append(ind)
				remains[labels[ind]] -= 1
		train_indexs = np.asarray(training_set)

		return test_indexs, val_indexs, train_indexs, all_indexs

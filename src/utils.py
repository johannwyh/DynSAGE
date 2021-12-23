import sys
import os
import torch
import random
import math
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import time
import torch.nn as nn
import numpy as np

def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, max_test_acc, name, cur_epoch, args):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	labels = getattr(dataCenter, ds+'_labels')
	b_sz = args.b_sz
	models = [graphSage, classification]

	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				param.requires_grad = False
				params.append(param)

	print("Scanning Validation Set")
	"""
	batches = math.ceil(len(val_nodes) / b_sz)
	visited_nodes = set()

	val_cor = 0
	with tqdm(range(batches)) as pbar:
		for index in pbar:
			nodes_batch = val_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			labels_batch = labels[nodes_batch]
			embs_batch = graphSage(nodes_batch)
			logists = classification(embs_batch)
			_, predicts = torch.max(logists, 1)
			val_cor += np.sum(labels_batch == predicts.cpu().data.numpy()).item()

	vali_f1 = 1.0 * val_cor / len(val_nodes)
	vali_acc = vali_f1

	"""		
	embs = graphSage(val_nodes)
	logists = classification(embs)
	_, predicts = torch.max(logists, 1)
	labels_val = labels[val_nodes]
	assert len(labels_val) == len(predicts)
	comps = zip(labels_val, predicts.data)

	predicts_result = predicts.cpu().data
	vali_f1 = f1_score(labels_val, predicts_result, average="micro")
	vali_acc = 1.0 * np.sum(labels_val == predicts_result.numpy()).item() / len(labels_val)
	
	print("Validation F1: {} , Acc: {:.2f}%".format(vali_f1, 100 * vali_acc))

	model_root = "models/{}".format(name)
	
	if vali_f1 > max_vali_f1:
		max_vali_f1 = vali_f1
		print("Scanning Test Set")
		"""
		batches = math.ceil(len(test_nodes) / b_sz)
		visited_nodes = set()
		test_cor = 0
		with tqdm(range(batches)) as pbar:
			for index in pbar:
				nodes_batch = test_nodes[index*b_sz:(index+1)*b_sz]
				visited_nodes |= set(nodes_batch)
				labels_batch = labels[nodes_batch]
				embs_batch = graphSage(nodes_batch)
				logists = classification(embs_batch)
				_, predicts = torch.max(logists, 1)
				test_cor += np.sum(labels_batch == predicts.cpu().data.numpy()).item()

		test_f1 = 1.0 * test_cor / len(test_nodes)
		test_acc = test_f1
		"""
		embs = graphSage(test_nodes)
		logists = classification(embs)
		_, predicts = torch.max(logists, 1)
		labels_test = labels[test_nodes]
		assert len(labels_test) == len(predicts)
		comps = zip(labels_test, predicts.data)

		predicts_result = predicts.cpu().data
		test_f1 = f1_score(labels_test, predicts_result, average="micro")
		test_acc = 1.0 * np.sum(labels_test == predicts_result.numpy()).item() / len(labels_test)
		
		print("Test F1: {} , Acc: {:.2f}%".format(test_f1, 100 * test_acc))
		max_test_acc = test_acc

		for param in params:
			param.requires_grad = True

		if args.save_model:
			if not os.path.exists(model_root):
				os.mkdir(model_root)
			#torch.save(models, os.path.join(model_root, 'model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1)))

	for param in params:
		param.requires_grad = True

	return max_vali_f1, max_test_acc, vali_f1

def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()

def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=800):
	print('Training Classification ...')
	c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
	# train classification, detached from the current graph
	#classification.init_params()
	b_sz = 50
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')
	features = get_gnn_embeddings(graphSage, dataCenter, ds)
	for epoch in range(epochs):
		train_nodes = shuffle(train_nodes)
		batches = math.ceil(len(train_nodes) / b_sz)
		visited_nodes = set()
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			labels_batch = labels[nodes_batch]
			embs_batch = features[nodes_batch]

			logists = classification(embs_batch)
			loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss /= len(nodes_batch)
			# print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

			loss.backward()
			
			nn.utils.clip_grad_norm_(classification.parameters(), 5)
			c_optimizer.step()
			c_optimizer.zero_grad()

		max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
	return classification, max_vali_f1

def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')

	if unsup_loss == 'margin':
		num_neg = 6
	elif unsup_loss == 'normal':
		num_neg = 100
	else:
		print("unsup_loss can be only 'margin' or 'normal'.")
		sys.exit(1)

	train_nodes = shuffle(train_nodes)

	models = [graphSage, classification]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	optimizer = torch.optim.SGD(params, lr=0.7)
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	batches = math.ceil(len(train_nodes) / b_sz)

	print("Batch_Size = {}".format(b_sz))

	visited_nodes = set()
	for index in range(batches):
		nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

		# extend nodes batch for unspervised learning
		# no conflicts with supervised learning
		# nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
		
		visited_nodes |= set(nodes_batch)

		# get ground-truth for the nodes batch
		labels_batch = labels[nodes_batch]

		# feed nodes batch to the graphSAGE
		# returning the nodes embeddings
		embs_batch = graphSage(nodes_batch)

		if learn_method == 'sup':
			# superivsed learning
			logists = classification(embs_batch)
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			loss = loss_sup
		elif learn_method == 'plus_unsup':
			# superivsed learning
			logists = classification(embs_batch)
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			# unsuperivsed learning
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_sup + loss_net
		else:
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_net

		print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
		loss.backward()
		for model in models:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()

		optimizer.zero_grad()
		for model in models:
			model.zero_grad()

	return graphSage, classification

def apply_dyn_model(epoch, dataCenter, raw_adj_lists, ds, graphSage, classification, optimizer, b_sz, device, args):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	train_nodes = getattr(dataCenter, ds+'_train')
	all_nodes = getattr(dataCenter, ds+'_all')
	labels = getattr(dataCenter, ds+'_labels')

	models = [graphSage, classification]
	
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	# 1. Train Under Training Set
	print("Scanning Training Set")
	for model in models:
		for param in model.parameters():
			param.requires_grad = True
	batches = math.ceil(len(train_nodes) / b_sz)
	visited_nodes = set()
	for index in range(batches):
		nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
		visited_nodes |= set(nodes_batch)
		labels_batch = labels[nodes_batch]
		embs_batch = graphSage(nodes_batch)
		logists = classification(embs_batch)

		loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
		loss /= len(nodes_batch)
		
		print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
		loss.backward()
		for model in models:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()

		optimizer.zero_grad()
		for model in models:
			model.zero_grad()

	# 2. Calculate logists for every node
	if args.graph_revise and epoch % args.revise_freq == 0:
		for model in models:
			for param in model.parameters():
				param.requires_grad = False
		print("Forwarding All Nodes")
		batches = math.ceil(len(all_nodes) / b_sz)
		visited_nodes = set()
		for index in tqdm(range(batches)):
			nodes_batch = all_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			embs_batch = graphSage(nodes_batch)
			logists = classification(embs_batch)
			graphSage.all_logists[nodes_batch] = torch.exp(logists.detach())
		
		if args.label_noise:
			P = graphSage.all_logists[train_nodes]
			graphSage.label_confusion = torch.mm(P.permute(1, 0), P) / float(len(train_nodes))

	# 3. Graph Revise
	if args.graph_revise and epoch % args.revise_freq == 0:
		print("Revising Graph Structure Using Logists")
		num_nodes = len(all_nodes)
		all_l_t = graphSage.all_logists.permute(1, 0)
		for u in tqdm(range(num_nodes)):
			u_l = graphSage.all_logists[u:u+1, :].clone()
			i = u_l.squeeze().argmax().item()
			u_l *= -1
			u_l[0][i] *= -1
			if u == 0:
				print(u_l)
			
			dist_vec = torch.mm(u_l, all_l_t).squeeze()
			neighs = dist_vec.topk(args.revise_topk + 1)[1]
			graphSage.adj_lists[u].clear()
			for v in neighs:
				if u == v:
					continue
				graphSage.adj_lists[u].add(v)
				if len(graphSage.adj_lists[u]) == args.revise_topk:
					break
			if args.retain_raw_graph:
				if u == 0:
					print(len(raw_adj_lists[u]))
				for v in raw_adj_lists[u]:
					graphSage.adj_lists[u].add(v)
			
	return graphSage, classification

def apply_dyn_model_v2(epoch, dataCenter, raw_adj_lists, ds, graphSage, classification, optimizer, b_sz, device, args):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	train_nodes_raw = getattr(dataCenter, ds+'_train')
	if args.expand_train_set:
		train_nodes = getattr(dataCenter, ds+'_train_exp')
	else:
		train_nodes = train_nodes_raw

	all_nodes = getattr(dataCenter, ds+'_all')
	labels = getattr(dataCenter, ds+'_labels')
	train_labels = getattr(dataCenter, ds+'_train_labels')

	train_nodes = shuffle(train_nodes)
	n_nodes, n_train = len(all_nodes), len(train_nodes)

	models = [graphSage, classification]
	
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	# 1. Train Under Training Set
	print("Scanning Training Set")
	for model in models:
		for param in model.parameters():
			param.requires_grad = True
	batches = math.ceil(len(train_nodes) / b_sz)
	visited_nodes = set()

	avg_loss = 0
	with tqdm(range(batches)) as pbar:
		for index in pbar:
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			labels_batch = train_labels[nodes_batch]
			embs_batch = graphSage(nodes_batch)
			logists = classification(embs_batch)

			loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss /= len(nodes_batch)
			
			avg_loss = (avg_loss * index + loss.item()) / (index + 1)
			pbar.set_postfix(loss="{:.4f}".format(avg_loss))
			#print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
			loss.backward()
			for model in models:
				nn.utils.clip_grad_norm_(model.parameters(), 5)
			optimizer.step()

			optimizer.zero_grad()
			for model in models:
				model.zero_grad()

	# 2. Calculate logists for every node
	if args.expand_train_set and epoch >= args.warm_up and epoch % args.expand_freq == 0:
		for model in models:
			for param in model.parameters():
				param.requires_grad = False
		print("Forwarding All Nodes")
		batches = math.ceil(len(all_nodes) / b_sz)
		visited_nodes = set()
		for index in tqdm(range(batches)):
			nodes_batch = all_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			embs_batch = graphSage(nodes_batch)
			logists = classification(embs_batch)
			graphSage.all_logists[nodes_batch] = torch.exp(logists.detach())
		
		if args.label_noise:
			P = graphSage.all_logists[train_nodes]
			graphSage.label_confusion = torch.mm(P.permute(1, 0), P) / float(len(train_nodes))

	# 3. Expand Training Set
	stata = None
	if args.expand_train_set and epoch >= args.warm_up and epoch % args.expand_freq == 0:
		stata = dict()
		candidate_index = torch.Tensor(range(n_nodes)).long()
		top1_log, pseudo_label = graphSage.all_logists.max(dim=1)
		qualified_mask = (top1_log > args.thres_top1)
		qualified_mask[train_nodes] = 0
		qualified_mask[val_nodes] = 0
		qualified_mask[test_nodes] = 0
		
		qualified_index = candidate_index[qualified_mask]
		train_labels[qualified_index] = pseudo_label[qualified_index].cpu().numpy()

		exp_train_nodes = set(train_nodes_raw).union(set(qualified_index.numpy()))
		#exp_train_nodes = set(train_nodes).union(set(qualified_index.numpy()))

		added_nodes = list(exp_train_nodes.difference(set(train_nodes)))
		deleted_nodes = list(set(train_nodes).difference(exp_train_nodes))
		added, deleted = len(added_nodes), len(deleted_nodes)
		exp_train_nodes = np.array(list(exp_train_nodes))
		setattr(dataCenter, ds+'_train_exp', exp_train_nodes)

		stata["TP"] = np.sum(pseudo_label[added_nodes].cpu().numpy() == labels[added_nodes])
		stata["FP"] = added - stata["TP"]
		stata["TN"] = np.sum(pseudo_label[deleted_nodes].cpu().numpy() != labels[deleted_nodes])
		stata["FN"] = deleted - stata["TN"]
		stata["Precision"] = stata["TP"] / (stata["TP"] + stata["FP"])
		print("Added {}, Deleted {}".format(added, deleted))

	# 4. Graph Revise
	if args.expand_train_set and args.graph_revise and epoch >= args.warm_up and epoch % args.expand_freq == 0:
		print("Revising Graph Structure Using Logists")
		num_nodes = len(all_nodes)
		all_log = graphSage.all_logists.detach()
		train_log = graphSage.all_logists[exp_train_nodes].detach()

		unique_labels = np.unique(labels)
		trainset_label = train_labels[exp_train_nodes]
		label_set = {x : set(exp_train_nodes[trainset_label == x]) for x in unique_labels}
		for key in label_set.keys():
			print('{} : {}'.format(key, len(label_set[key])))

		added_cnt = 0
		correct_cnt = 0
		pseudo_label_np = pseudo_label.cpu().numpy()

		for u in tqdm(exp_train_nodes):
			dist_vec = ((train_log - all_log[u]) ** 2).sum(dim=1)
			sorted_nodes = dist_vec.topk(args.revise_topk+1, largest=False)[1][1:].cpu().numpy()
			knn_neighs = set()
			for i in sorted_nodes:
				v = exp_train_nodes[i]
				if train_labels[u] == train_labels[v]:
					knn_neighs.add(v)
			graphSage.adj_lists[u] = raw_adj_lists[u].union(knn_neighs)

			added_cnt += len(graphSage.adj_lists[u]) - len(raw_adj_lists[u])
			added_nodes = np.array(list(graphSage.adj_lists[u].difference(raw_adj_lists[u])))
			if len(added_nodes) > 0:
				correct_cnt += (pseudo_label_np[u] == labels[u]) * (pseudo_label_np[added_nodes] == labels[added_nodes]).sum()
		"""
		for u in tqdm(range(num_nodes)):

			if u in exp_train_nodes:
				# fully connected network for training nodes
				#graphSage.adj_lists[u] = raw_adj_lists[u].union(label_set[train_labels[u]])

				# KNN for non-training nodes
				dist_vec = ((all_log - all_log[u]) ** 2).sum(dim = 1)
				sorted_nodes = dist_vec.topk(args.revise_topk + 1, largest=False)[1][1:]
				knn_neighs = set(sorted_nodes.cpu().numpy())

				#dist_vec = ((train_log - all_log[u]) ** 2).sum(dim = 1)
				#sorted_nodes = dist_vec.topk(args.revise_topk + 1, largest=False)[1][1:].cpu().numpy()
				#knn_neighs = set(exp_train_nodes[sorted_nodes])
				
				graphSage.adj_lists[u] = raw_adj_lists[u].union(knn_neighs)
			else:
				# KNN for non-training nodes
				dist_vec = ((all_log - all_log[u]) ** 2).sum(dim = 1)
				sorted_nodes = dist_vec.topk(args.revise_topk + 1, largest=False)[1][1:]
				knn_neighs = set(sorted_nodes.cpu().numpy())

				#dist_vec = ((train_log - all_log[u]) ** 2).sum(dim = 1)
				#sorted_nodes = dist_vec.topk(args.revise_topk, largest=False)[1].cpu().numpy()
				#knn_neighs = set(exp_train_nodes[sorted_nodes])
				
				graphSage.adj_lists[u] = raw_adj_lists[u].union(knn_neighs)

			added_cnt += len(graphSage.adj_lists[u]) - len(raw_adj_lists[u])
			added_nodes = np.array(list(graphSage.adj_lists[u].difference(raw_adj_lists[u])))
			correct_cnt += (pseudo_label_np[u] == labels[u]) * (pseudo_label_np[added_nodes] == labels[added_nodes]).sum()
		"""
		stata['EdgeCnt'] = added_cnt
		stata['CorrPre'] = correct_cnt / added_cnt

	return graphSage, classification, stata



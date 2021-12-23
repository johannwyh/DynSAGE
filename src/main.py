import sys
import os
import torch
import argparse
import pyhocon
import random
import matplotlib.pyplot as plt
from copy import deepcopy

from src.dataCenter import *
from src.utils import *
from src.models import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=20)
parser.add_argument('--seed', type=int, default=233)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--transductive', action='store_true', 
	help="if set, use a transductive setting")
parser.add_argument('--node_per_class', type=int, default=20, 
	help="in transductive setting, use node_per_class samples for each label in training set.")
parser.add_argument('--config', type=str, default='./src/experiments.conf')
parser.add_argument('--cheat', action='store_true', help="Whether to use a cheat cluster graph")
parser.add_argument('--save-model', action='store_true', help="Set to save best validation model")
# Pretrained
parser.add_argument('--pretrained-weight', type=str, default='')

# Dynamic SAGE
parser.add_argument('--DynamicSAGE', action='store_true', help="Whether to train DynamicSAGE model.")

#v1
parser.add_argument('--graph-revise', action='store_true', help="Whether to revise the entire graph after an epoch")
parser.add_argument('--retain-raw-graph', action='store_true', help="Whether to retain original graph when revising top-k neighbors.")
parser.add_argument('--revise-topk', type=int, default=50, help="Number of edges per node to be added to G_m")
parser.add_argument('--revise-freq', type=int, default=1, help="Frequency of Revising Graph")
parser.add_argument('--label-noise', action='store_true', help="Whether to add label noise matrix to metric calculation")
parser.add_argument('--weighted-aggr', action='store_true', help="Whether to use metric to do weighted aggregation")

# v2
parser.add_argument('--expand-train-set', action='store_true')
parser.add_argument('--expand-freq', type=int, default=1)
parser.add_argument('--thres-top1', type=float, default=0.99, help='Top1 > thres, to be added to training set')
parser.add_argument('--warm-up', type=int, default=50, help='Number of epochs that remain normal GraphSAGE')

# plot
parser.add_argument('--plot', action='store_true', help="Whether to plot visualization.")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# load config file
	config = pyhocon.ConfigFactory.parse_file(args.config)

	# load data
	ds = args.dataSet
	dataCenter = DataCenter(config, transductive=args.transductive, node_per_class=args.node_per_class, cheat=args.cheat)	
	dataCenter.load_dataSet(ds)
	raw_feats = getattr(dataCenter, ds+'_feats')
	n_nodes = getattr(dataCenter, ds+'_n_nodes')
	if raw_feats is None:
		embeddings = torch.nn.Embedding(n_nodes, config['setting.hidden_emb_size']).to(device)
		index_tensor = torch.LongTensor(range(n_nodes)).to(device)
		features = embeddings(index_tensor).detach()
		embeddings, index_tensor = None, None
		torch.cuda.empty_cache()
	else:	
		features = torch.FloatTensor(raw_feats).to(device)
	num_labels = len(set(getattr(dataCenter, ds+'_labels')))

	graphSage = GraphSage(
		args, 
		config['setting.num_layers'], 
		features.size(1), 
		config['setting.hidden_emb_size'], 
		features, 
		getattr(dataCenter, ds+'_adj_lists'), 
		device, 
		gcn=args.gcn, 
		agg_func=args.agg_func,
		weighted_aggr=args.weighted_aggr,
		n_nodes=n_nodes,
		n_labels=num_labels
	)
	graphSage.to(device)
	classification = Classification(config['setting.hidden_emb_size'], num_labels)
	classification.to(device)
	
	if args.pretrained_weight != '':
		models_loaded = torch.load(args.pretrained_weight)
		sage_tilde = models_loaded[0]
		c_tilde = models_loaded[1]
		graphSage.load_from_object(sage_tilde)
		classification.load_state_dict(c_tilde.state_dict())

	unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists'), getattr(dataCenter, ds+'_train'), device)
	raw_adj_lists = deepcopy(getattr(dataCenter, ds+'_adj_lists'))

	if args.learn_method == 'sup':
		print('GraphSage with Supervised Learning')
	elif args.learn_method == 'plus_unsup':
		print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
	else:
		print('GraphSage with Net Unsupervised Learning')

	models = [graphSage, classification]
	params = []
	for model in models:
		for param in model.parameters():
			params.append(param)
	optimizer = torch.optim.SGD(params, lr=0.7)
	print("SGD Optimizer Initialized")

	max_test_acc = 0
	index, TP, FP, TN, FN, Pre = [], [], [], [], [], []
	edgecnt, corrpre = [], []
	vali = []
	for epoch in range(args.epochs):
		print('----------------------{} EPOCH {}-----------------------'.format(args.name, epoch))
		if args.DynamicSAGE:
			graphSage, classification, stata = apply_dyn_model_v2(epoch + 1, dataCenter, raw_adj_lists, ds, graphSage, classification, optimizer, args.b_sz, device, args)
		else:
			graphSage, classification = apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, args.b_sz, args.unsup_loss, device, args.learn_method)
		
		if (epoch+1) % 2 == 0 and args.learn_method == 'unsup':
			classification, args.max_vali_f1 = train_classification(dataCenter, graphSage, classification, ds, device, args.max_vali_f1, args.name)
		if args.learn_method != 'unsup':
			args.max_vali_f1, max_test_acc, vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, args.max_vali_f1, max_test_acc, args.name, epoch, args)
		print("Test Acc on Best Validation: {:.2f}%".format(max_test_acc * 100))

		if stata is not None:
			index.append(epoch + 1)
			TP.append(stata["TP"])
			FP.append(stata["FP"])
			TN.append(stata["TN"])
			FN.append(stata["FN"])
			Pre.append(stata["Precision"])
			vali.append(vali_f1)
			if args.graph_revise:
				edgecnt.append(stata["EdgeCnt"] // 10)
				corrpre.append(stata["CorrPre"])
	
	revise_tag = "revise" if args.graph_revise else "no-revise"
	title = "{}|freq={}|thres={}|K={}|{}".format(args.dataSet, args.expand_freq, args.thres_top1, args.revise_topk, revise_tag)
	print(title)
	
	if args.plot:
		font_dict = {'size' : 14}
		tick_font_dict = {'size' : 10}
		plt.plot(index, TP, 'bo--', label='Correct Added')
		plt.plot(index, FP, 'ro--', label='Wrong Added')
		plt.plot(index, TN, 'go--', label='Correct Deleted')
		plt.plot(index, FN, 'mo--', label='Wrong Deleted')
		plt.xlabel("Epoch", fontdict=font_dict)
		plt.ylabel("Nodes", fontdict=font_dict)
		plt.xticks(size=tick_font_dict['size'])
		plt.yticks(size=tick_font_dict['size'])
		title = "{}|freq={}|thres={}|K={}|{}".format(args.dataSet, args.expand_freq, args.thres_top1, args.revise_topk, revise_tag)
		plt.title("Added Nodes Stata", fontdict=font_dict)
		plt.legend()
		plt.plot()
		fn = "plots/final_v2/NodeStata_{}|{}.jpg".format(args.dataSet, args.name)
		plt.savefig(fn)
		
		plt.clf()
		plt.plot(index, Pre, 'bo--')
		plt.xlabel('Epoch', fontdict=font_dict)
		plt.ylabel('Precision', fontdict=font_dict)
		plt.xticks(size=tick_font_dict['size'])
		plt.yticks(size=tick_font_dict['size'])
		plt.title("Precision of Added Train Nodes", fontdict=font_dict)
		plt.plot()
		plt.savefig("plots/final_v2/NodePre_{}|{}.jpg".format(args.dataSet, args.name))
		
		plt.clf()
		plt.plot(index, vali, 'bo--')
		plt.xlabel('Epoch', fontdict=font_dict)
		plt.ylabel('Accuracy', fontdict=font_dict)
		plt.xticks(size=tick_font_dict['size'])
		plt.yticks(size=tick_font_dict['size'])
		plt.title("Validation Accuracy", fontdict=font_dict)
		plt.plot()
		plt.savefig("plots/final_v2/ValiAcc_{}|{}.jpg".format(args.dataSet, args.name))
		
		if args.graph_revise:
			plt.clf()
			plt.plot(index, edgecnt, 'bo--')
			plt.xlabel('Epoch', fontdict=font_dict)
			plt.ylabel('Total Num (/10)', fontdict=font_dict)
			plt.xticks(size=tick_font_dict['size'])
			plt.yticks(size=tick_font_dict['size'])
			plt.title("Number of added edges", fontdict=font_dict)
			plt.plot()
			plt.savefig("plots/final_v2/EdgeCnt_{}|{}.jpg".format(args.dataSet, args.name))

			plt.clf()
			plt.plot(index, corrpre, 'bo--')
			plt.xlabel('Epoch', fontdict=font_dict)
			plt.ylabel('Precision', fontdict=font_dict)
			plt.xticks(size=tick_font_dict['size'])
			plt.yticks(size=tick_font_dict['size'])
			plt.title("Precision of added edges", fontdict=font_dict)
			plt.plot()
			plt.savefig("plots/final_v2/EdgePre_{}|{}.jpg".format(args.dataSet, args.name))
			
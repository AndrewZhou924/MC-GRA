import argparse
import os
import random
import warnings
from copy import deepcopy

import joblib
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from dataset import Dataset
from models.gat import GAT, embedding_gat
from models.gcn import GCN, embedding_GCN  # gcn_hetero
from models.graphsage import embedding_graphsage, graphsage
from sklearn.metrics import auc, average_precision_score, roc_curve
from topology_attack import PGDAttack
from torchmetrics import AUROC
from utils import *

warnings.filterwarnings('ignore')

def test(adj, features, labels, idx_test, victim_model):
    origin_adj = adj
    adj, features, labels = to_tensor(adj, features, labels, device=device)

    victim_model.eval()
    adj_norm = normalize_adj_tensor(adj)
    output = victim_model(features, adj_norm)[0]
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test

def dot_product_decode(Z):
    if args.dataset in ['polblogs', 'brazil', 'usair']:
        Z = F.normalize(Z, p=2, dim=1)
    Z = torch.matmul(Z, Z.t())
    adj = torch.relu(Z-torch.eye(Z.shape[0]))
    return adj

def preprocess_Adj(adj, feature_adj):
    n=len(adj)
    cnt=0
    adj=adj.numpy()
    feature_adj=feature_adj.numpy()
    for i in range(n):
        for j in range(n):
            if feature_adj[i][j]>0.14 and adj[i][j]==0.0:
                adj[i][j]=1.0
                cnt+=1
    print(cnt)
    return torch.FloatTensor(adj)

def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

def metric(ori_adj, inference_adj, idx):

    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    AUC_adj = auc(fpr, tpr)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    print("Inference attack AUC: %f AP: %f" % (AUC_adj, average_precision_score(real_edge, pred_edge)))
    
    return AUC_adj

def homo_hetero_edge_extractor(adj, y):
    homo_adj = np.zeros_like(adj)
    hetero_adj = np.zeros_like(adj)

    for row in range(adj.shape[0]):
        tmp = np.nonzero(adj[row])
        if len(tmp) == 0: continue
        cols = tmp[0]
        for col in cols:
            if y[row] == y[col]:
                homo_adj[row][col] = 1
            else:
                hetero_adj[row][col] = 1

    return homo_adj, hetero_adj

def adj_auc(ori_adj, inference_adj, y):

    def auc_ap_calc(gt_adj, pred_adj):
        real_edge = gt_adj.reshape(-1)
        pred_edge = pred_adj.reshape(-1)
        fpr, tpr, _ = roc_curve(real_edge, pred_edge)
        return auc(fpr, tpr), average_precision_score(real_edge, pred_edge)
    
    homo, hetero = homo_hetero_edge_extractor(ori_adj, y)
    homo_auc, homo_ap = auc_ap_calc(homo, inference_adj)
    hetero_auc, hetero_ap = auc_ap_calc(hetero, inference_adj)    

    return homo_auc, homo_ap, hetero_auc, hetero_ap


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to optimize in GraphMI attack.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'usair', 'brazil', 'enzyme', 'ogb_arxiv'], help='dataset')
parser.add_argument('--density', type=float, default=1.0, help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=0.1)

parser.add_argument('--arch', type=str, default='gcn')
parser.add_argument('--nlayer', type=int, default=2)
parser.add_argument('--MI_type', type=str, default='KDE')

parser.add_argument('--layer_MI', nargs='+',
                    help='the layer MI constrain', required=True)

parser.add_argument('--layer_inter_MI', nargs='+',
                    help='the inter-layer MI constrain', required=True)

parser.add_argument('--aug_pe', type=float, default='proability of augmentation')

parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

setup_seed(args.seed)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='dataset', name=args.dataset, setting='GCN')
adj, features, labels, init_adj = data.adj, data.features, data.labels, data.init_adj

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
#choose the target nodes
idx_attack = np.arange(adj.shape[0]) # np.array(random.sample(range(adj.shape[0]), int(adj.shape[0]*args.nlabel))) 
num_edges = int(0.5 * args.density * adj.sum()/adj.shape[0]**2 * len(idx_attack)**2)
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, onehot_feature=False)
feature_adj = dot_product_decode(features)
init_adj = torch.FloatTensor(init_adj.todense())

if args.arch == 'gcn':
    base_model = GCN(
        nfeat=features.shape[1], 
        nclass=labels.max().item() + 1, 
        nhid=16, 
        nlayer=args.nlayer,
        dropout=0.5, 
        weight_decay=5e-4, 
        device=device,
    )
elif args.arch == 'sage':
    base_model = graphsage(
        nfeat=features.shape[1], 
        nclass=labels.max().item() + 1, 
        nhid=16, 
        nlayer=args.nlayer,
        dropout=0.5, 
        weight_decay=5e-4, 
        device=device
    )
elif args.arch == 'gat':
    base_model = GAT(
        nfeat=features.shape[1], 
        nhid=16, 
        nclass=labels.max().item() + 1, 
        nheads=4, 
        dropout=0.5, 
        alpha=0.1, 
        nlayer=args.nlayer, 
        device=args.device,
    )
else:
    print('Unknown model arch')


def objective(param):
    # Setup Victim Model
    victim_model = deepcopy(base_model)
    victim_model = victim_model.to(device)

    logs = victim_model.fit(
        features=features, 
        adj=adj, 
        labels=labels, 
        idx_train=idx_train, 
        idx_val=idx_val, 
        idx_test=idx_test,
        beta=param, 
        verbose=False, 
        MI_type=args.MI_type, #  linear_CKA, DP, linear_HSIC, KDE
        stochastic = 1,
        aug_pe= param['aug_pe'],
        plain_acc=param['plain_acc']
    )
    # IAZ, IYZ, full_losses = logs['IAZ'], logs['IYZ'], logs['full_losses']
    
    # torch.save(victim_model.state_dict(), args.dataset+'_gcn_2.pt')

    # layer_aucs = logs['final_layer_aucs']

    # AUC = np.sum(layer_aucs)
    AUC = None 

    print('=== testing GCN on original(clean) graph ===')
    ACC = test(adj, features, labels, idx_test, victim_model)

    return ACC, AUC, victim_model # , IAZ, IYZ, full_losses

def GraphMI(victim_model):
    if args.arch == 'gcn':
        embedding = embedding_GCN(nfeat=features.shape[1], nhid=16, device=device)
    elif args.arch == 'sage':
        embedding = embedding_graphsage(nfeat=features.shape[1], nhid=16, device=device)
    elif args.arch == 'gat':
        embedding = embedding_gat(
            nfeat=features.shape[1], 
            nhid=16, 
            dropout=0.5, 
            alpha=0.1, 
            nheads=4, 
            nlayer=args.nlayer, 
            device=device
        )
    else:
        print('Unknown arch')
        return 

    if args.arch == 'gat':
        embedding.set_layers(1)

    embedding.load_state_dict(transfer_state_dict(victim_model.state_dict(), embedding.state_dict()))

    # Setup Attack Model
    Y_A = victim_model(features.to(device), adj.to(device))[0]
    model = PGDAttack(model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type='CE', device=device)

    model = model.to(device)

    model.attack(features, init_adj, labels, idx_attack, num_edges, epochs=args.epochs)
    inference_adj = model.modified_adj.cpu()
    print('=== calculating link inference AUC&AP ===')
    attack_AUC = metric(adj.numpy(), inference_adj.numpy(), idx_attack)

    homo_auc, homo_ap, hetero_auc, hetero_ap = adj_auc(
        adj.numpy(), 
        inference_adj.numpy(), 
        labels
    )
    print('Homo AUC:{:.3f}, Homo AP:{:.3f}, Hetero AUC:{:.3f}, Hetero AP:{:.3f}'.format(
            homo_auc, homo_ap, hetero_auc, hetero_ap
        )
    )
    if args.arch != 'gat':
        embedding.gc = deepcopy(victim_model.gc)

    embedding.set_layers(args.nlayer)
    H_A2 = embedding(features.to(device), adj.to(device))

    H_A2 = dot_product_decode(H_A2.detach().cpu())
    Y_A2 = dot_product_decode(Y_A.detach().cpu())

    idx=np.arange(adj.shape[0])
    auc=metric(adj.numpy(), H_A2.numpy(), idx)
    print("last_gc adj=", round(auc, 3))
    auc=metric(adj.numpy(), Y_A2.numpy(), idx)
    print("out adj=", round(auc, 3))
    return attack_AUC, inference_adj.numpy()

plain_acc_maps = {
    'cora': 0.757,
    'citeseer': 0.6303,
    'polblogs': 0.8386,
    'usair': 0.4703,
    'brazil': 0.7308,
    'AIDS': 0.6682,
    'enzyme': 0.6461,
    'ogb_arxiv':0.376,
}

param = {}

param['plain_acc'] = plain_acc_maps[args.dataset]
param['aug_pe'] = args.aug_pe
layer_MI_params = list(map(float, args.layer_MI))
layer_inter_params = list(map(float, args.layer_inter_MI))
for i in args.nlayer:
    param['layer-{}'.format(i)] = layer_MI_params[i]
    if (i+1) <= len(args.nlayer)-1:
        param['layer_inter-{}'.format(i)] = layer_inter_params[i]

if args.dataset == 'ogb_arxiv':
    # no mission
    param['aug_pe'] = 0.9

    param['layer-0'] = 0.2
    param['layer-1'] = 0.3
    param['layer-2'] = 0.06

    param['layer_inter-0'] = 0.01
    param['layer_inter-1'] = 0.01

# AIDS
if args.dataset == 'AIDS':
    param['aug_pe'] = 0.07

    param['layer-0'] = 2.4
    param['layer-1'] = 3.9
    param['layer-2'] = 1.3

    param['layer_inter-0'] = 1.3
    param['layer_inter-1'] = 1.3

    # hetero CKA
    # param['aug_pe'] = 0.4

    # param['layer-0'] = 0
    # param['layer-1'] = 9.9
    # param['layer-2'] = 0

    # param['layer_inter-0'] = 3
    # param['layer_inter-1'] = 3


if args.dataset == 'citeseer':
    param['aug_pe'] = 0.702

    param['layer-0'] = .09
    param['layer-1'] = .006
    param['layer-2'] = .01

    param['layer_inter-0'] = 5e-10
    param['layer_inter-1'] = 1e-10


    # hetero
    # param['aug_pe'] = 0.702

    # param['layer-0'] = 0
    # param['layer-1'] = 5
    # param['layer-2'] = 0

    # param['layer_inter-0'] = 1
    # param['layer_inter-1'] = 1

if args.dataset == 'cora':
    param['aug_pe'] = 0.17

    param['layer-0'] = 3.2
    param['layer-1'] = 0.77
    param['layer-2'] = 0.02

    param['layer_inter-0'] = 0.27
    param['layer_inter-1'] = 0.96

    # hetero 
    # param['aug_pe'] = 0.2

    # param['layer-0'] = 3.2
    # param['layer-1'] = 100
    # param['layer-2'] = 0.02

    # param['layer_inter-0'] = 0.27
    # param['layer_inter-1'] = 0.96

if args.dataset == 'brazil':
    param['aug_pe'] = 0.5

    param['layer-0'] = 1.9
    param['layer-1'] = 2.5
    param['layer-2'] = 1

    param['layer_inter-0'] = 1.2
    param['layer_inter-1'] = 1.2

    # hetero
    # param['aug_pe'] = 0.9

    # param['layer-0'] = 0
    # param['layer-1'] = 20
    # param['layer-2'] = 0

    # param['layer_inter-0'] = 1
    # param['layer_inter-1'] = 1

if args.dataset == 'usair':
    # DP 
    param['aug_pe'] = 0.89

    param['layer-0'] = 6.6
    param['layer-1'] = 1.0
    param['layer-2'] = 0.5

    param['layer_inter-0'] = 1.3
    param['layer_inter-1'] = 3.8
    
    # hetero
    # linear HSIC
    # param['aug_pe'] = 0.89

    # param['layer-0'] = 0
    # param['layer-1'] = 10.0
    # param['layer-2'] = 0

    # param['layer_inter-0'] = 1.3
    # param['layer_inter-1'] = 3.8


if args.dataset == 'polblogs':
    param['aug_pe'] = 0.3

    param['layer-0'] = 3
    param['layer-1'] = 2
    param['layer-2'] = 2

    param['layer_inter-0'] = 1
    param['layer_inter-1'] = 1

    # gcn
    # 4
    # param = {'plain_acc': 0.839, 'aug_pe': 0.04, 'layer-0': 4.9, 'layer-1': 2.7, 'layer-2': 2.7, 'layer-3': 2.3000000000000003, 'layer-4': 2.0, 'layer_inter-0': 1.3, 'layer_inter-1': 0.0, 'layer_inter-2': 0.2, 'layer_inter-3': 0.0}
    # 6
    # param = {
    #     'plain_acc': 0.839, 
    #     'aug_pe': 0.5, 
    #     'layer-0': 200, 
    #     'layer-1': 10,
    #     'layer-2': 1.3, 
    #     'layer-3': 2.3, 
    #     'layer-4': 2.5, 
    #     'layer-5': 2.7, 
    #     'layer-6': 0.9, 
    #     'layer_inter-0': 2.0,
    #     'layer_inter-1': 0.4, 
    #     'layer_inter-2': 2.0, 
    #     'layer_inter-3': 0.8, 
    #     'layer_inter-4': 0.4, 
    #     'layer_inter-5': 0.2
    # }

    # GAT
    # 2 
    # param = {
    #     'plain_acc': 0.839, 
    #     'aug_pe': 0.38, 
    #     'layer-0': 4.3, 
    #     'layer-1': 1.8, 
    #     'layer-2': 0.8, 
    #     'layer_inter-0': 1.5, 
    #     'layer_inter-1': 1.5
    # }

    # 4
    # param = {'plain_acc': 0.839, 'aug_pe': 0.02, 'layer-0': 3.6, 'layer-1': 0.8, 'layer-2': 1.9000000000000001, 'layer-3': 2.3000000000000003, 'layer-4': 0.0, 'layer_inter-0': 1e-10, 'layer_inter-1': 1e-10, 'layer_inter-2': 1e-10, 'layer_inter-3': 1e-10}

    # 6
    # python reproduce_optuna.py --arch=gat --nlayer=6 --dataset=polblogs

    # SAGE 
    # 2 
    # param = {'plain_acc': 0.863, 'aug_pe': 0.3, 'layer-0': 0.2, 'layer-1': 0.1, 'layer-2': 0.6, 'layer_inter-0': 0.9, 'layer_inter-1': 0.2}

    # 4
    # param = {'plain_acc': 0.899, 'aug_pe': 0.3, 'layer-0': 5.0, 'layer-1': 2, 'layer-2': 1.5, 'layer-3': 0.2, 'layer-4': 1.0, 'layer_inter-0': 1e-10, 'layer_inter-1': 1e-10, 'layer_inter-2': 1e-10, 'layer_inter-3': 1e-10}
    # param = {'plain_acc': 0.899, 'aug_pe': 0.14, 'layer-0': 4.9, 'layer-1': 0.4, 'layer-2': 1.5, 'layer-3': 0.2, 'layer-4': 1.0, 'layer_inter-0': 1e-10, 'layer_inter-1': 1e-10, 'layer_inter-2': 1e-10, 'layer_inter-3': 1e-10}

    # 6
    # param = {'plain_acc': 0.839, 'aug_pe': 0.3, 'layer-0': 0.0, 'layer-1': 2.0, 'layer-2': 0.8, 'layer-3': 0.5, 'layer-4': 0.5, 'layer-5': 0.6000000000000001, 'layer-6': 0.0, 'layer_inter-0': 1e-10, 'layer_inter-1': 1e-10, 'layer_inter-2': 1e-10, 'layer_inter-3': 1e-10, 'layer_inter-4': 1e-10, 'layer_inter-5': 1e-10}


if args.dataset == 'enzyme':
    # no mission 
    param['aug_pe'] = 0.5

    param['layer-0'] = 20
    param['layer-1'] = 8 
    param['layer-2'] = 8

    param['layer_inter-0'] = 6
    param['layer_inter-1'] = 6

# param = None

ACC, AUC, victim_model = objective(param) # , IAZ, IYZ, full_losses
# loss_IYZ, loss_IAZ, loss_inter, loss_mission = full_losses

attack_AUC, inference_adj = GraphMI(victim_model)
print(param)
print('Acc: {}; AUC: {}'.format(round(ACC.item(), 3), round(attack_AUC.item(), 3)))

path = os.path.join('results/', args.dataset)
os.makedirs(path, exist_ok=True)

torch.save(
    victim_model.state_dict(),
    os.path.join(path, f'{args.dataset}.pt')
)

# torch.save(
#     IAZ, 
#     os.path.join(path, 'IAZ.pt')
# )

# torch.save(
#     IYZ, 
#     os.path.join(path, 'IYZ.pt')
# )

# torch.save(
#     loss_IYZ, 
#     os.path.join(path, 'loss_IYZ.pt')
# )

# torch.save(
#     loss_IAZ, 
#     os.path.join(path, 'loss_IAZ.pt')
# )

# torch.save(
#     loss_inter, 
#     os.path.join(path, 'loss_inter.pt')
# )

# torch.save(
#     loss_mission, 
#     os.path.join(path, 'loss_mission.pt')
# )

# torch.save(
#     inference_adj,
#     os.path.join(path, 'inference_adj.pt')
# )

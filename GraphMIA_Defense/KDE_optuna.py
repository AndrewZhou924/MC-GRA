# import argparse
# import random
# import warnings
# from copy import deepcopy

# import joblib
# import numpy as np
# import optuna
# import torch
# import torch.nn.functional as F
# from sklearn.metrics import auc, average_precision_score, roc_curve
# from torchmetrics import AUROC

# from dataset import Dataset
# from models.gat import GAT, embedding_gat
# from models.gcn import GCN, embedding_GCN
# from models.graphsage import embedding_graphsage, graphsage
# from topology_attack import PGDAttack
# from utils import *

# warnings.filterwarnings('ignore')

# def test(adj, features, labels, idx_test, victim_model):
#     origin_adj = adj
#     adj, features, labels = to_tensor(adj, features, labels, device=device)

#     victim_model.eval()
#     adj_norm = normalize_adj_tensor(adj)
#     output = victim_model(features, adj_norm)[0]

#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

#     return acc_test

# def dot_product_decode(Z):
#     # Z = F.normalize(Z, p=2, dim=1)
#     Z = torch.matmul(Z, Z.t())
#     adj = torch.relu(Z-torch.eye(Z.shape[0]))
#     return adj

# def preprocess_Adj(adj, feature_adj):
#     n=len(adj)
#     cnt=0
#     adj=adj.numpy()
#     feature_adj=feature_adj.numpy()
#     for i in range(n):
#         for j in range(n):
#             if feature_adj[i][j]>0.14 and adj[i][j]==0.0:
#                 adj[i][j]=1.0
#                 cnt+=1
#     print(cnt)
#     return torch.FloatTensor(adj)

# def transfer_state_dict(pretrained_dict, model_dict):
#     state_dict = {}
#     for k, v in pretrained_dict.items():
#         if k in model_dict.keys():
#             state_dict[k] = v
#         else:
#             print("Missing key(s) in state_dict :{}".format(k))
#     return state_dict

# def metric(ori_adj, inference_adj, idx):
#     real_edge = ori_adj[idx, :][:, idx].reshape(-1)
#     pred_edge = inference_adj[idx, :][:, idx].reshape(-1)

#     fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
#     AUC_adj = auc(fpr, tpr)
#     index = np.where(real_edge == 0)[0]
#     index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
#     real_edge = np.delete(real_edge, index_delete)
#     pred_edge = np.delete(pred_edge, index_delete)
#     print("Inference attack AUC: %f AP: %f" % (AUC_adj, average_precision_score(real_edge, pred_edge)))
    
#     return AUC_adj

# parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=15, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=100,
#                     help='Number of epochs to optimize in GraphMI attack.')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=16,
#                     help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')
# parser.add_argument('--dataset', type=str, default='cora',
#                     choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'usair', 'brazil', 'enzyme'], help='dataset')
# parser.add_argument('--density', type=float, default=1.0, help='Edge density estimation')
# parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
# parser.add_argument('--nlabel', type=float, default=0.1)

# parser.add_argument('--arch', type=str, default='gcn')
# parser.add_argument('--nlayer', type=int, default=2)

# parser.add_argument('--MI_type', type=str, default='KDE')
# parser.add_argument('--device', type=str, default='cuda:0')

# args = parser.parse_args()

# setup_seed(args.seed)

# device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# if device != 'cpu':
#     torch.cuda.manual_seed(args.seed)

# data = Dataset(root='dataset', name=args.dataset, setting='GCN')
# adj, features, labels, init_adj = data.adj, data.features, data.labels, data.init_adj

# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# #choose the target nodes
# idx_attack = np.arange(adj.shape[0]) # np.array(random.sample(range(adj.shape[0]), int(adj.shape[0]*args.nlabel))) 
# num_edges = int(0.5 * args.density * adj.sum()/adj.shape[0]**2 * len(idx_attack)**2)
# adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, onehot_feature=False)
# feature_adj = dot_product_decode(features)
# init_adj = torch.FloatTensor(init_adj.todense())

# if args.arch == 'gcn':
#     base_model = GCN(
#         nfeat=features.shape[1], 
#         nclass=labels.max().item() + 1, 
#         nhid=16, 
#         nlayer=args.nlayer,
#         dropout=0.5, 
#         weight_decay=5e-4, 
#         device=device
#     )
# elif args.arch == 'sage':
#     base_model = graphsage(
#         nfeat=features.shape[1], 
#         nclass=labels.max().item() + 1, 
#         nhid=16, 
#         nlayer=args.nlayer,
#         dropout=0.5, 
#         weight_decay=5e-4, 
#         device=device
#     )
# elif args.arch == 'gat':
#     base_model = GAT(
#         nfeat=features.shape[1], 
#         nhid=16, 
#         nclass=labels.max().item() + 1, 
#         nheads=4, 
#         dropout=0.5, 
#         alpha=0.1, 
#         nlayer=args.nlayer, 
#         device=args.device,
#     )
# else:
#     print('Unknown model arch')


# def objective(trial):
    
#     param = {}
#     param['aug_pe'] = trial.suggest_float('aug_pe', low=0, high=0.6, step=0.01)
#     # brazil 0.2

#     # hsic dp cka gcn 

#     # brazil dp 
#     # CKA
#     # AIDS (4, 15) (0,15)
#     # usair (2, 7) (0,7)
#     if args.MI_type != 'KDE':
#         for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
#             param_name = 'layer-{}'.format(layer_id)
#             if layer_id == 0:
#                 param[param_name] = trial.suggest_float(param_name, low=8, high=15, step=0.01)
#             else:
#                 param[param_name] = trial.suggest_float(param_name, low=0, high=5, step=0.01)
    
#         for layer_id in range(args.nlayer):
#             param_inter = 'layer_inter-{}'.format(layer_id)
#             param[param_inter] = trial.suggest_float(param_name, low=0, high=15, step=0.1)

#     elif args.MI_type == 'KDE':
#         if args.dataset == 'cora':
#             for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
#                 param_name = 'layer-{}'.format(layer_id)
#                 if layer_id == 0:
#                     param[param_name] = trial.suggest_int(param_name, low=5e3, high=5e5, step=1e2)
#                 else:
#                     param[param_name] = trial.suggest_int(param_name, low=0, high=5e4, step=1e2)

#             for layer_id in range(args.nlayer):
#                 param_inter = 'layer_inter-{}'.format(layer_id)
#                 param[param_inter] = trial.suggest_float(param_inter, low=0, high=4, step=0.01)

#         elif args.dataset == 'citeseer':
#             for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
#                 param_name = 'layer-{}'.format(layer_id)
#                 if layer_id == 0:
#                     param[param_name] = trial.suggest_int(param_name, low=5e3, high=5e5, step=1e2)
#                 else:
#                     param[param_name] = trial.suggest_int(param_name, low=0, high=1e3, step=10)

#             for layer_id in range(args.nlayer):
#                 param_inter = 'layer_inter-{}'.format(layer_id)
#                 param[param_inter] = trial.suggest_float(param_inter, low=0, high=2, step=0.01)
#     # KDE 
#     # gcn
#     # citeseer
#     # for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
#     #     param_name = 'layer-{}'.format(layer_id)
#     #     param[param_name] = trial.suggest_int(param_name, low=0, high=1e4, step=1e2)
        
#     # usair MI (1e3, 1e7, 1e4) inter(0,6,0.01)
#     # AIDS MI (1e3, 1e5, 1e4) inter(0,6,0.01) (还能加大范围)
#     # ===
#     # brazil MI (0, 1e4, 1e2) inter(0,4,0.01)
#     # polblogs MI (0, 1e4, 1e2) inter(0,4,0.01)

#     # cora MI MI (1e, 5e3, 1e2) inter(0,3,0.01)
#     # citeseer MI (1e, 5e3, 1e2) inter(0,3,0.01)


#     # for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
#     #     param_name = 'layer-{}'.format(layer_id)
#     #     param[param_name] = trial.suggest_int(param_name, low=0, high=1e5, step=1e2)
    
#     # for layer_id in range(args.nlayer):
#     #     param_inter = 'layer_inter-{}'.format(layer_id)
#     #     param[param_inter] = trial.suggest_float(param_inter, low=0, high=3, step=0.01)

#     # Setup Victim Model
#     victim_model = deepcopy(base_model)
#     victim_model = victim_model.to(device)

#     layer_aucs = victim_model.fit(features, adj, labels, idx_train, idx_val, 
#         beta=param, 
#         verbose=False, 
#         MI_type=args.MI_type, #  linear_CKA, DP, linear_HSIC, KDE
#         stochastic=1,
#         aug_pe=param['aug_pe'],
#     )
#     AUC = np.sum(layer_aucs)

#     print('=== testing GCN on original(clean) graph ===')
#     ACC = test(adj, features, labels, idx_test, victim_model)

#     return ACC, AUC

# # sb: single beta
# # mb: multiple beta

# print('===== Begin Optuna =====')

# study = optuna.create_study(
#     study_name=args.dataset+ '_' + str(args.arch) + '_' + str(args.nlayer) + '_mb', 
#     storage='sqlite:///{}.db'.format(args.dataset+ '_' + str(args.arch) + '_' + str(args.nlayer)+ '_mb'),
#     directions=["maximize", 'minimize'],
#     sampler=optuna.samplers.TPESampler(seed=args.seed),
#     load_if_exists=True,
#     # pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
# )

# study.optimize(objective, n_trials=400, n_jobs=4) #  timeout=600

# print('===== Finish Optuna =====')

# print('===== Begin Saving =====')
# joblib.dump(study, args.dataset+ '_' + str(args.arch) + '_' + str(args.nlayer)  +"_CKA_study.pkl")
# print('===== End Saving =====')


import argparse
import random
import warnings
from copy import deepcopy

import joblib
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, average_precision_score, roc_curve
from torchmetrics import AUROC

from dataset import Dataset
from models.gat import GAT, embedding_gat
from models.gcn import GCN, embedding_GCN
from models.graphsage import embedding_graphsage, graphsage
from topology_attack import PGDAttack
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
    # Z = F.normalize(Z, p=2, dim=1)
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
        device=device
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
    
if args.nlayer == 2:
    if args.arch=='gcn':
        plain_acc_maps = {
            'cora': 0.757,
            'citeseer': 0.6303,
            'polblogs': 0.8386,
            'usair': 0.4703,
            'brazil': 0.7308,
            'AIDS': 0.6682,
            'enzyme': 0.6461,
            'ogb_arxiv': 0.376,
        }
    elif args.arch == 'sage':
        plain_acc_maps = {'polblogs': 0.863,}
    elif args.arch == 'gat':
        plain_acc_maps = {'polblogs': 0.906,}

elif args.nlayer == 4:
    if args.arch == 'gcn':
        plain_acc_maps = {'polblogs': 0.888,}
    elif args.arch == 'sage':
        plain_acc_maps = {'polblogs': 0.899,}
    elif args.arch == 'gat':
        plain_acc_maps = {'polblogs': 0.901,}

elif args.nlayer == 6:
    if args.arch == 'gcn':
        plain_acc_maps = {'polblogs': 0.513,}
    elif args.arch == 'sage':
        plain_acc_maps = {'polblogs': 0.804,}
    elif args.arch == 'gat':
        plain_acc_maps = {'polblogs': 0.908,}

def objective(trial):
    param = {}
    param['plain_acc'] = plain_acc_maps[args.dataset]
        
    # KDE
    ## usair
    ## aug: 0.6, 0.9, 0.01
    ## MI: (0) 5,8, 0.01; (1,2) 0,3 0.01
    ## inter 0,3 0.01

    if args.dataset=='citeseer':
        param['aug_pe'] = trial.suggest_float('aug_pe', low=0.6, high=0.8, step=0.01)
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            if layer_id == 0:
                param[param_name] = trial.suggest_float(param_name, low=0, high=1.2, step=0.01)
            else:
                param[param_name] = trial.suggest_float(param_name, low=0.001, high=0.01, step=0.001)

        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = 1e-10# trial.suggest_float(param_inter, low=0, high=1, step=0.01)
    
    if args.dataset=='enzyme':
        param['aug_pe'] = trial.suggest_float('aug_pe', low=0.5, high=0.7, step=0.01)
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            if layer_id == 0:
                param[param_name] = trial.suggest_float(param_name, low=19, high=25, step=0.2)
            else:
                param[param_name] = trial.suggest_float(param_name, low=6, high=12, step=0.1)

        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = trial.suggest_float(param_inter, low=4, high=8, step=0.1)

    if args.dataset=='ogb_arxiv':
        param['aug_pe'] = trial.suggest_float('aug_pe', low=0.7, high=0.9, step=0.02)
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            if layer_id == 0:
                param[param_name] = trial.suggest_float(param_name, low=0, high=0.02, step=0.001)
            else:
                param[param_name] = trial.suggest_float(param_name, low=0, high=0.01, step=0.001)

        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = trial.suggest_float(param_inter, low=0, high=0.02, step=0.001)

    if args.dataset=='polblogs':
        if args.arch == 'gcn':
            aug_max = 0.5
            MI_0 = 5
            MI_rest = 3
            inter = 2
        elif args.arch == 'sage':
            aug_max = 0.5
            MI_0 = 5
            MI_rest = 2
            inter = 1.5
        elif args.arch == 'gat':
            aug_max = 0.5
            MI_0 = 6
            MI_rest = 3
            inter = 2
        
        param['aug_pe'] = trial.suggest_float('aug_pe', low=0, high=aug_max, step=0.01)
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            if layer_id == 0:
                param[param_name] = trial.suggest_float(param_name, low=0, high=MI_0, step=0.1)
            else:
                param[param_name] = trial.suggest_float(param_name, low=0, high=MI_rest, step=0.1)

        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = trial.suggest_float(param_inter, low=0, high=inter, step=0.1)

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
        stochastic=1,
        aug_pe=param['aug_pe'],
        plain_acc=param['plain_acc']
    )
    # IAZ, IYZ, full_losses = logs['IAZ'], logs['IYZ'], logs['full_losses']
    
    layer_aucs = logs['final_layer_aucs']

    AUC = np.sum(layer_aucs)

    print('=== testing GCN on original(clean) graph ===')
    ACC = test(adj, features, labels, idx_test, victim_model)

    return ACC, AUC

# sb: single beta
# mb: multiple beta

print('===== Begin Optuna =====')

study_name = '_'.join([
    args.dataset,
    args.arch,
    str(args.nlayer),
    args.MI_type,
    'mb',
])
study = optuna.create_study(
    study_name=study_name,
    storage='sqlite:///{}.db'.format(study_name),
    directions=["maximize", 'minimize'],
    sampler=optuna.samplers.TPESampler(seed=args.seed),
    load_if_exists=True,
)

study.optimize(objective, n_trials=250, n_jobs=1)

print('===== Finish Optuna =====')

print('===== Begin Saving =====')
joblib.dump(study, study_name+'.pkl')
print('===== End Saving =====')

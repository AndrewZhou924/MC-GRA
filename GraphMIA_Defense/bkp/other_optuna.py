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
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'usair', 'brazil', 'enzyme'], help='dataset')
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

plain_acc_maps = {
    'cora': 0.757,
    'citeseer': 0.6303,
    'polblogs': 0.8386,
    'usair': 0.4703,
    'brazil': 0.7308,
    'AIDS': 0.6682,
}

def objective(trial):
    param = {}
    param['plain_acc'] = plain_acc_maps[args.dataset]

    if args.dataset=='cora':
        param['aug_pe'] = trial.suggest_float('aug_pe', low=0.5, high=0.8, step=0.01)
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            if layer_id == 0:
                tmp_data = trial.suggest_int(param_name, low=1, high=100, step=2)
                param[param_name] = tmp_data * 1e-10
            else:
                tmp_data = trial.suggest_int(param_name, low=1, high=1000,step=10)
                param[param_name] = tmp_data * 1e-14

        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = 1e-13# trial.suggest_float(param_inter, low=0, high=0.05, step=0.01)

    elif args.dataset=='citeseer':
        param['aug_pe'] = trial.suggest_float('aug_pe', low=0, high=0.6, step=0.01)
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            if layer_id == 0:
                param[param_name] = trial.suggest_float(param_name, low=0, high=5, step=0.01)
            else:
                param[param_name] = trial.suggest_float(param_name, low=0, high=2.5, step=0.01)

        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = trial.suggest_float(param_inter, low=0, high=2.5, step=0.01)

    elif args.dataset=='AIDS':
        param['aug_pe'] = trial.suggest_float('aug_pe', low=0, high=0.5, step=0.01)
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            if layer_id == 0:
                param[param_name] = trial.suggest_float(param_name, low=0, high=1, step=0.01)
            else:
                param[param_name] = trial.suggest_float(param_name, low=0, high=1, step=0.01)

        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = trial.suggest_float(param_inter, low=0, high=0.05, step=0.01)
    
    elif args.dataset=='usair':
        param['aug_pe'] = trial.suggest_float('aug_pe', low=0, high=0.2, step=0.01)
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            if layer_id == 0:
                param[param_name] = trial.suggest_float(param_name, low=0, high=0.05, step=0.001)
            else:
                param[param_name] = trial.suggest_float(param_name, low=0, high=0.01, step=0.001)

        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = trial.suggest_float(param_inter, low=0, high=0.01, step=0.001)

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

study.optimize(objective, n_trials=350, n_jobs=4)

print('===== Finish Optuna =====')

print('===== Begin Saving =====')
joblib.dump(study, study_name+'.pkl')
print('===== End Saving =====')

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

    # def dot_product_decode(Z,):
    #         Z = torch.matmul(Z, Z.t())
    #         adj = Z-torch.eye(Z.shape[0]).to(Z.get_device())
    #         return adj
        
    # def calculate_AUC(Z, Adj):
    #     auroc_metric = AUROC(task='binary')
    #     Z = Z.detach()
    #     Z = dot_product_decode(Z)

    #     real_edge = Adj.reshape(-1)
    #     pred_edge = Z.reshape(-1)

    #     auc_score = auroc_metric(pred_edge, real_edge)
    #     return auc_score

    # layer_aucs = []
    # for idx, node_emb in enumerate(node_embs):
    #     layer_aucs.append(calculate_AUC(node_emb, origin_adj.to(node_emb.device)).item())

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test

def dot_product_decode(Z):
    # only of usair, polblogs, brazil
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


def objective(trial):
    
    param = {}
    param['aug_pe'] = trial.suggest_float('aug_pe', low=0, high=0.6, step=0.01)

    # hsic dp cka gcn 
    # for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
    #     param_name = 'layer-{}'.format(layer_id)
    #     param[param_name] = trial.suggest_float(param_name, low=2, high=7, step=0.01)
    
    # for layer_id in range(args.nlayer):
    #     param_inter = 'layer_inter-{}'.format(layer_id)
    #     param[param_inter] = trial.suggest_float(param_name, low=0, high=7, step=0.01)

    # KDE 

    if args.arch=='gcn':
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            param[param_name] = trial.suggest_int(param_name, low=0, high=1e4, step=1e2) # low=5e6, high=5e8, step=1e5)
            
        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = trial.suggest_float(param_inter, low=0, high=5, step=0.01)

    # # gat
    # if args.arch=='gat':
    #     for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
    #         param_name = 'layer-{}'.format(layer_id)
    #         if layer_id == 0:
    #             param[param_name] = trial.suggest_int(param_name, low=0, high=1e7, step=1e4) # low=5e6, high=5e8, step=1e5)
    #         else:
    #             param[param_name] = trial.suggest_int(param_name, low=0, high=1e5, step=1e3) # 3e7

    #     for layer_id in range(args.nlayer):
    #         param_inter = 'layer_inter-{}'.format(layer_id)
    #         param[param_inter] = trial.suggest_float(param_inter, low=0, high=5, step=0.01)
    
    ## sage
    if args.arch=='sage':
        for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
            param_name = 'layer-{}'.format(layer_id)
            param[param_name] = trial.suggest_int(param_name, low=0, high=5e5, step=1e2)
 
        for layer_id in range(args.nlayer):
            param_inter = 'layer_inter-{}'.format(layer_id)
            param[param_inter] = trial.suggest_float(param_inter, low=0, high=5, step=0.01)

    # Setup Victim Model
    victim_model = deepcopy(base_model)
    victim_model = victim_model.to(device)

    layer_aucs = victim_model.fit(features, adj, labels, idx_train, idx_val, 
        beta=param, 
        verbose=False, 
        MI_type=args.MI_type, #  linear_CKA, DP, linear_HSIC, KDE
        stochastic=1,
        aug_pe=param['aug_pe'],
    )
    AUC = np.sum(layer_aucs)

    print('=== testing GCN on original(clean) graph ===')
    ACC = test(adj, features, labels, idx_test, victim_model)

    return ACC, AUC

plain_acc_maps = {
    'cora': 0.757,
    'citeseer': 0.6303,
    'polblogs': 0.8386,
    'usair': 0.4703,
    'brazil': 0.7308,
    'AIDS': 0.6682,
}

def reproduce(trial): 
    param = {}
    param['plain_acc'] = plain_acc_maps[args.dataset]

    if args.MI_type == 'KDE':
        # param['aug_pe'] = trial.suggest_float('aug_pe', low=0, high=0.6, step=0.01)
        # for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
        #     param_name = 'layer-{}'.format(layer_id)
        #     if layer_id == 0:
        #         param[param_name] = trial.suggest_float(param_name, low=3, high=6, step=0.1)
        #     else:
        #         param[param_name] = trial.suggest_float(param_name, low=0, high=5, step=0.1)

        # for layer_id in range(args.nlayer):
        #     param_inter = 'layer_inter-{}'.format(layer_id)
        #     param[param_inter] = trial.suggest_float(param_inter, low=0, high=5, step=0.1)

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
    
    else:
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

        # param['aug_pe'] = trial.suggest_float('aug_pe', low=0.5, high=0.8, step=0.01)
        # for layer_id in range(args.nlayer+1): # n gconv + 1 Linear
        #     param_name = 'layer-{}'.format(layer_id)
        #     if layer_id == 0:
        #         tmp_data = trial.suggest_int(param_name, low=1, high=100, step=2)
        #         param[param_name] = tmp_data * 1e-10
        #     else:
        #         tmp_data = trial.suggest_int(param_name, low=1, high=1000,step=10)
        #         param[param_name] = tmp_data * 1e-14

        # for layer_id in range(args.nlayer):
        #     param_inter = 'layer_inter-{}'.format(layer_id)
        #     param[param_inter] = 1e-13# trial.suggest_float(param_inter, low=0, high=0.05, step=0.01)

    # Setup Victim Model
    victim_model = deepcopy(base_model)
    victim_model = victim_model.to(device)

    victim_model.fit(
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

    print('=== testing GCN on original(clean) graph ===')
    ACC = test(adj, features, labels, idx_test, victim_model)

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
    adj_norm = normalize_adj_tensor(adj.to(device))
    Y_A = victim_model(features.to(device), adj_norm.to(device))[0]
    model = PGDAttack(model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type='CE', device=device)

    model = model.to(device)

    model.attack(features, init_adj, labels, idx_attack, num_edges, epochs=args.epochs)
    inference_adj = model.modified_adj.cpu()
    print('=== calculating link inference AUC&AP ===')
    AUC = metric(adj.numpy(), inference_adj.numpy(), idx_attack)

    if args.arch != 'gat':
        embedding.gc = deepcopy(victim_model.gc)

    embedding.set_layers(args.nlayer)
    H_A2 = embedding(features.to(device), adj.to(device))

    H_A2 = dot_product_decode(H_A2.detach().cpu())
    Y_A2 = dot_product_decode(Y_A.detach().cpu())

    layer_aucs = []
    idx=np.arange(adj.shape[0])
    auc=metric(adj.numpy(), H_A2.numpy(), idx)
    layer_aucs.append(auc)
    print("last_gc adj=", auc)
    auc=metric(adj.numpy(), Y_A2.numpy(), idx)
    layer_aucs.append(auc)
    print("out adj=", auc)

    return param, ACC, layer_aucs, AUC 

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

study = optuna.load_study(
    study_name=study_name, 
    storage='sqlite:///{}.db'.format(study_name),
)

plain_acc_maps = {
    'cora': 0.757,
    'citeseer': 0.630,
    'polblogs': 0.839,
    'usair': 0.470,
    'brazil': 0.731,
    'AIDS': 0.668,
}

with open(study_name+'_reproduce.txt', 'a') as f:
    if args.dataset == 'citeseer':
        for trail in study.trials:
            if trail.values[0] < 0.55: continue
            print('Optuna ACC: ', round(trail.values[0], 3))
            beta, acc, layer_aucs, AUC = reproduce(trail) 
            tmp_str = 'Beta: {}, Acc: {}, layer AUC: {}, graphMI AUC: {}'.format(beta, acc, layer_aucs, AUC)
            f.write(tmp_str)
            f.write('\n')
            f.flush()
    if args.MI_type != 'DP':
        for trail in study.best_trials: # best_trials trials
            # if trail.values[0] < (plain_acc_maps[args.dataset]-0.04): continue
            # print('Optuna ACC: ', round(trail.values[0], 3))
            beta, acc, layer_aucs, AUC = reproduce(trail) 
            # if acc < (plain_acc_maps[args.dataset]-0.04): continue
            tmp_str = 'Beta: {}, Acc: {}, layer AUC: {}, graphMI AUC: {}'.format(beta, acc, layer_aucs, AUC)
            f.write(tmp_str)
            f.write('\n')
            f.flush()
    if args.MI_type == 'DP':#
        for trail in study.best_trials:
            if trail.values[0] < (plain_acc_maps[args.dataset]-0.04): continue
            print('Optuna Acc: ', round(trail.values[0], 3))
            beta, acc, layer_aucs, AUC = reproduce(trail) 
            tmp_str = 'Beta: {}, Acc: {}, layer AUC: {}, graphMI AUC: {}'.format(beta, acc, layer_aucs, AUC)
            f.write(tmp_str)
            f.write('\n')
            f.flush()
f.close()

print('===== Finish Optuna =====')
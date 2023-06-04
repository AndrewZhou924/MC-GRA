import argparse
import random
import warnings
from copy import deepcopy

import joblib
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from dataset import Dataset
from models.gcn import GCN, embedding_GCN
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
    print("Test set results:", "loss= {:.4f}".format(
        loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test


def dot_product_decode(Z):
    # Z = F.normalize(Z, p=2, dim=1)
    Z = torch.matmul(Z, Z.t())
    adj = torch.relu(Z-torch.eye(Z.shape[0]))
    return adj


def preprocess_Adj(adj, feature_adj):
    n = len(adj)
    cnt = 0
    adj = adj.numpy()
    feature_adj = feature_adj.numpy()
    for i in range(n):
        for j in range(n):
            if feature_adj[i][j] > 0.14 and adj[i][j] == 0.0:
                adj[i][j] = 1.0
                cnt += 1
    print(cnt)
    return torch.FloatTensor(adj)


def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        # else:
        #     print("Missing key(s) in state_dict :{}".format(k))
    return state_dict


def metric(ori_adj, inference_adj, idx):
    # real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    # pred_edge = inference_adj[idx, :][:, idx].reshape(-1)

    # fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    # # index = np.where(real_edge == 0)[0]
    # # index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    # # real_edge = np.delete(real_edge, index_delete)
    # # pred_edge = np.delete(pred_edge, index_delete)
    # print("Inference attack AUC: %f" % (auc(fpr, tpr)))

    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    AUC_adj = auc(fpr, tpr)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(
        len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    print("Inference attack AUC: %f AP: %f" %
          (AUC_adj, average_precision_score(real_edge, pred_edge)))

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
parser.add_argument('--density', type=float, default=1.0,
                    help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD',
                    choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=0.1)

parser.add_argument('--nlayer', type=int, default=2)

parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

setup_seed(args.seed)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='dataset', name=args.dataset, setting='GCN')
adj, features, labels, init_adj = data.adj, data.features, data.labels, data.init_adj

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# choose the target nodes
# np.array(random.sample(range(adj.shape[0]), int(adj.shape[0]*args.nlabel)))
idx_attack = np.arange(adj.shape[0])
num_edges = int(0.5 * args.density * adj.sum() /
                adj.shape[0]**2 * len(idx_attack)**2)
adj, features, labels = preprocess(
    adj, features, labels, preprocess_adj=False, onehot_feature=False)
feature_adj = dot_product_decode(features)
init_adj = torch.FloatTensor(init_adj.todense())

victim_model = GCN(
    nfeat=features.shape[1],
    nclass=labels.max().item() + 1,
    nhid=16,
    nlayer=args.nlayer,
    dropout=0.5,
    weight_decay=5e-4,
    device=device
)


# Setup Victim Model
victim_model.load_state_dict(torch.load(args.dataset+'_gcn_KDE_2.pt'))
victim_model = victim_model.to(device)

embedding = embedding_GCN(nfeat=features.shape[1], nhid=16, device=device)

embedding.load_state_dict(transfer_state_dict(
    victim_model.state_dict(), embedding.state_dict()))

# Setup Attack Model
Y_A = victim_model(features.to(device), adj.to(device))[0]
model = PGDAttack(model=victim_model, embedding=embedding,
                  nnodes=adj.shape[0], loss_type='CE', device=device)

model = model.to(device)

model.attack(features, init_adj, labels, idx_attack,
             num_edges, epochs=args.epochs)
inference_adj = model.modified_adj.cpu()
print('=== testing GCN on original(clean) graph ===')
ACC = test(adj, features, labels, idx_test, victim_model)
print('=== calculating link inference AUC&AP ===')
AUC = metric(adj.numpy(), inference_adj.numpy(), idx_attack)

embedding.gc = deepcopy(victim_model.gc)

embedding.set_layers(args.nlayer)
H_A2 = embedding(features.to(device), adj.to(device))

H_A2 = dot_product_decode(H_A2.detach().cpu())
Y_A2 = dot_product_decode(Y_A.detach().cpu())

idx = np.arange(adj.shape[0])

auc = metric(adj.numpy(), H_A2.numpy(), idx)
print("last_gc adj=", auc)

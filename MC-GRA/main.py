import argparse
import os
import random
from copy import deepcopy

import baseline
import gaussian_parameterized
import gcn_parameterized
import numpy as np
import torch
import torch.nn.functional as F
from dataset import Dataset
from models.gat import GAT, embedding_gat
from models.gcn import GCN, embedding_GCN
from models.graphsage import embedding_graphsage, graphsage
from sklearn.metrics import auc, roc_curve
from topology_attack import PGDAttack
from tqdm import tqdm
from utils import *


def test(adj, features, labels, victim_model):
    adj, features, labels = to_tensor(adj, features, labels, device=device)

    victim_model.eval()
    adj_norm = normalize_adj_tensor(adj)
    output = victim_model(features, adj_norm)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:", "loss= {:.4f}".format(
        loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def dot_product_decode(Z):
    Z = torch.matmul(Z, Z.t())
    adj = torch.relu(Z-torch.eye(Z.shape[0]))
    adj = torch.sigmoid(adj)
    return adj


def dot_product_decode2(Z):
    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'AIDS':
        Z = torch.matmul(Z, Z.t())
        adj = torch.relu(Z-torch.eye(Z.shape[0]))
        adj = torch.sigmoid(adj)

    if args.dataset == 'brazil' or args.dataset == 'usair' or args.dataset == 'polblogs':
        Z = F.normalize(Z, p=2, dim=1)
        Z = torch.matmul(Z, Z.t())
        adj = torch.relu(Z-torch.eye(Z.shape[0]))
    return adj


def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
    return state_dict


def metric_pool(ori_adj, inference_adj, idx, index_delete, print_cfg=False):
    real_edge = ori_adj[idx, :][:, idx].reshape(-1).cpu()
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1).cpu()
    fpr, tpr, _ = roc_curve(real_edge, pred_edge)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    AUC = auc(fpr, tpr)
    if print_cfg:
        print(f"current auc={AUC}")
    return AUC


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
parser.add_argument('--nlayers', type=int, default=2,
                    help="number of layers in GCN.")
parser.add_argument('--arch', type=str,
                    choices=["gcn", "gat", "sage"], default='gcn')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'usair', 'brazil', 'chameleon', 'ENZYME', 'squirrel', 'ogb_arxiv'], help='dataset')
# citseer pubmed
parser.add_argument('--density', type=float,
                    default=10000000.0, help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD',
                    choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=1.0)
parser.add_argument('--iter', type=int, help="iterate times", default=1)
parser.add_argument('--max_eval', type=int,
                    help="max evaluation times for searching", default=100)
parser.add_argument('--log_name', type=str,
                    help="file name to save the result", default="result.txt")
parser.add_argument("--mode", type=str, default="evaluate", choices=["evaluate", "search", "baseline", "ensemble", "aux",
                                                                     "draw_violin", "notrain_test", "dev", "ensemble_search", "gaussian", "gcn_attack"])
parser.add_argument("--measure", type=str, default="HSIC",
                    choices=["HSIC", "MSELoss", "KL", "KDE", "CKA", "DP"])
parser.add_argument("--measure2", type=str, default="HSIC",
                    choices=["HSIC", "MSELoss", "KL", "KDE", "CKA", "DP"])
parser.add_argument("--nofeature", action='store_true')

parser.add_argument('--weight_aux', type=float, default=0,
                    help="the weight of auxiliary loss")
parser.add_argument('--weight_sup', type=float, default=1,
                    help="the weight of supervised loss")
parser.add_argument('--w1', type=float, default=0)
parser.add_argument('--w2', type=float, default=0)
parser.add_argument('--w3', type=float, default=0)
parser.add_argument('--w4', type=float, default=0)
parser.add_argument('--w5', type=float, default=0)
parser.add_argument('--w6', type=float, default=0)
parser.add_argument('--w7', type=float, default=0)
parser.add_argument('--w8', type=float, default=0)
parser.add_argument('--w9', type=float, default=0)
parser.add_argument('--w10', type=float, default=0)
parser.add_argument('--eps', type=float, default=0,
                    help="eps for adding noise")

parser.add_argument('--useH_A', action='store_true')
parser.add_argument('--useY_A', action='store_true')
parser.add_argument('--useY', action='store_true')
parser.add_argument('--ensemble', action='store_true')
parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--defense', action='store_true')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='./dataset', name=args.dataset, setting='GCN')
adj, features, labels, init_adj = data.adj, data.features, data.labels, data.init_adj


idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# choose the target nodes
idx_attack = np.array(random.sample(
    range(adj.shape[0]), int(adj.shape[0]*args.nlabel)))


num_edges = int(0.5 * args.density * adj.sum() /
                adj.shape[0]**2 * len(idx_attack)**2)

adj, features, labels = preprocess(
    adj, features, labels, preprocess_adj=False, onehot_feature=False)

feature_adj = dot_product_decode(features)
if args.nofeature:
    feature_adj = torch.eye(*feature_adj.size())

init_adj = torch.FloatTensor(init_adj.todense())


# Setup Victim Model

if args.arch == "gcn":
    victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, nlayer=args.nlayers,
                       dropout=0.5, weight_decay=5e-4, device=device)
    if args.defense:
        victim_model.load_state_dict(torch.load(
            f'./defense/{args.dataset}_{args.arch}_{args.nlayer}.pt', map_location=device))
        victim_model = victim_model.to(device)
    else:
        victim_model = victim_model.to(device)
        victim_model.fit(features, adj, labels, idx_train, idx_val)

    embedding = embedding_GCN(
        nfeat=features.shape[1], nhid=16, nlayer=args.nlayers, device=device)
    embedding.load_state_dict(transfer_state_dict(
        victim_model.state_dict(), embedding.state_dict()))

    embedding.gc = deepcopy(victim_model.gc)


if args.arch == 'sage':
    victim_model = graphsage(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, nlayer=args.nlayers,
                             dropout=0.5, weight_decay=5e-4, device=device)

    if args.defense:
        victim_model.load_state_dict(torch.load(
            f'./defense/{args.dataset}_{args.arch}_{args.nlayer}.pt'))
        victim_model = victim_model.to(device)
    else:
        victim_model = victim_model.to(device)
        victim_model.fit(features, adj, labels, idx_train, idx_val)

    embedding = embedding_graphsage(
        nfeat=features.shape[1], nhid=16, nlayer=args.nlayers, device=device)
    embedding.load_state_dict(transfer_state_dict(
        victim_model.state_dict(), embedding.state_dict()))
    # print(victim_model.state_dict().keys())
    embedding.gc = deepcopy(victim_model.gc)


if args.arch == 'gat':
    victim_model = GAT(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, nlayer=args.nlayers,
                       dropout=0.5, alpha=0.1, nheads=5, device=device)

    if args.defense:
        victim_model.load_state_dict(torch.load(
            f'./defense/{args.dataset}_{args.arch}_{args.nlayer}.pt'))
        victim_model = victim_model.to(device)
    else:
        victim_model = victim_model.to(device)
        victim_model.fit(features, adj, labels, idx_train,
                         idx_val, train_iters=200)

    embedding = embedding_gat(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, nlayer=args.nlayers,
                              dropout=0.5, alpha=0.1, nheads=5, device=device)
    embedding.load_state_dict(transfer_state_dict(
        victim_model.state_dict(), embedding.state_dict()))

    embedding.attentions = victim_model.attentions


embedding = embedding.to(device)
H_A = embedding(features.to(device), adj.to(device))
Y_A = victim_model(features.to(device), adj.to(device))

embedding.set_layers(1)
H_A1 = embedding(features.to(device), adj.to(device))
embedding.set_layers(2)
H_A2 = embedding(features.to(device), adj.to(device))


idx_attack = np.array(random.sample(
    range(adj.shape[0]), int(adj.shape[0]*args.nlabel)))

num_edges = int(0.5 * args.density * adj.sum() /
                adj.shape[0]**2 * len(idx_attack)**2)

# Setup Attack Model

model = PGDAttack(model=victim_model, embedding=embedding,
                  nnodes=adj.shape[0], loss_type='CE', device=device)

model = model.to(device)

baseline_model = baseline.PGDAttack(
    model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type="CE", device=device)
baseline_model = baseline_model.to(device)


def objective(arg):

    ori_adj = adj.numpy()
    idx = idx_attack
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    index = np.where(real_edge == 0)[0]
    index_delete = index[:int(len(real_edge)-2*np.sum(real_edge))]

    idx = idx_train
    real_edge_train = ori_adj[idx, :][:, idx].reshape(-1)
    index_train = np.where(real_edge_train == 0)[0]
    index_delete_train = index_train[:int(
        len(real_edge_train)-2*np.sum(real_edge_train))]

    idx = np.arange(adj.shape[0])
    real_edge_all = ori_adj[idx, :][:, idx].reshape(-1)
    index_all = np.where(real_edge_all == 0)[0]
    index_delete_all = index_all[:int(
        len(real_edge_all)-2*np.sum(real_edge_all))]

    lr = arg["lrexp"]
    lr = 10**lr
    weight_sup = arg["weight_sup"]
    w1 = arg["w_feature_pgd"]
    w2 = arg["w_pgd_pgdembed"]
    w3 = arg["w_pgd_eyeembed"]
    w4 = arg["w_pgdembed_feature"]
    w5 = arg["w_pgdembed_eyeembed"]
    w6 = arg["w_infoentropy_pgd"]
    w7 = arg["w_infoentropy_pgdembed"]
    w8 = arg["w_size"]
    w9 = arg["w_HA"]
    w10 = arg["w_YA"]
    weight_param = (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
    args.measure = arg["measure"]
    args.eps = arg["eps"]
    model = PGDAttack(model=victim_model, embedding=embedding, H_A=H_A2,
                      Y_A=Y_A, nnodes=adj.shape[0], loss_type='CE', device=device)
    model = model.to(device)

    model.attack(args, index_delete,
                 lr, 0, weight_sup, weight_param, feature_adj, 0, 0,
                 0, idx_train, idx_val,
                 idx_test, adj, features, init_adj, labels, idx_attack, num_edges, 0, epochs=args.epochs)

    inference_adj = model.modified_adj.cpu()
    test(adj, features, labels, victim_model)
    auc = metric_pool(adj, inference_adj, idx_attack, index_delete)
    auc_train = metric_pool(adj, inference_adj, idx_train, index_delete_train)
    auc_all = metric_pool(adj, inference_adj, np.arange(
        adj.shape[0]), index_delete_all, True)

    os.makedirs("./results/", exist_ok=True)
    with open(os.path.join("./results", args.log_name), "a") as f:
        f.write(f"current parameter: {args}\n")
        f.write(f"In attack graph: AUC={auc}\t")
        f.write(f"In train graph: AUC={auc_train}\t")
        f.write(f"In Whole Graph: AUC={auc_all}\n")
        f.write(f"current density: {inference_adj.mean()}\n")
        f.write(f"image name: {weight_param}_{weight_sup}_{lr}.png\n")
        f.write(
            "============================================================================================\n")
    return 1-auc


def test_baseline():
    ori_adj = adj.numpy()
    idx = idx_attack
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    index = np.where(real_edge == 0)[0]
    index_delete = index[:int(len(real_edge)-2*np.sum(real_edge))]

    idx = idx_train
    real_edge_train = ori_adj[idx, :][:, idx].reshape(-1)
    index_train = np.where(real_edge_train == 0)[0]
    index_delete_train = index_train[:int(
        len(real_edge_train)-2*np.sum(real_edge_train))]

    if args.mode == "evaluate":
        index_delete = np.load("./saved_data/"+args.dataset+"_idx.npy")
    elif args.mode == "search":
        np.save("./saved_data/"+args.dataset+"_idx.npy", index_delete)
    idx = np.arange(adj.shape[0])
    real_edge_all = ori_adj[idx, :][:, idx].reshape(-1)
    index_all = np.where(real_edge_all == 0)[0]
    index_delete_all = index_all[:int(
        len(real_edge_all)-2*np.sum(real_edge_all))]

    lr = args.lr
    lr = 10**lr
    weight_sup = 1
    w1 = 0
    w2 = 0
    w3 = 0
    w4 = 0
    w5 = 0
    w6 = 0
    w7 = 0
    w8 = 0
    w9 = 0
    w10 = 0
    weight_param = (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)

    baseline_model = baseline.PGDAttack(
        model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type="CE", device=device)
    baseline_model = baseline_model.to(device)
    baseline_model.attack(index_delete,
                          lr, 0, weight_sup, weight_param, feature_adj, 0, 0,
                          0, idx_train, idx_val,
                          idx_test, adj, features, init_adj, labels, idx_attack, num_edges, 0, epochs=args.epochs)
    inference_adj = baseline_model.modified_adj.cpu()
    test(adj, features, labels, victim_model)
    auc = metric_pool(adj, inference_adj, idx_attack, index_delete)
    auc_train = metric_pool(adj, inference_adj, idx_train, index_delete_train)
    auc_all = metric_pool(adj, inference_adj, np.arange(
        adj.shape[0]), index_delete_all)

    os.makedirs("./results/", exist_ok=True)
    with open(os.path.join("./results", args.log_name), "a") as f:
        f.write(f"current parameter: {args}\n")
        f.write(f"attack graph: AUC={auc}\t")
        f.write(f"train graph: AUC={auc_train}\t")
        f.write(f"Whole Graph: AUC={auc_all}\n")
        f.write(f"current density: {inference_adj.mean()}\n")
        f.write(f"image name: {weight_param}_{weight_sup}_{lr}.png\n")
        f.write(
            "============================================================================================\n")
    return inference_adj


def evaluate():
    arg = {
        "lrexp": args.lr,
        "weight_sup": args.weight_sup,
        "w_feature_pgd": args.w1,
        "w_pgd_pgdembed": args.w2,
        "w_pgd_eyeembed": args.w3,
        "w_pgdembed_feature": args.w4,
        "w_pgdembed_eyeembed": args.w5,
        "w_infoentropy_pgd": args.w6,
        "w_infoentropy_pgdembed": args.w7,
        "w_size": args.w8,
        "w_HA": args.w9,
        "w_YA": args.w10,
        "measure": args.measure,
        "eps": args.eps,
    }
    objective(arg)


def notrain_test():

    embedding.set_layers(1)
    H_A1 = embedding(features.to(device), adj.to(device))
    embedding.set_layers(2)
    H_A2 = embedding(features.to(device), adj.to(device))
    H_A1 = dot_product_decode(H_A1.detach().cpu())
    H_A2 = dot_product_decode(H_A2.detach().cpu())
    Y_A2 = dot_product_decode(Y_A.detach().cpu())
    label_adj = np.load("./saved_data/"+args.dataset+".npy")
    label_adj = torch.Tensor(label_adj)

    idx = np.arange(adj.shape[0])
    real_edge = adj[idx, :][:, idx].reshape(-1)
    index = np.where(real_edge == 0)[0]
    index_delete = index[:int(len(real_edge)-2*real_edge.sum())]

    auc = metric_pool(adj, feature_adj, idx, index_delete)
    print("feautre adj=", auc)
    auc = metric_pool(adj, H_A1, idx, index_delete)
    print("layer1 adj=", auc)
    auc = metric_pool(adj, H_A2, idx, index_delete)
    print("layer2 adj=", auc)
    auc = metric_pool(adj, Y_A2, idx, index_delete)
    print("out adj=", auc)
    auc = metric_pool(adj, label_adj, idx, index_delete)


def prepare():
    n = adj.shape[0]
    label_adj = [[0 for i in range(n)]for j in range(n)]
    for i in tqdm(range(n)):
        for j in range(n):
            if labels[i] == labels[j]:
                label_adj[i][j] = 1
            else:
                label_adj[i][j] = 0
    label_adj = torch.Tensor(label_adj).numpy()
    np.save("./saved_data/"+args.dataset+".npy", label_adj)


def eval_gaussian():
    arg = {
        "lrexp": args.lr,
        "weight_sup": args.weight_sup,
        "w_feature_pgd": args.w1,
        "w_pgd_pgdembed": args.w2,
        "w_pgd_eyeembed": args.w3,
        "w_pgdembed_feature": args.w4,
        "w_pgdembed_eyeembed": args.w5,
        "w_infoentropy_pgd": args.w6,
        "w_infoentropy_pgdembed": args.w7,
        "w_size": args.w8,
        "w_HA": args.w9,
        "w_YA": args.w10,
        "measure": args.measure,
    }
    gcn_reparameterize_attack(arg)


def eval_gcn():
    arg = {
        "lrexp": args.lr,
        "weight_sup": args.weight_sup,
        "w_feature_pgd": args.w1,
        "w_pgd_pgdembed": args.w2,
        "w_pgd_eyeembed": args.w3,
        "w_pgdembed_feature": args.w4,
        "w_pgdembed_eyeembed": args.w5,
        "w_infoentropy_pgd": args.w6,
        "w_infoentropy_pgdembed": args.w7,
        "w_size": args.w8,
        "w_HA": args.w9,
        "w_YA": args.w10,
        "measure": args.measure,
    }
    gaussian_reparameterize_attack(arg)


def gaussian_reparameterize_attack(arg):
    ori_adj = adj.numpy()
    idx = idx_attack
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    index = np.where(real_edge == 0)[0]
    index_delete = index[:int(len(real_edge)-2*np.sum(real_edge))]

    idx = idx_train
    real_edge_train = ori_adj[idx, :][:, idx].reshape(-1)
    index_train = np.where(real_edge_train == 0)[0]
    index_delete_train = index_train[:int(
        len(real_edge_train)-2*np.sum(real_edge_train))]

    idx = np.arange(adj.shape[0])
    real_edge_all = ori_adj[idx, :][:, idx].reshape(-1)
    index_all = np.where(real_edge_all == 0)[0]
    index_delete_all = index_all[:int(
        len(real_edge_all)-2*np.sum(real_edge_all))]

    lr = arg["lrexp"]
    lr = 10**lr
    weight_sup = arg["weight_sup"]
    w1 = arg["w_feature_pgd"]
    w2 = arg["w_pgd_pgdembed"]
    w3 = arg["w_pgd_eyeembed"]
    w4 = arg["w_pgdembed_feature"]
    w5 = arg["w_pgdembed_eyeembed"]
    w6 = arg["w_infoentropy_pgd"]
    w7 = arg["w_infoentropy_pgdembed"]
    w8 = arg["w_size"]
    w9 = arg["w_HA"]
    w10 = arg["w_YA"]
    weight_param = (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
    args.measure = arg["measure"]

    gaussian_model = gaussian_parameterized.PGDAttack(
        H_A=H_A2, Y_A=Y_A, model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type="CE", device=device)
    gaussian_model = gaussian_model.to(device)
    gaussian_model.attack(args, index_delete,
                          lr, 0, weight_sup, weight_param, feature_adj, 0, 0,
                          0, idx_train, idx_val,
                          idx_test, adj, features, init_adj, labels, idx_attack, num_edges, 0, epochs=args.epochs)

    inference_adj = gaussian_model.modified_adj.cpu()
    test(adj, features, labels, victim_model)
    auc = metric_pool(adj, inference_adj, idx_attack, index_delete)
    auc_train = metric_pool(adj, inference_adj, idx_train, index_delete_train)
    auc_all = metric_pool(adj, inference_adj, np.arange(
        adj.shape[0]), index_delete_all)
    os.makedirs("./results/", exist_ok=True)
    with open(os.path.join("./results", args.log_name), "a") as f:
        f.write(f"current gaussian parameter: {args}\n")
        f.write(f"attack graph: AUC={auc}\t")
        f.write(f"train graph: AUC={auc_train}\t")
        f.write(f"Whole Graph: AUC={auc_all}\n")
        f.write(f"current density: {inference_adj.mean()}\n")
        f.write(f"image name: {weight_param}_{weight_sup}_{lr}.png\n")
        f.write(
            "============================================================================================\n")
    return 1-auc


def gcn_reparameterize_attack(arg):
    ori_adj = adj.numpy()
    idx = idx_attack
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    index = np.where(real_edge == 0)[0]
    index_delete = index[:int(len(real_edge)-2*np.sum(real_edge))]

    idx = idx_train
    real_edge_train = ori_adj[idx, :][:, idx].reshape(-1)
    index_train = np.where(real_edge_train == 0)[0]
    index_delete_train = index_train[:int(
        len(real_edge_train)-2*np.sum(real_edge_train))]

    idx = np.arange(adj.shape[0])
    real_edge_all = ori_adj[idx, :][:, idx].reshape(-1)
    index_all = np.where(real_edge_all == 0)[0]
    index_delete_all = index_all[:int(
        len(real_edge_all)-2*np.sum(real_edge_all))]

    lr = arg["lrexp"]
    lr = 10**lr
    weight_sup = arg["weight_sup"]
    w1 = arg["w_feature_pgd"]
    w2 = arg["w_pgd_pgdembed"]
    w3 = arg["w_pgd_eyeembed"]
    w4 = arg["w_pgdembed_feature"]
    w5 = arg["w_pgdembed_eyeembed"]
    w6 = arg["w_infoentropy_pgd"]
    w7 = arg["w_infoentropy_pgdembed"]
    w8 = arg["w_size"]
    w9 = arg["w_HA"]
    w10 = arg["w_YA"]
    weight_param = (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
    args.measure = arg["measure"]

    gcn_model = gcn_parameterized.PGDAttack(H_A=H_A2, Y_A=Y_A, features=features, model=victim_model,
                                            embedding=embedding, nnodes=adj.shape[0], loss_type="CE", device=device)
    gcn_model = gcn_model.to(device)
    gcn_model.attack(args, index_delete,
                     lr, 0, weight_sup, weight_param, feature_adj, 0, 0,
                     0, idx_train, idx_val,
                     idx_test, adj, features, init_adj, labels, idx_attack, num_edges, 0, epochs=args.epochs)

    inference_adj = gcn_model.modified_adj.cpu()
    test(adj, features, labels, victim_model)
    auc = metric_pool(adj, inference_adj, idx_attack, index_delete)
    auc_train = metric_pool(adj, inference_adj, idx_train, index_delete_train)
    auc_all = metric_pool(adj, inference_adj, np.arange(
        adj.shape[0]), index_delete_all)
    os.makedirs("./results/", exist_ok=True)
    with open(os.path.join("./results", args.log_name), "a") as f:
        f.write(f"current gcn parameter: {args}\n")
        f.write(f"attack graph: AUC={auc}\t")
        f.write(f"train graph: AUC={auc_train}\t")
        f.write(f"Whole Graph: AUC={auc_all}\n")
        f.write(f"current density: {inference_adj.mean()}\n")
        f.write(f"image name: {weight_param}_{weight_sup}_{lr}.png\n")
        f.write(
            "============================================================================================\n")
    return 1-auc


if __name__ == '__main__':

    if args.mode == "baseline":
        test_baseline()
    if args.mode == "evaluate":
        evaluate()
    if args.mode == "notrain_test":
        notrain_test()
    if args.mode == "prepare":
        prepare()
    if args.mode == "gaussian":
        eval_gaussian()
    if args.mode == "gcn_attack":
        eval_gcn()

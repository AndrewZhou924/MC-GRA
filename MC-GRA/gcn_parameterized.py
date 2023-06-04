
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import utils
from base_attack import BaseAttack
from sklearn.metrics import auc, average_precision_score, roc_curve
from torch.nn import KLDivLoss, MSELoss
from torch.nn import functional as F
from torchmetrics import AUROC
from tqdm import tqdm
from utils import *


def calc_mrr(real_edge, pred_edge):
    false_samples = np.extract((real_edge == 0), pred_edge)
    true_samples = np.extract((real_edge == 1), pred_edge)
    sum = 0
    for x in true_samples:
        pos = (false_samples > x).astype(int).sum() + 1
        sum += pos/false_samples.size
    sum /= true_samples.size
    return sum


def calc_mrr_all(real_edge, pred_edge):
    true_samples = np.extract((real_edge == 1), pred_edge)
    sum = 0
    for x in true_samples:
        pos = (pred_edge > x).astype(int).sum() + 1
        sum += pos/pred_edge.size
    sum /= true_samples.size
    return sum


def calc_ap_at_n(real_edge, pred_edge):
    false_samples = np.extract((real_edge == 0), pred_edge)
    true_samples = np.extract((real_edge == 1), pred_edge)
    min_true = true_samples.min()
    false_samples = np.extract((false_samples >= min_true), false_samples)


def metric(ori_adj, inference_adj, idx, index_delete):
    auroc = AUROC(task='binary')
    real_edge = ori_adj[idx, :][:, idx].reshape(-1).cpu()
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1).cpu()
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    return auroc(pred_edge, real_edge)


def metric_pool(ori_adj, inference_adj, idx, index_delete):
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    fpr, tpr, _ = roc_curve(real_edge, pred_edge)
    AP = average_precision_score(real_edge, pred_edge)
    AUC = auc(fpr, tpr)

    MRR = calc_mrr(real_edge, pred_edge)
    MRR_ALL = calc_mrr_all(real_edge, pred_edge)

    return AP, AUC, MRR, MRR_ALL


def dot_product_decode(Z):
    Z = F.normalize(Z, p=2, dim=1)
    Z = torch.matmul(Z, Z.t())
    adj = torch.relu(Z-torch.eye(Z.shape[0]).to("cuda"))
    return adj


def sampling_MI(prob, tau=0.5, reduction='mean'):
    prob = prob.clamp(1e-4, 1-1e-4)
    entropy1 = prob * torch.log(prob / tau)
    entropy2 = (1-prob) * torch.log((1-prob) / (1-tau))
    res = entropy1 + entropy2
    if reduction == 'none':
        return res
    elif reduction == 'mean':
        return torch.mean(res)
    elif reduction == 'sum':
        return torch.sum(res)


def Info_entropy(prob):
    prob = torch.clamp(prob, 1e-4, 1-1e-4)
    entropy = prob * torch.log2(prob)
    return -torch.mean(entropy)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class PGDAttack(BaseAttack):

    def __init__(self, features=None, model=None, embedding=None, H_A=None, Y_A=None, nnodes=None, loss_type='CE', feature_shape=None,
                 attack_structure=True, attack_features=False, device='cpu'):
        super(PGDAttack, self).__init__(model, nnodes,
                                        attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        self.edge_select = None
        self.features = features.to(device)
        self.complementary = None
        self.complementary_after = None
        self.embedding = embedding
        self.H_A = H_A
        self.Y_A = Y_A
        self.adj_changes_after = torch.zeros(
            int(nnodes * (nnodes - 1) / 2), requires_grad=True)
        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.gc = deepcopy(embedding.gc)

        if attack_features:
            assert True, 'Topology Attack does not support attack feature'

    def test(self, idx_attack, idx_val, idx_test, adj, features, labels, victim_model):
        device = self.device
        adj, features, labels = to_tensor(adj, features, labels, device=device)

        victim_model.eval()
        adj_norm = normalize_adj_tensor(adj)
        output = victim_model(features, adj_norm)
        acc_test = accuracy(output[idx_test], labels[idx_test])

        return acc_test.item()

    def attack(self, args, index_delete, lr_ori, weight_aux, weight_supervised, weight_param, feature_adj,
               aux_adj, aux_feature, aux_num_edges, idx_train, idx_val, idx_test, adj,
               ori_features, ori_adj, labels, idx_attack, num_edges,
               dropout_rate, epochs=200, sample=False, **kwargs):
        '''
            Parameters:
            index_delete:           deleted zero edges, for metric
            lr_ori:                 learning rate
            weight:                 the weight for aux_loss1 and aux_loss2
            aux_adj:                adjancy matrix of aux graph
            aux_feature:            node feature of aux graph
            aux_num_edges:          no use yet.
            idx_train, idx_val:     index of nodes in train/val set. For testing, no use yet.
            idx_test:               index of nodes in test set. For testing.
            adj:                    adjancy matrix of origional graph
            ori_features:           node feature of origional graph
            labels:                 node labels of origional graph
            idx_attack:             index of nodes for recovery.
            num_edges:              no use in attack.
            epochs:                 epochs for recovery training.
            dropout_rate:           dropout rate in testing.
        '''

        if args.max_eval == 1:
            lr_ori = 10**args.lr
        self.args = args

        optimizer = torch.optim.Adam([self.adj_changes], lr=lr_ori)
        plt.cla()
        victim_model = self.surrogate
        self.sparse_features = sp.issparse(ori_features)
        fake_labels = labels
        ori_adj, ori_features, labels = utils.to_tensor(
            ori_adj, ori_features, labels, device=self.device)
        aux_adj, aux_feature, _ = utils.to_tensor(
            adj=aux_adj, features=aux_feature, labels=fake_labels, device=self.device)
        victim_model.eval()
        self.embedding.eval()
        adj = adj.to(self.device)
        label_adj = np.load("./saved_data/"+args.dataset+".npy")
        label_adj = torch.Tensor(label_adj).to(self.device)

        # lists for drawing
        acc_test_list = []
        origin_loss_list = []
        x_axis = []
        sparsity_list = []
        modified_adj = self.get_modified_adj(ori_adj)
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        output = victim_model(ori_features, modified_adj)
        self.delete_eye(modified_adj)

        adj_tmp = torch.eye(adj_norm.shape[0]).to(self.device)
        em = self.embedding(ori_features, adj_tmp)
        adj_changes = self.dot_product_decode(em)

        feature_adj = feature_adj.to(self.device)
        w1, w2, _, _, _, w6, w7, w8, w9, w10 = weight_param
        if args.max_eval == 1:
            w1 = args.w1
            w2 = args.w2
            w6 = args.w6
            w7 = args.w7
            w7 = args.w8
            w9 = args.w9
            w10 = args.w10
        for t in tqdm(range(epochs)):
            optimizer.zero_grad()
            modified_adj = self.get_modified_adj(ori_adj)
            modified_adj = self.adding_noise(modified_adj, args.eps)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            adj_tmp = torch.eye(adj_norm.shape[0]).to(self.device)
            em = self.embedding(ori_features, adj_tmp)
            adj_changes = self.dot_product_decode(em)

            origin_loss = self._loss(output[idx_attack], labels[idx_attack]) + torch.norm(self.adj_changes,
                                                                                          p=2) * 0.001
            origin_loss_list.append(origin_loss.item())
            loss = weight_supervised*origin_loss
            self.embedding.set_layers(1)
            H_A1 = self.embedding(ori_features, adj)
            self.embedding.set_layers(2)
            H_A2 = self.embedding(ori_features, adj)
            Y_A = victim_model(ori_features, adj)

            em = self.embedding(ori_features, modified_adj-ori_adj)
            self.adj_changes_after = self.dot_product_decode(em)
            modified_adj1 = self.get_modified_adj_after(ori_adj)
            adj_norm2 = utils.normalize_adj_tensor(modified_adj1)

            CKA = CudaCKA(device=self.device)
            calc = CKA.linear_HSIC
            calc2 = MSELoss()

            if args.measure == "MSELoss":
                calc = MSELoss()

            if args.measure == "KL":
                calc = self.calc_kl
            if args.measure == "KDE":
                calc = MutualInformation(
                    sigma=0.4, num_bins=feature_adj.shape[0], normalize=True)

            if args.measure == "CKA":
                CKA = CudaCKA(device=self.device)
                calc = CKA.linear_CKA

            if args.measure == "DP":
                calc = self.dot_product

            # 10 constrains area:
            c1 = c2 = _ = _ = _ = c6 = c7 = c8 = c9 = c10 = 0
            if w1 != 0 and feature_adj.max() != feature_adj.min():
                c1 = w1 * calc(feature_adj, adj_norm)*1000 * \
                    Align_Parameter_Cora["c1"]
                if args.measure == "KDE":
                    loss += c1[0]
                elif args.measure == "HSIC":
                    loss += -c1
                else:
                    loss += c1
            if w2 != 0:
                c2 = w2 * calc(adj_norm, modified_adj1) * \
                    100*Align_Parameter_Cora["c2"]
                if args.measure == "KDE":
                    loss += c2[0]
                elif args.measure == "HSIC":
                    loss += -c2
                else:
                    loss += c2
            if w6 != 0:
                c6 = w6 * Info_entropy(adj_norm)*100*Align_Parameter_Cora["c6"]
                loss += c6
            if w7 != 0:
                c7 = w7 * Info_entropy(modified_adj1) * \
                    Align_Parameter_Cora["c7"]
                loss += c7
            if w8 != 0:
                c8 = w8 * torch.clamp(torch.sum(torch.abs(self.adj_changes)),
                                      min=0.01)*0.0001*Align_Parameter_Cora["c8"]
                loss += c8
            if w9 != 0:
                num_layers = self.embedding.nlayer
                for i in range(num_layers-1, num_layers):
                    self.embedding.set_layers(i+1)
                    em_cur = self.embedding(
                        ori_features, modified_adj - ori_adj)
                    H_A_cur = self.embedding(ori_features, adj)
                    if args.measure == "KDE":
                        calc2 = MutualInformation(
                            sigma=0.4, num_bins=H_A1.shape[1], normalize=True)
                        c9 = w9 * \
                            (calc2(H_A_cur[idx_attack], em_cur[idx_attack])[
                             0])*Align_Parameter_Cora["c9"]
                    elif args.measure == "HSIC":
                        c9 = -1 * w9 * \
                            (calc(H_A_cur[idx_attack], em_cur[idx_attack])
                             )*Align_Parameter_Cora["c9"]
                    else:
                        c9 = w9 * \
                            (calc(H_A_cur[idx_attack], em_cur[idx_attack])
                             )*Align_Parameter_Cora["c9"]
                    loss += c9
            output2 = victim_model(ori_features, modified_adj)
            if w10 != 0:
                if args.measure == "KDE":
                    calc2 = MutualInformation(
                        sigma=0.4, num_bins=Y_A.shape[1], normalize=True)
                    c10 = w10 * calc2(Y_A[idx_attack], torch.softmax(output2[idx_attack], dim=1))[
                        0]*Align_Parameter_Cora["c10"]
                elif args.measure == "HSIC":
                    c10 = -w10 * calc(Y_A[idx_attack], torch.softmax(
                        output2[idx_attack], dim=1))*Align_Parameter_Cora["c10"]
                else:
                    c10 = w10 * calc(Y_A[idx_attack], torch.softmax(
                        output2[idx_attack], dim=1))*Align_Parameter_Cora["c10"]
                loss += c10

            loss.backward()

            if self.loss_type == 'CE':
                optimizer.step()

            self.projection(num_edges)
            self.adj_changes.data.copy_(torch.clamp(
                self.adj_changes.data, min=0, max=1))

            em = self.embedding(ori_features, adj_norm)
            adj_changes = self.dot_product_decode(em)
            modified_adj = self.get_modified_adj2(
                ori_adj, adj_changes).detach()
            victim_model.eval()
            modified_adj = self.get_modified_adj(ori_adj)
            sparsity_list.append(modified_adj.detach().cpu().mean())
            adj_norm2 = utils.normalize_adj_tensor(modified_adj)
            output2 = victim_model(ori_features, adj_norm2)
            cur_acc = utils.accuracy(
                output2[idx_test], labels[idx_test]).item()
            acc_test_list.append(cur_acc)

            x_axis.append(t)

        em = self.embedding(ori_features, adj_norm)
        self.adj_changes.data = self.dot_product_decode(em)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()

        self.embedding.set_layers(1)
        H_A1 = self.embedding(ori_features, self.modified_adj)
        self.embedding.set_layers(2)
        H_A2 = self.embedding(ori_features, self.modified_adj)
        Y_A2 = victim_model(ori_features, self.modified_adj)
        ori_HA = self.dot_product_decode2(self.H_A.detach())
        ori_YA = self.dot_product_decode2(self.Y_A.detach())
        H_A1 = self.dot_product_decode2(H_A1.detach())
        H_A2 = self.dot_product_decode2(H_A2.detach())
        Y_A2 = self.dot_product_decode2(Y_A2.detach())
        cur_adj = self.modified_adj + H_A1 + H_A2 + feature_adj + Y_A2
        if args.useH_A:
            cur_adj = cur_adj + ori_HA
        if args.useY_A:
            cur_adj = cur_adj + ori_YA
        if args.useY:
            cur_adj = cur_adj + label_adj

        self.modified_adj = cur_adj.detach()

        return 0, 0, 0, 0

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000 * onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
        return loss

    def projection(self, num_edges):
        if torch.clamp(self.adj_changes, 0, 1).sum() > num_edges:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, num_edges, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(
                self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(
                self.adj_changes.data, min=0, max=1))

    def get_modified_adj2(self, ori_adj, adj_changes):

        if self.complementary is None:
            self.complementary = torch.ones_like(
                ori_adj) - torch.eye(self.nnodes).to(self.device)

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes
        m = m + m.t()

        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def get_modified_adj(self, ori_adj):

        Identify_adj = torch.eye(ori_adj.shape[0]).to(self.device)
        x = self.features.detach().to(self.device)
        for layer in self.gc:
            x = F.relu(layer(x, Identify_adj))

        x = F.normalize(x, p=2, dim=1)
        x = torch.matmul(x, x.t())

        return x

    def get_modified_adj_after(self, ori_adj):

        if self.complementary_after is None:
            self.complementary_after = torch.ones_like(
                ori_adj) - torch.eye(self.nnodes).to(self.device)

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes_after
        m = m + m.t()

        modified_adj = self.complementary_after * m + ori_adj

        return modified_adj

    def bisection(self, a, b, num_edges, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes - x, 0, 1).sum() - num_edges

        miu = a
        while ((b - a) >= epsilon):
            miu = (a + b) / 2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu

    def dot_product_decode(self, Z):
        Z = F.normalize(Z, p=2, dim=1)
        A_pred = torch.relu(torch.matmul(Z, Z.t()))
        tril_indices = torch.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        return A_pred[tril_indices[0], tril_indices[1]]

    def dot_product_decode2(self, Z):
        if self.args.dataset in ['cora', 'citeseer']:
            Z = torch.matmul(Z, Z.t())
            _adj = torch.relu(Z-torch.eye(Z.shape[0]).to(self.device))
            _adj = torch.sigmoid(_adj)

        elif self.args.dataset in ['polblogs', 'usair', 'brazil']:
            Z = F.normalize(Z, p=2, dim=1)
            Z = torch.matmul(Z, Z.t()).to(self.device)
            _adj = torch.relu(Z-torch.eye(Z.shape[0]).to(self.device))

        elif self.args.dataset == 'AIDS':
            Z = torch.matmul(Z, Z.t())
            _adj = torch.relu(Z-torch.eye(Z.shape[0]).to(self.device))

        return _adj

    def delete_eye(self, A):
        complementary = torch.ones_like(
            A) - torch.eye(self.nnodes).to(self.device)
        A = A*complementary

    def adding_noise(self, modified_adj, eps=0):
        noise = torch.randn_like(modified_adj)
        modified_adj += noise*eps
        modified_adj = torch.clamp(modified_adj, max=1, min=0)
        return modified_adj

    def dot_product(self, X, Y):
        return torch.norm(torch.matmul(Y.t(), X), p=2)

    def calc_kl(self, X, Y):
        X = F.softmax(X)
        Y = F.log_softmax(Y)
        kl = KLDivLoss(reduction="batchmean")
        return kl(Y, X)

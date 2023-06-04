import numpy as np
import scipy.sparse as sp
import torch
import utils
from base_attack import BaseAttack
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm


class PGDAttack(BaseAttack):

    def __init__(self, model=None, embedding=None, nnodes=None, loss_type='CE', feature_shape=None,
                 attack_structure=True, attack_features=False, device='cpu'):
        super(PGDAttack, self).__init__(model, nnodes,
                                        attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        self.edge_select = None
        self.complementary = None
        self.embedding = embedding
        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(
                torch.FloatTensor(int(nnodes * (nnodes - 1) / 2)))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert True, 'Topology Attack does not support attack feature'

    def attack(self, index_delete, lr_ori, weight_aux, weight_supervised, weight_param, feature_adj,
               aux_adj, aux_feature, aux_num_edges, idx_train, idx_val, idx_test, adj,
               ori_features, ori_adj, labels, idx_attack, num_edges,
               dropout_rate, epochs=200, sample=False, **kwargs):

        optimizer = torch.optim.Adam([self.adj_changes], lr=lr_ori)

        victim_model = self.surrogate
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(
            ori_adj, ori_features, labels, device=self.device)

        victim_model.eval()
        self.embedding.eval()
        loss_list = []
        lr = lr_ori
        print(lr)
        for t in tqdm(range(epochs)):
            optimizer.zero_grad()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            if t > -1:
                loss = self._loss(output[idx_attack], labels[idx_attack]) + torch.norm(self.adj_changes,
                                                                                       p=2) * 0.001
            else:
                loss_smooth_feat = self.feature_smoothing(
                    modified_adj, ori_features)
                loss = self._loss(output[idx_attack], labels[idx_attack]) + torch.norm(self.adj_changes,
                                                                                       p=2) * 0.001 + 1e-4 * loss_smooth_feat

            test_acc = utils.accuracy(output[idx_attack], labels[idx_attack])
            print("loss= {:.4f}".format(loss.item()),
                  "test_accuracy= {:.4f}".format(test_acc.item()))
            loss_list.append(loss.item())
            loss.backward()

            if self.loss_type == 'CE':
                if sample:
                    lr = 200 / np.sqrt(t + 1)
                optimizer.step()

            self.projection(num_edges)
            self.adj_changes.data.copy_(torch.clamp(
                self.adj_changes.data, min=0, max=1))

        em = self.embedding(ori_features, adj_norm)
        self.adj_changes.data = self.dot_product_decode(em)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()

        return output.detach()

    def kmeans(self):
        center = np.random.choice(len(self.adj_changes), 2, replace=False)
        center = self.adj_changes[center]
        label = torch.zeros_like(self.adj_changes)
        for i in range(20):
            tmp0 = (self.adj_changes-center[0])**2
            tmp1 = (self.adj_changes-center[1])**2
            label = torch.min(
                torch.cat((tmp0.unsqueeze(0), tmp1.unsqueeze(0)), 0), 0)[1]
            label = label.float()
            tmp = torch.dot((torch.ones_like(label) - label),
                            self.adj_changes)/(torch.ones_like(label) - label).sum()
            if torch.abs(tmp - center[0]) < 1e-5:
                print("stop early! ", i)
                break

            center[0] = tmp
            center[1] = torch.dot(label, self.adj_changes) / label.sum()

        if center[0] > center[1]:
            label = torch.ones_like(label) - label
        return label

    def random_sample(self, ori_adj, ori_features, labels, idx_attack):
        K = 20
        best_loss = 1000
        victim_model = self.surrogate
        with torch.no_grad():
            ori_s = self.adj_changes.cpu().detach().numpy()
            s = ori_s / ori_s.sum()
            for _ in range(K):
                sampled = np.random.choice(len(s), 5000, replace=False, p=s)
                self.adj_changes.data.copy_(torch.zeros_like(torch.tensor(s)))
                for k in sampled:
                    self.adj_changes[k] = 1.0

                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss_smooth_feat = self.feature_smoothing(
                    modified_adj, ori_features)
                loss = self._loss(output[idx_attack], labels[idx_attack]) + torch.norm(self.adj_changes,
                                                                                       p=2) * 0.001 + 5e-7 * loss_smooth_feat
                test_acc = utils.accuracy(
                    output[idx_attack], labels[idx_attack])
                print("loss= {:.4f}".format(loss.item()),
                      "test_accuracy= {:.4f}".format(test_acc.item()))
                if best_loss > loss:
                    best_loss = loss
                    best_s = sampled

            self.adj_changes.data.copy_(torch.zeros_like(torch.tensor(s)))
            for k in best_s:
                self.adj_changes[k] = 1.0

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

    def feature_smoothing(self, adj, X):
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = torch.matmul(torch.matmul(r_mat_inv, L), r_mat_inv)

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat

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

    def get_modified_adj2(self):

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()

        return m

    def get_modified_adj(self, ori_adj):

        if self.complementary is None:
            self.complementary = torch.ones_like(
                ori_adj) - torch.eye(self.nnodes).to(self.device)

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()

        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def SVD(self):
        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes.detach()
        m = m + m.t()
        U, S, V = np.linalg.svd(m.cpu().numpy())
        U, S, V = torch.FloatTensor(U).to(self.device), torch.FloatTensor(S).to(self.device), torch.FloatTensor(V).to(
            self.device)
        alpha = 0.02
        tmp = torch.zeros_like(S).to(self.device)
        diag_S = torch.diag(torch.where(S > alpha, S, tmp))
        adj = torch.matmul(torch.matmul(U, diag_S), V)
        return adj[tril_indices[0], tril_indices[1]]

    def filter(self, Z):
        A = torch.zeros(Z.size()).to(self.device)
        return torch.where(Z > 0.9, Z, A)

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

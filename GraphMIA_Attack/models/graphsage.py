import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import utils
from copy import deepcopy
from sklearn.metrics import f1_score

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features*2, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(adj, input)
        else:
            support = torch.mm(adj, input)
        support = torch.cat([input, support], dim=1)
        output = torch.spmm(support, self.weight)
        #output = F.normalize(output, p=2, dim=1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class embedding_graphsage(nn.Module):
    def __init__(self, nfeat, nhid, nlayer=2, with_bias=False, device=None):

        super(embedding_graphsage, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.hidden_sizes = [nhid]
        self.gc1=GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc=[]
        self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
        for i in range(nlayer-1):
            self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
        # self.gc1=self.gc[0]
        # self.gc2=self.gc[1]
        self.with_bias = with_bias

    def forward(self, x, adj):
        # self.gc[0].to(self.device)
        # x = F.relu(self.gc1(x, adj)) 
        for i in range(self.nlayer):
            #print(i)
            layer=self.gc[i].to(self.device)
            x = F.relu(layer(x, adj))
        return x
    # def __init__(self, nfeat, nhid,nlayer=2, with_bias=True, device=None):

    #     super(embedding_GCN, self).__init__()

    #     assert device is not None, "Please specify 'device'!"
    #     self.device = device
    #     self.nfeat = nfeat
    #     self.hidden_sizes = [nhid]
    #     self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
    #     self.with_bias = with_bias

    # def forward(self, x, adj):
    #     x = F.relu(self.gc1(x, adj))
    #     return x

    def initialize(self):
        self.gc1.reset_parameters()
        for layer in self.gc:
            layer.reset_parameters()

        
    def set_layers(self, nlayer):
        self.nlayer = nlayer

class graphsage(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayer=2, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=False, device=None):

        super(graphsage, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.nlayer = nlayer
        self.gc=[]
        self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
        for i in range(nlayer-1):
            self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
        
        
        self.gc1 = self.gc[0]
        self.gc2 = self.gc1
        self.linear1 = nn.Linear(nhid ,nclass ,bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        for i,layer in enumerate(self.gc):
            layer=layer.to(self.device)
            if self.with_relu:
                x=F.relu(layer(x, adj))
            else:
                x=layer(x, adj)
            if i!= len(self.gc)-1:
                x=F.dropout(x,self.dropout, training=self.training)
        x = self.linear1(x)
        return F.log_softmax(x, dim=1)
    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layers in self.gc:
            layers.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=True, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
#
            saved_var = dict()
            for tensor_name, tensor in self.named_parameters():
                saved_var[tensor_name] = torch.zeros_like(tensor)
#
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            for tensor_name, tensor in self.named_parameters():
                new_grad = tensor.grad
                saved_var[tensor_name].add_(new_grad)

            for tensor_name, tensor in self.named_parameters():
                if self.device.type == 'cuda':
                    noise = torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, 0.05)
                else:
                    noise = torch.FloatTensor(tensor.grad.shape).normal_(0, 0.05)
                saved_var[tensor_name].add_(noise)
                tensor.grad = saved_var[tensor_name]

            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}, val acc: {}'.format(i, loss_train.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test


    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized data

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


# # import math
# # from copy import deepcopy

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # from torch.nn.modules.module import Module
# # from torch.nn.parameter import Parameter
# # from torchmetrics import AUROC
# # # from torchmetrics import AUROC
# # from tqdm import trange

# # import utils
# # from MI_constrain import CudaCKA


# # class GraphConvolution(Module):
# #     """Simple GCN layer, similar to https://github.com/tkipf/pygcn
# #     """

# #     def __init__(self, in_features, out_features, with_bias=True):
# #         super(GraphConvolution, self).__init__()
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
# #         if with_bias:
# #             self.bias = Parameter(torch.FloatTensor(out_features))
# #         else:
# #             self.register_parameter('bias', None)
# #         self.reset_parameters()

# #     def reset_parameters(self):
# #         stdv = 1. / math.sqrt(self.weight.size(1))
# #         self.weight.data.uniform_(-stdv, stdv)
# #         if self.bias is not None:
# #             self.bias.data.uniform_(-stdv, stdv)

# #     def forward(self, input, adj):
# #         """ Graph Convolutional Layer forward function
# #         """
# #         if input.data.is_sparse:
# #             support = torch.spmm(input, self.weight)
# #         else:
# #             support = torch.mm(input, self.weight)
# #         output = torch.spmm(adj, support)
# #         if self.bias is not None:
# #             return output + self.bias
# #         else:
# #             return output

# #     def __repr__(self):
# #         return self.__class__.__name__ + ' (' \
# #                + str(self.in_features) + ' -> ' \
# #                + str(self.out_features) + ')'

# # # class embedding_GCN(nn.Module):
# # #     def __init__(self, nfeat, nhid, nlayer=1, with_bias=True, device=None):

# # #         super(embedding_GCN, self).__init__()

# # #         assert device is not None, "Please specify 'device'!"
# # #         self.device = device
# # #         self.nfeat = nfeat
# # #         self.nlayer = nlayer
# # #         self.hidden_sizes = [nhid]
# # #         self.gc=[]
# # #         self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
# # #         for i in range(nlayer-1):
# # #             self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
# # #         self.gc1=self.gc[0]
# # #         # self.gc2=self.gc[1]
# # #         self.with_bias = with_bias

# # #     def forward(self, x, adj):
# # #         # for i in range(self.nlayer):
# # #         #     layer=self.gc[i].to(self.device)
# # #         #     x = F.relu(layer(x, adj))
# # #         # return x
# # #         x = F.relu(self.gc1(x, adj))
# # #         return x

# # #     def initialize(self):
# # #         self.gc1.reset_parameters()
# # #         for layer in self.gc:
# # #             layer.rset_parameters()

        
# # #     def set_layers(self, nlayer):
# # #         self.nlayer = nlayer

# # # class embedding_GCN(nn.Module):
# # #     def __init__(self, nfeat, nhid, nlayer=2, with_bias=True, device=None):

# # #         super(embedding_GCN, self).__init__()

# # #         assert device is not None, "Please specify 'device'!"
# # #         self.device = device
# # #         self.nfeat = nfeat
# # #         self.nlayer = nlayer
# # #         self.hidden_sizes = [nhid]
# # #         self.gc1=GraphConvolution(nfeat, nhid, with_bias=with_bias)
# # #         self.gc=[]
# # #         self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
# # #         for i in range(nlayer-1):
# # #             self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
# # #         # self.gc1=self.gc[0]
# # #         # self.gc2=self.gc[1]
# # #         self.with_bias = with_bias

# # #     def forward(self, x, adj):
# # #         # self.gc[0].to(self.device)
# # #         # x = F.relu(self.gc1(x, adj)) 
# # #         for i in range(self.nlayer):
# # #             #print(i)
# # #             layer=self.gc[i].to(self.device)
# # #             x = F.relu(layer(x, adj))
# # #         return x

# # #     def initialize(self):
# # #         self.gc1.reset_parameters()
# # #         for layer in self.gc:
# # #             layer.rset_parameters()

        
# # #     def set_layers(self, nlayer):
# # #         self.nlayer = nlayer

# # class embedding_GCN(nn.Module):
# #     def __init__(self, nfeat, nhid, nlayer=1, with_bias=True, device=None):

# #         super(embedding_GCN, self).__init__()

# #         assert device is not None, "Please specify 'device'!"
# #         self.device = device
# #         self.nfeat = nfeat
# #         self.nlayer = nlayer
# #         self.hidden_sizes = [nhid]
# #         self.gc1=GraphConvolution(nfeat, nhid, with_bias=with_bias)
# #         self.gc=[]
# #         self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
# #         for i in range(nlayer-1):
# #             self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
# #         # self.gc1=self.gc[0]
# #         # self.gc2=self.gc[1]
# #         self.with_bias = with_bias

# #     def forward(self, x, adj):
# #         # self.gc[0].to(self.device)
# #         # x = F.relu(self.gc1(x, adj)) 
# #         for i in range(self.nlayer):
# #             #print(i)
# #             layer=self.gc[i].to(self.device)
# #             if i != self.nlayer-1:
# #                 x = F.relu(layer(x, adj))
# #             else:
# #                 x = layer(x,adj)
# #         return x

# #     def initialize(self):
# #         self.gc1.reset_parameters()
# #         for layer in self.gc:
# #             layer.rset_parameters()

        
# #     def set_layers(self, nlayer):
# #         self.nlayer = nlayer


# # class GCN(nn.Module):
# #     """ 2 Layer Graph Convolutional Network.

# #     Parameters
# #     ----------
# #     nfeat : int
# #         size of input feature dimension
# #     nhid : int
# #         number of hidden units
# #     nclass : int
# #         size of output dimension
# #     dropout : float
# #         dropout rate for GCN
# #     lr : float
# #         learning rate for GCN
# #     weight_decay : float
# #         weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
# #     with_relu : bool
# #         whether to use relu activation function. If False, GCN will be linearized.
# #     with_bias: bool
# #         whether to include bias term in GCN weights.
# #     device: str
# #         'cpu' or 'cuda'.
# #     """

# #     def __init__(
# #         self, 
# #         nfeat, 
# #         nhid, 
# #         nclass, 
# #         nlayer=2, 
# #         dropout=0.5, 
# #         lr=0.01, 
# #         weight_decay=5e-4, 
# #         with_relu=True, 
# #         with_bias=True, 
# #         device=None
# #     ):

# #         super(GCN, self).__init__()

# #         assert device is not None, "Please specify 'device'!"
# #         self.device = device
# #         self.nfeat = nfeat
# #         self.hidden_sizes = [nhid]
# #         self.nclass = nclass
# #         self.nlayer = nlayer
# #         self.gc=[]
# #         self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
# #         for i in range(nlayer-1):
# #             self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))

# #         self.gc1=self.gc[0]
# #         self.gc2=self.gc[1]

# #         self.linear1 = nn.Linear(nhid ,nclass ,bias=with_bias)
# #         self.dropout = dropout
# #         self.lr = lr
# #         if not with_relu:
# #             self.weight_decay = 0
# #         else:
# #             self.weight_decay = weight_decay
# #         self.with_relu = with_relu
# #         self.with_bias = with_bias
# #         self.output = None
# #         self.best_model = None
# #         self.best_output = None
# #         self.adj_norm = None
# #         self.features = None
# #         self.origin_adj = None
        
# #         self.initialize()


# #     def forward(self, x, adj):
# #         node_emb = []
# #         for i,layer in enumerate(self.gc):
# #             layer=layer.to(self.device)
# #             # 最后一层不添加 relu
# #             if self.with_relu and i!= len(self.gc)-1:
# #                 x=F.relu(layer(x, adj))
# #             else:
# #                 x=layer(x, adj)
# #             node_emb.append(x)
# #             if i!= len(self.gc)-1:
# #                 x=F.dropout(x,self.dropout, training=self.training)

# #         x = self.linear1(x)
# #         node_emb.append(x)
# #         return F.log_softmax(x, dim=1), node_emb

# #     def initialize(self):
# #         """Initialize parameters of GCN.
# #         """
# #         for layers in self.gc:
# #             layers.reset_parameters()

# #     def fit(
# #             self, 
# #             features, 
# #             adj, 
# #             labels, 
# #             idx_train, 
# #             idx_val=None,
# #             train_iters=200, 
# #             initialize=True, 
# #             verbose=True, 
# #             normalize=True, 
# #             patience=500, 
# #             beta=None,
# #             MI_type='MI',
# #             stochastic=0,
# #             con=0,
# #             **kwargs
# #         ):
# #         """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

# #         Parameters
# #         ----------
# #         features :
# #             node features
# #         adj :
# #             the adjacency matrix. The format could be torch.tensor or scipy matrix
# #         labels :
# #             node labels
# #         idx_train :
# #             node training indices
# #         idx_val :
# #             node validation indices. If not given (None), GCN training process will not adpot early stopping
# #         train_iters : int
# #             number of training epochs
# #         initialize : bool
# #             whether to initialize parameters before training
# #         verbose : bool
# #             whether to show verbose logs
# #         normalize : bool
# #             whether to normalize the input adjacency matrix.
# #         patience : int
# #             patience for early stopping, only valid when `idx_val` is given
# #         """

# #         self.device = self.gc1.weight.device
# #         # if initialize:
# #         #     self.initialize()

# #         if type(adj) is not torch.Tensor:
# #             features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
# #         else:
# #             features = features.to(self.device)
# #             adj = adj.to(self.device)
# #             labels = labels.to(self.device)

# #         if normalize:
# #             if utils.is_sparse_tensor(adj):
# #                 adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
# #             else:
# #                 adj_norm = utils.normalize_adj_tensor(adj)
# #         else:
# #             adj_norm = adj

# #         self.adj_norm = adj_norm
# #         self.features = features
# #         self.labels = labels
# #         self.origin_adj = adj
        
# #         if stochastic:
# #             print('=== training with random Aug ===')
# #             features, adj = utils.stochastic(features, adj_norm)

# #         if beta is None and not con:
# #             print('=== training plain gcn model ===')
# #             self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)
# #         elif con:
# #             self._train_with_contrastive(labels, idx_train, idx_val, train_iters, verbose)
# #         else:
# #             print('train with MI constrain')
# #             layer_aucs = self._train_with_MI_constrain(labels=labels, idx_train=idx_train, idx_val=idx_val, train_iters=train_iters, beta=beta, MI_type=MI_type, verbose=verbose)
# #             return layer_aucs
            
# #     def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
# #         # if verbose:
# #         #     print('=== training gcn model ===')
# #         # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# #         # best_loss_val = 100
# #         # best_acc_val = 0

# #         # for i in range(train_iters):
# #         #     self.train()
# #         #     optimizer.zero_grad()
# #         #     output, node_embs = self.forward(self.features, self.adj_norm)

# #         #     def dot_product_decode(Z,):
# #         #         Z = torch.matmul(Z, Z.t())
# #         #         adj = Z-torch.eye(Z.shape[0]).to(Z.get_device())
# #         #         return adj
        
# #         #     def calculate_AUC(Z, Adj):
# #         #         auroc_metric = AUROC(task='binary')
# #         #         Z = Z.detach()
# #         #         Z = dot_product_decode(Z)

# #         #         real_edge = Adj.reshape(-1)
# #         #         pred_edge = Z.reshape(-1)

# #         #         auc_score = auroc_metric(pred_edge, real_edge)
# #         #         return auc_score
# #         #     layer_aucs = []
# #         #     for idx, node_emb in enumerate(node_embs):
# #         #         layer_aucs.append(calculate_AUC(node_emb, self.origin_adj).item())

# #         #     loss_train = F.nll_loss(output[idx_train], labels[idx_train])
# #         #     loss_train.backward()
# #         #     optimizer.step()

# #         #     self.eval()
# #         #     output = self.forward(self.features, self.adj_norm)[0]
# #         #     loss_val = F.nll_loss(output[idx_val], labels[idx_val])
# #         #     acc_val = utils.accuracy(output[idx_val], labels[idx_val])

# #         #     if verbose and i % 10 == 0:
# #         #         print('Epoch {}, training loss: {}, val acc: {}'.format(i, loss_train.item(), acc_val))

# #         #     if best_loss_val > loss_val:
# #         #         best_loss_val = loss_val
# #         #         self.output = output
# #         #         weights = deepcopy(self.state_dict())

# #         #     if acc_val > best_acc_val:
# #         #         best_acc_val = acc_val
# #         #         self.output = output
# #         #         weights = deepcopy(self.state_dict())

# #         # if verbose:
# #         #     print('=== picking the best model according to the performance on validation ===')
# #         # self.load_state_dict(weights)
# #         # return layer_aucs
# #         if verbose:
# #             print('=== training gcn model ===')
# #         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# #         best_loss_val = 100
# #         best_acc_val = 0

# #         for i in range(train_iters):
# #             self.train()
# #             optimizer.zero_grad()
# #             output = self.forward(self.features, self.adj_norm)[0]
# #             loss_train = F.nll_loss(output[idx_train], labels[idx_train])
# #             loss_train.backward()
# #             optimizer.step()

# #             self.eval()
# #             output = self.forward(self.features, self.adj_norm)[0]
# #             loss_val = F.nll_loss(output[idx_val], labels[idx_val])
# #             acc_val = utils.accuracy(output[idx_val], labels[idx_val])

# #             if verbose and i % 10 == 0:
# #                 print('Epoch {}, training loss: {}, val acc: {}'.format(i, loss_train.item(), acc_val))

# #             if best_loss_val > loss_val:
# #                 best_loss_val = loss_val
# #                 self.output = output
# #                 weights = deepcopy(self.state_dict())

# #             if acc_val > best_acc_val:
# #                 best_acc_val = acc_val
# #                 self.output = output
# #                 weights = deepcopy(self.state_dict())

# #         if verbose:
# #             print('=== picking the best model according to the performance on validation ===')
# #         self.load_state_dict(weights)
# #         return None


# #     def _train_with_MI_constrain(self, labels, idx_train, idx_val, train_iters, beta, MI_type='MI', verbose=True):
# #         if verbose:
# #             print('=== training MI constrain ===')
# #         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# #         best_loss_val = 100
# #         best_acc_val = 0

# #         IAZ_func = getattr(utils, MI_type) # MI, HSIC, LP, linear_CKA

# #         def dot_product_decode(Z,):
# #             Z = torch.matmul(Z, Z.t())
# #             adj = Z-torch.eye(Z.shape[0]).to(Z.device)
# #             return adj
        
# #         def calculate_AUC(Z, Adj):
# #             auroc_metric = AUROC(task='binary')
# #             Z = Z.detach()
# #             Z = dot_product_decode(Z)

# #             real_edge = Adj.reshape(-1)
# #             pred_edge = Z.reshape(-1)

# #             auc_score = auroc_metric(pred_edge, real_edge)
# #             return auc_score

# #         for i in trange(train_iters, desc='training...'):
# #             self.train()
# #             optimizer.zero_grad()
# #             output, node_embs = self.forward(self.features, self.adj_norm) # adj_norm
            
# #             # 添加参数, 多层beta 与 单层beta

# #             # loss_IAZ = beta * IAZ_func(self.adj_norm, node_embs[0])

# #             loss_IAZ = 0
# #             loss_inter = 0
# #             loss_mission = 0
# #             layer_aucs = []
# #             # TODO:
# #             # 1. 多层
# #             for idx, node_emb in enumerate(node_embs):
# #                 # 层间约束
# #                 # if (idx+1) <= len(node_embs)-1:
# #                 #     next_node_emb = node_embs[idx+1]
# #                 #     next_node_emb = (next_node_emb@next_node_emb.T)
# #                 #     loss_inter += IAZ_func(next_node_emb, node_emb)

# #                 # MI约束
# #                 param_name = 'layer-{}'.format(idx)
# #                 beta_cur = beta[param_name]
# #                 # TODO: 采样到2M边， BCELoss, (gt 取反 0->1)
# #                 loss_IAZ += beta_cur * IAZ_func(self.adj_norm, node_emb)

# #                 layer_aucs.append(calculate_AUC(node_emb, self.origin_adj).item())
# #                 # poblogs beta /=2
# #                 # beta_cur /= 4

# #                 # 任务约束
# #                 # 排除linear层
# #                 if idx != len(node_embs)-1:
# #                     output_layer = self.linear1(node_emb)
# #                     output_layer = F.log_softmax(output_layer, dim=1)
# #                     loss_mission += F.nll_loss(output_layer[idx_train], labels[idx_train])
  
# #             output = F.log_softmax(output, dim=1)
# #             loss_IYZ = F.nll_loss(output[idx_train], labels[idx_train])

# #             loss_train = loss_IYZ + loss_IAZ + loss_inter + loss_mission
# #             loss_train.backward()
# #             optimizer.step()

# #             self.eval()
# #             output = self.forward(self.features, self.adj_norm)[0]
# #             output = F.log_softmax(output, dim=1)
# #             loss_val = F.nll_loss(output[idx_val], labels[idx_val])
# #             acc_val = utils.accuracy(output[idx_val], labels[idx_val])

# #             if verbose and i % 10 == 0:
 
# #                 print('Epoch {}, loss_IYZ: {}, loss_IAZ: {}, val acc: {}'.format( 
# #                     i, 
# #                     round(loss_IYZ.item(), 4), 
# #                     round(loss_IAZ.item(), 4),
# #                     acc_val
# #                 ))

# #             if best_loss_val > loss_val:
# #                 best_loss_val = loss_val
# #                 self.output = output
# #                 weights = deepcopy(self.state_dict())
# #                 final_layer_aucs = layer_aucs

# #             if acc_val > best_acc_val:
# #                 best_acc_val = acc_val
# #                 self.output = output
# #                 weights = deepcopy(self.state_dict())
# #                 final_layer_aucs = layer_aucs

# #         if verbose:
# #             print('=== picking the best model according to the performance on validation ===')
# #         # self.load_state_dict(weights)
# #         return final_layer_aucs

# #     def _train_with_contrastive(self, labels, idx_train, idx_val, train_iters, verbose):
# #         if verbose:
# #             print('=== training contrastive model ===')
# #         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# #         best_loss_val = 100
# #         best_acc_val = 0

# #         cl_criterion = utils.SelfAdversarialClLoss()

# #         for i in range(train_iters):
# #             self.train()
# #             optimizer.zero_grad()
# #             output = self.forward(self.features, self.adj_norm)[0]

# #             x1, adj1 = utils.stochastic(self.features, self.adj_norm)
# #             x2, adj2 = utils.stochastic(self.features, self.adj_norm)

# #             node_embs_1 = self.forward(x1, adj1)[1]
# #             node_embs_2 = self.forward(x2, adj2)[1]

# #             last_gc_1 = F.normalize(node_embs_1[-2], dim=1)
# #             last_gc_2 = F.normalize(node_embs_2[-2], dim=1)

# #             loss_cl = utils.stochastic_loss(last_gc_1, last_gc_2, cl_criterion, margin=1e3)

# #             loss_train = F.nll_loss(output[idx_train], labels[idx_train])

# #             loss_train += loss_cl
# #             loss_train.backward()
# #             optimizer.step()


# #             self.eval()
# #             output = self.forward(self.features, self.adj_norm)[0]
# #             loss_val = F.nll_loss(output[idx_val], labels[idx_val])
# #             acc_val = utils.accuracy(output[idx_val], labels[idx_val])

# #             if verbose and i % 10 == 0:
# #                 print('Epoch {}, training loss: {}, val acc: {}'.format(i, loss_train.item(), acc_val))

# #             if best_loss_val > loss_val:
# #                 best_loss_val = loss_val
# #                 self.output = output
# #                 weights = deepcopy(self.state_dict())

# #             if acc_val > best_acc_val:
# #                 best_acc_val = acc_val
# #                 self.output = output
# #                 weights = deepcopy(self.state_dict())

# #         if verbose:
# #             print('=== picking the best model according to the performance on validation ===')
# #         self.load_state_dict(weights)

# #     def test(self, idx_test):
# #         """Evaluate GCN performance on test set.

# #         Parameters
# #         ----------
# #         idx_test :
# #             node testing indices
# #         """
# #         self.eval()
# #         output = self.predict()
# #         # output = self.output
# #         loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
# #         acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
# #         print("Test set results:",
# #               "loss= {:.4f}".format(loss_test.item()),
# #               "accuracy= {:.4f}".format(acc_test.item()))
# #         return acc_test


# #     def predict(self, features=None, adj=None):
# #         """By default, the inputs should be unnormalized data

# #         Parameters
# #         ----------
# #         features :
# #             node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
# #         adj :
# #             adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


# #         Returns
# #         -------
# #         torch.FloatTensor
# #             output (log probabilities) of GCN
# #         """

# #         self.eval()
# #         if features is None and adj is None:
# #             return self.forward(self.features, self.adj_norm)
# #         else:
# #             if type(adj) is not torch.Tensor:
# #                 features, adj = utils.to_tensor(features, adj, device=self.device)

# #             self.features = features
# #             if utils.is_sparse_tensor(adj):
# #                 self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
# #             else:
# #                 self.adj_norm = utils.normalize_adj_tensor(adj)
# #             return self.forward(self.features, self.adj_norm)


# import math
# from copy import deepcopy

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.nn.modules.module import Module
# from torch.nn.parameter import Parameter
# from torchmetrics import AUROC
# # from torchmetrics import AUROC
# from tqdm import trange

# import utils
# from MI_constrain import CudaCKA


# class GraphConvolution(Module):
#     """Simple GCN layer, similar to https://github.com/tkipf/pygcn
#     """

#     def __init__(self, in_features, out_features, with_bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if with_bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj):
#         """ Graph Convolutional Layer forward function
#         """
#         if input.data.is_sparse:
#             support = torch.spmm(input, self.weight)
#         else:
#             support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'

# # class embedding_GCN(nn.Module):
# #     def __init__(self, nfeat, nhid, nlayer=1, with_bias=True, device=None):

# #         super(embedding_GCN, self).__init__()

# #         assert device is not None, "Please specify 'device'!"
# #         self.device = device
# #         self.nfeat = nfeat
# #         self.nlayer = nlayer
# #         self.hidden_sizes = [nhid]
# #         self.gc=[]
# #         self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
# #         for i in range(nlayer-1):
# #             self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
# #         self.gc1=self.gc[0]
# #         # self.gc2=self.gc[1]
# #         self.with_bias = with_bias

# #     def forward(self, x, adj):
# #         # for i in range(self.nlayer):
# #         #     layer=self.gc[i].to(self.device)
# #         #     x = F.relu(layer(x, adj))
# #         # return x
# #         x = F.relu(self.gc1(x, adj))
# #         return x

# #     def initialize(self):
# #         self.gc1.reset_parameters()
# #         for layer in self.gc:
# #             layer.rset_parameters()

        
# #     def set_layers(self, nlayer):
# #         self.nlayer = nlayer

# class embedding_GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nlayer=2, with_bias=True, device=None):

#         super(embedding_GCN, self).__init__()

#         assert device is not None, "Please specify 'device'!"
#         self.device = device
#         self.nfeat = nfeat
#         self.nlayer = nlayer
#         self.hidden_sizes = [nhid]
#         self.gc1=GraphConvolution(nfeat, nhid, with_bias=with_bias)
#         self.gc=[]
#         self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
#         for i in range(nlayer-1):
#             self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
#         # self.gc1=self.gc[0]
#         # self.gc2=self.gc[1]
#         self.with_bias = with_bias

#     def forward(self, x, adj):
#         # self.gc[0].to(self.device)
#         # x = F.relu(self.gc1(x, adj)) 
#         # for i in range(self.nlayer):
#         #     #print(i)
#         #     layer=self.gc[i].to(self.device)
#         #     x = F.relu(layer(x, adj))
#         # return x

#         x = F.relu(self.gc1(x, adj))
#         return x

#     def initialize(self):
#         self.gc1.reset_parameters()
#         for layer in self.gc:
#             layer.rset_parameters()

        
#     def set_layers(self, nlayer):
#         self.nlayer = nlayer


# class GCN(nn.Module):
#     """ 2 Layer Graph Convolutional Network.

#     Parameters
#     ----------
#     nfeat : int
#         size of input feature dimension
#     nhid : int
#         number of hidden units
#     nclass : int
#         size of output dimension
#     dropout : float
#         dropout rate for GCN
#     lr : float
#         learning rate for GCN
#     weight_decay : float
#         weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
#     with_relu : bool
#         whether to use relu activation function. If False, GCN will be linearized.
#     with_bias: bool
#         whether to include bias term in GCN weights.
#     device: str
#         'cpu' or 'cuda'.
#     """

#     def __init__(
#         self, 
#         nfeat, 
#         nhid, 
#         nclass, 
#         nlayer=2, 
#         dropout=0.5, 
#         lr=0.01, 
#         weight_decay=5e-4, 
#         with_relu=True, 
#         with_bias=True, 
#         device=None
#     ):

#         super(GCN, self).__init__()

#         assert device is not None, "Please specify 'device'!"
#         self.device = device
#         self.nfeat = nfeat
#         self.hidden_sizes = [nhid]
#         self.nclass = nclass
#         self.nlayer = nlayer
#         self.gc=[]
#         self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
#         for i in range(nlayer-1):
#             self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))

#         self.gc1=self.gc[0]
#         self.gc2=self.gc[1]

#         self.linear1 = nn.Linear(nhid ,nclass ,bias=with_bias)
#         self.dropout = dropout
#         self.lr = lr
#         if not with_relu:
#             self.weight_decay = 0
#         else:
#             self.weight_decay = weight_decay
#         self.with_relu = with_relu
#         self.with_bias = with_bias
#         self.output = None
#         self.best_model = None
#         self.best_output = None
#         self.adj_norm = None
#         self.features = None
#         self.origin_adj = None
        
#         self.initialize()


#     def forward(self, x, adj):
#         node_emb = []
#         for i,layer in enumerate(self.gc):
#             layer=layer.to(self.device)
#             # 最后一层不添加 relu
#             if self.with_relu and i!= len(self.gc)-1:
#                 x=F.relu(layer(x, adj))
#             else:
#                 x=layer(x, adj)
          
#             if i!= len(self.gc)-1:
#                 x=F.dropout(x,self.dropout, training=self.training)

#             node_emb.append(x)
#         x = self.linear1(x)
#         node_emb.append(x)
#         return F.log_softmax(x, dim=1), node_emb

#     def initialize(self):
#         """Initialize parameters of GCN.
#         """
#         for layers in self.gc:
#             layers.reset_parameters()

#     def fit(
#             self, 
#             features, 
#             adj, 
#             labels, 
#             idx_train, 
#             idx_val=None,
#             train_iters=200, 
#             initialize=True, 
#             verbose=True, 
#             normalize=True, 
#             patience=500, 
#             beta=None,
#             MI_type='MI',
#             stochastic=0,
#             con=0,
#             **kwargs
#         ):
#         """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

#         Parameters
#         ----------
#         features :
#             node features
#         adj :
#             the adjacency matrix. The format could be torch.tensor or scipy matrix
#         labels :
#             node labels
#         idx_train :
#             node training indices
#         idx_val :
#             node validation indices. If not given (None), GCN training process will not adpot early stopping
#         train_iters : int
#             number of training epochs
#         initialize : bool
#             whether to initialize parameters before training
#         verbose : bool
#             whether to show verbose logs
#         normalize : bool
#             whether to normalize the input adjacency matrix.
#         patience : int
#             patience for early stopping, only valid when `idx_val` is given
#         """

#         self.device = self.gc1.weight.device
#         # if initialize:
#         #     self.initialize()

#         if type(adj) is not torch.Tensor:
#             features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
#         else:
#             features = features.to(self.device)
#             adj = adj.to(self.device)
#             labels = labels.to(self.device)

#         if normalize:
#             if utils.is_sparse_tensor(adj):
#                 adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
#             else:
#                 adj_norm = utils.normalize_adj_tensor(adj)
#         else:
#             adj_norm = adj

#         self.adj_norm = adj_norm
#         self.features = features
#         self.labels = labels
#         self.origin_adj = adj
        
#         if stochastic:
#             print('=== training with random Aug ===')
#             features, adj = utils.stochastic(features, adj_norm)

#         if beta is None and not con:
#             self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)
#         elif con:
#             self._train_with_contrastive(labels, idx_train, idx_val, train_iters, verbose)
#         else:
#             print('train with MI constrain')
#             layer_aucs = self._train_with_MI_constrain(labels=labels, idx_train=idx_train, idx_val=idx_val, train_iters=train_iters, beta=beta, MI_type=MI_type, verbose=verbose)
#             return layer_aucs
            
#     def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
#         if verbose:
#             print('=== training gcn model ===')
#         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

#         best_loss_val = 100
#         best_acc_val = 0

#         for i in range(train_iters):
#             self.train()
#             optimizer.zero_grad()
#             output = self.forward(self.features, self.adj_norm)[0]

#             loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#             loss_train.backward()
#             optimizer.step()

#             self.eval()
#             output = self.forward(self.features, self.adj_norm)[0]
#             loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#             acc_val = utils.accuracy(output[idx_val], labels[idx_val])

#             if verbose and i % 10 == 0:
#                 print('Epoch {}, training loss: {}, val acc: {}'.format(i, loss_train.item(), acc_val))

#             if best_loss_val > loss_val:
#                 best_loss_val = loss_val
#                 self.output = output
#                 weights = deepcopy(self.state_dict())

#             if acc_val > best_acc_val:
#                 best_acc_val = acc_val
#                 self.output = output
#                 weights = deepcopy(self.state_dict())

#         if verbose:
#             print('=== picking the best model according to the performance on validation ===')
#         self.load_state_dict(weights)

#     def _train_with_MI_constrain(self, labels, idx_train, idx_val, train_iters, beta, MI_type='MI', verbose=True):
#         if verbose:
#             print('=== training MI constrain ===')
#         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

#         best_loss_val = 100
#         best_acc_val = 0

#         IAZ_func = getattr(utils, MI_type) # MI, HSIC, LP, linear_CKA

#         def dot_product_decode(Z,):
#             Z = torch.matmul(Z, Z.t())
#             adj = Z-torch.eye(Z.shape[0]).to(Z.get_device())
#             return adj
        
#         def calculate_AUC(Z, Adj):
#             auroc_metric = AUROC(task='binary')
#             Z = Z.detach()
#             Z = dot_product_decode(Z)

#             real_edge = Adj.reshape(-1)
#             pred_edge = Z.reshape(-1)

#             auc_score = auroc_metric(pred_edge, real_edge)
#             return auc_score

#         for i in trange(train_iters, desc='training...'):
#             self.train()
#             optimizer.zero_grad()
#             output, node_embs = self.forward(self.features, self.adj_norm) # adj_norm
            
#             # 添加参数, 多层beta 与 单层beta

#             # loss_IAZ = beta * IAZ_func(self.adj_norm, node_embs[0])

#             loss_IAZ = 0
#             loss_inter = 0
#             loss_mission = 0
#             layer_aucs = []
#             # TODO:
#             # 1. 多层
#             for idx, node_emb in enumerate(node_embs):
#                 # 层间约束
#                 # if (idx+1) <= len(node_embs)-1:
#                 #     next_node_emb = node_embs[idx+1]
#                 #     next_node_emb = (next_node_emb@next_node_emb.T)
#                 #     loss_inter += IAZ_func(next_node_emb, node_emb)

#                 # MI约束
#                 param_name = 'layer-{}'.format(idx)
#                 beta_cur = beta[param_name]
#                 # TODO: 采样到2M边， BCELoss, (gt 取反 0->1)
#                 loss_IAZ += beta_cur * IAZ_func(self.adj_norm, node_emb)

#                 layer_aucs.append(calculate_AUC(node_emb, self.origin_adj).item())
#                 # poblogs beta /=2
#                 # beta_cur /= 4

#                 # 任务约束
#                 # 排除linear层
#                 if idx != len(node_embs)-1:
#                     output_layer = self.linear1(node_emb)
#                     output_layer = F.log_softmax(output_layer, dim=1)
#                     loss_mission += F.nll_loss(output_layer[idx_train], labels[idx_train])
  
#             output = F.log_softmax(output, dim=1)
#             loss_IYZ = F.nll_loss(output[idx_train], labels[idx_train])

#             loss_train = loss_IYZ + loss_IAZ + loss_inter + loss_mission
#             loss_train.backward()
#             optimizer.step()

#             self.eval()
#             output = self.forward(self.features, self.adj_norm)[0]
#             output = F.log_softmax(output, dim=1)
#             loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#             acc_val = utils.accuracy(output[idx_val], labels[idx_val])

#             if verbose and i % 10 == 0:
 
#                 print('Epoch {}, loss_IYZ: {}, loss_IAZ: {}, val acc: {}'.format( 
#                     i, 
#                     round(loss_IYZ.item(), 4), 
#                     round(loss_IAZ.item(), 4),
#                     acc_val
#                 ))

#             if best_loss_val > loss_val:
#                 best_loss_val = loss_val
#                 self.output = output
#                 weights = deepcopy(self.state_dict())

#             if acc_val > best_acc_val:
#                 best_acc_val = acc_val
#                 self.output = output
#                 weights = deepcopy(self.state_dict())

#         if verbose:
#             print('=== picking the best model according to the performance on validation ===')
#         self.load_state_dict(weights)
#         return layer_aucs

#     def _train_with_contrastive(self, labels, idx_train, idx_val, train_iters, verbose):
#         if verbose:
#             print('=== training contrastive model ===')
#         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

#         best_loss_val = 100
#         best_acc_val = 0

#         cl_criterion = utils.SelfAdversarialClLoss()

#         for i in range(train_iters):
#             self.train()
#             optimizer.zero_grad()
#             output = self.forward(self.features, self.adj_norm)[0]

#             x1, adj1 = utils.stochastic(self.features, self.adj_norm)
#             x2, adj2 = utils.stochastic(self.features, self.adj_norm)

#             node_embs_1 = self.forward(x1, adj1)[1]
#             node_embs_2 = self.forward(x2, adj2)[1]

#             last_gc_1 = F.normalize(node_embs_1[-2], dim=1)
#             last_gc_2 = F.normalize(node_embs_2[-2], dim=1)

#             loss_cl = utils.stochastic_loss(last_gc_1, last_gc_2, cl_criterion, margin=1e3)

#             loss_train = F.nll_loss(output[idx_train], labels[idx_train])

#             loss_train += loss_cl
#             loss_train.backward()
#             optimizer.step()


#             self.eval()
#             output = self.forward(self.features, self.adj_norm)[0]
#             loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#             acc_val = utils.accuracy(output[idx_val], labels[idx_val])

#             if verbose and i % 10 == 0:
#                 print('Epoch {}, training loss: {}, val acc: {}'.format(i, loss_train.item(), acc_val))

#             if best_loss_val > loss_val:
#                 best_loss_val = loss_val
#                 self.output = output
#                 weights = deepcopy(self.state_dict())

#             if acc_val > best_acc_val:
#                 best_acc_val = acc_val
#                 self.output = output
#                 weights = deepcopy(self.state_dict())

#         if verbose:
#             print('=== picking the best model according to the performance on validation ===')
#         self.load_state_dict(weights)

#     def test(self, idx_test):
#         """Evaluate GCN performance on test set.

#         Parameters
#         ----------
#         idx_test :
#             node testing indices
#         """
#         self.eval()
#         output = self.predict()
#         # output = self.output
#         loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
#         acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
#         print("Test set results:",
#               "loss= {:.4f}".format(loss_test.item()),
#               "accuracy= {:.4f}".format(acc_test.item()))
#         return acc_test


#     def predict(self, features=None, adj=None):
#         """By default, the inputs should be unnormalized data

#         Parameters
#         ----------
#         features :
#             node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
#         adj :
#             adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


#         Returns
#         -------
#         torch.FloatTensor
#             output (log probabilities) of GCN
#         """

#         self.eval()
#         if features is None and adj is None:
#             return self.forward(self.features, self.adj_norm)
#         else:
#             if type(adj) is not torch.Tensor:
#                 features, adj = utils.to_tensor(features, adj, device=self.device)

#             self.features = features
#             if utils.is_sparse_tensor(adj):
#                 self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
#             else:
#                 self.adj_norm = utils.normalize_adj_tensor(adj)
#             return self.forward(self.features, self.adj_norm)


import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torchmetrics import AUROC
# from torchmetrics import AUROC
from tqdm import trange

import utils
from MI_constrain import CudaCKA


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class embedding_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nlayer=1, with_bias=True, device=None):

        super(embedding_GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.hidden_sizes = [nhid]
        self.gc=[]
        self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
        for i in range(nlayer-1):
            self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
        self.gc1=self.gc[0]
        # self.gc2=self.gc[1]
        self.with_bias = with_bias

    def forward(self, x, adj):
        for i in range(self.nlayer):
            layer=self.gc[i].to(self.device)
            x = F.relu(layer(x, adj))
        return x
        # x = F.relu(self.gc1(x, adj))
        return x

    def initialize(self):
        self.gc1.reset_parameters()
        for layer in self.gc:
            layer.rset_parameters()

        
    def set_layers(self, nlayer):
        self.nlayer = nlayer

# class embedding_GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nlayer=2, with_bias=True, device=None):

#         super(embedding_GCN, self).__init__()

#         assert device is not None, "Please specify 'device'!"
#         self.device = device
#         self.nfeat = nfeat
#         self.nlayer = nlayer
#         self.hidden_sizes = [nhid]
#         self.gc1=GraphConvolution(nfeat, nhid, with_bias=with_bias)
#         self.gc=[]
#         self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
#         for i in range(nlayer-1):
#             self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
#         # self.gc1=self.gc[0]
#         # self.gc2=self.gc[1]
#         self.with_bias = with_bias

#     def forward(self, x, adj):
#         # self.gc[0].to(self.device)
#         # x = F.relu(self.gc1(x, adj)) 
#         for i in range(self.nlayer):
#             #print(i)
#             layer=self.gc[i].to(self.device)
#             x = F.relu(layer(x, adj))
#         return x

#     def initialize(self):
#         self.gc1.reset_parameters()
#         for layer in self.gc:
#             layer.rset_parameters()

        
#     def set_layers(self, nlayer):
#         self.nlayer = nlayer


class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.
    """

    def __init__(
        self, 
        nfeat, 
        nhid, 
        nclass, 
        nlayer=2, 
        dropout=0.5, 
        lr=0.01, 
        weight_decay=5e-4, 
        with_relu=True, 
        with_bias=True, 
        device=None
    ):

        super(GCN, self).__init__()

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

        self.gc1=self.gc[0]
        self.gc2=self.gc[1]

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
        self.origin_adj = None
        
        self.initialize()


    def forward(self, x, adj):
        node_emb = []
        for i,layer in enumerate(self.gc):
            layer=layer.to(self.device)
            # 最后一层不添加 relu
            if self.with_relu: # and i!= len(self.gc)-1
                x=F.relu(layer(x, adj))
            else:
                x=layer(x, adj)
          
            if i!= len(self.gc)-1:
                x=F.dropout(x,self.dropout, training=self.training)

            node_emb.append(x)
        x = self.linear1(x)
        node_emb.append(x)
        return F.log_softmax(x, dim=1), node_emb

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layers in self.gc:
            layers.reset_parameters()

    def fit(
            self, 
            features, 
            adj, 
            labels, 
            idx_train, 
            idx_val=None,
            train_iters=200, 
            initialize=True, 
            verbose=True, 
            normalize=True, 
            patience=500, 
            beta=None,
            MI_type='KDE',
            stochastic=0,
            con=0,
            aug_pe=0.1,
            **kwargs
        ):
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
        # if initialize:
        #     self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if stochastic:
            print('=== training with random Aug ===')
            features, adj = utils.stochastic(features, adj, pe=aug_pe)

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
        self.origin_adj = adj
        
        if beta is None and not con:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)
        elif con:
            self._train_with_contrastive(labels, idx_train, idx_val, train_iters, verbose)
        else:
            print('train with MI constrain')
            layer_aucs = self._train_with_MI_constrain(labels=labels, idx_train=idx_train, idx_val=idx_val, train_iters=train_iters, beta=beta, MI_type=MI_type, verbose=verbose)
            return layer_aucs
            
    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)[0]

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)[0]
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

    def _train_with_MI_constrain(self, labels, idx_train, idx_val, train_iters, beta, MI_type='MI', verbose=True):
        if verbose:
            print('=== training MI constrain ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        IAZ_func = getattr(utils, MI_type) # MI, HSIC, LP, linear_CKA

        def dot_product_decode(Z,):
            Z = torch.matmul(Z, Z.t())
            adj = torch.relu(Z-torch.eye(Z.shape[0]).to(Z.device))
            return adj
        
        def calculate_AUC(Z, Adj):
            auroc_metric = AUROC(task='binary')
            Z = Z.detach()
            Z = dot_product_decode(Z)

            real_edge = Adj.reshape(-1)
            pred_edge = Z.reshape(-1)

            auc_score = auroc_metric(pred_edge, real_edge)
            return auc_score

        # beta = {
        #     'layer-0': beta,
        #     'layer-1': 0,
        #     'layer-2': 0,
        # }

        for i in trange(train_iters, desc='training...'):
            self.train()
            optimizer.zero_grad()
            output, node_embs = self.forward(self.features, self.adj_norm) # adj_norm
            
            # 添加参数, 多层beta 与 单层beta

            # loss_IAZ = beta * IAZ_func(self.adj_norm, node_embs[0])

            loss_IAZ = 0
            loss_inter = 0
            loss_mission = 0
            layer_aucs = []
            # TODO:
            # 1. 多层
            for idx, node_emb in enumerate(node_embs):
                # 层间约束
                if (idx+1) <= len(node_embs)-1:
                    param_inter = 'layer_inter-{}'.format(idx)
                    beta_inter = beta[param_inter]
                    next_node_emb = node_embs[idx+1]
                    next_node_emb = (next_node_emb@next_node_emb.T)
                    loss_inter += beta_inter * IAZ_func(next_node_emb, node_emb)

                # MI约束
                param_name = 'layer-{}'.format(idx)
                beta_cur = beta[param_name]
                # TODO: 采样到2M边， BCELoss, (gt 取反 0->1)
                loss_IAZ += beta_cur * IAZ_func(self.adj_norm, node_emb)

                layer_aucs.append(calculate_AUC(node_emb, self.origin_adj).item())

                # 任务约束
                # 排除linear层
                if idx != len(node_embs)-1:
                    output_layer = self.linear1(node_emb)
                    output_layer = F.log_softmax(output_layer, dim=1)
                    loss_mission += F.nll_loss(output_layer[idx_train], labels[idx_train])
  
            output = F.log_softmax(output, dim=1)
            loss_IYZ = F.nll_loss(output[idx_train], labels[idx_train])

            loss_train = loss_IYZ + loss_IAZ + loss_inter + loss_mission
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)[0]
            output = F.log_softmax(output, dim=1)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 10 == 0:
 
                print('Epoch {}, loss_IYZ: {}, loss_IAZ: {}, val acc: {}'.format( 
                    i, 
                    round(loss_IYZ.item(), 4), 
                    round(loss_IAZ.item(), 4),
                    acc_val
                ))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                final_layer_aucs = layer_aucs

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                final_layer_aucs = layer_aucs

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        return final_layer_aucs

    def _train_with_contrastive(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training contrastive model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        cl_criterion = utils.SelfAdversarialClLoss()

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)[0]

            x1, adj1 = utils.stochastic(self.features, self.adj_norm)
            x2, adj2 = utils.stochastic(self.features, self.adj_norm)

            node_embs_1 = self.forward(x1, adj1)[1]
            node_embs_2 = self.forward(x2, adj2)[1]

            last_gc_1 = F.normalize(node_embs_1[-2], dim=1)
            last_gc_2 = F.normalize(node_embs_2[-2], dim=1)

            loss_cl = utils.stochastic_loss(last_gc_1, last_gc_2, cl_criterion, margin=1e3)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            loss_train += loss_cl
            loss_train.backward()
            optimizer.step()


            self.eval()
            output = self.forward(self.features, self.adj_norm)[0]
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



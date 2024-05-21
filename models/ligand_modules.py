import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import List, Tuple, Any, Optional, Union


def build_mlp(in_dim: int,
              h_dim: Union[int, List],
              out_dim: int = None,
              dropout_p: float = 0.2,
              activation: str = 'relu') -> nn.Sequential:
    """Builds an MLP.
    Parameters
    ----------
    in_dim: int,
        Input dimension of the MLP
    h_dim: int,
        Hidden layer dimension of the MLP
    out_dim: int, default None
        Output size of the MLP. If None, a Linear layer is returned, with ReLU
    dropout_p: float, default 0.2,
        Dropout probability
    """
    if isinstance(h_dim, int):
        h_dim = [h_dim]

    sizes = [in_dim] + h_dim
    mlp_size_tuple = list(zip(*(sizes[:-1], sizes[1:])))

    if isinstance(dropout_p, float):
        dropout_p = [dropout_p] * len(mlp_size_tuple)

    layers = []

    for idx, (prev_size, next_size) in enumerate(mlp_size_tuple):
        layers.append(nn.Linear(prev_size, next_size))
        if activation == 'relu':
            layers.append(nn.LeakyReLU())
        elif activation == 'lrelu':
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p[idx]))

    if out_dim is not None:
        layers.append(nn.Linear(sizes[-1], out_dim))

    return nn.Sequential(*layers)



class WLNConvLast(MessagePassing):

    def __init__(self, hsize: int, bias: bool):
        super(WLNConvLast, self).__init__(aggr='mean')
        self.hsize = hsize
        self.bias = bias
        self._build_components()

    def _build_components(self):
        self.W0 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.W1 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.W2 = nn.Linear(self.hsize, self.hsize, self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        mess = self.W0(x_i) * self.W1(edge_attr) * self.W2(x_j)
        return mess



class WLNConv(MessagePassing):

    def __init__(self,
                 node_fdim: int,
                 edge_fdim: int,
                 depth: int, hsize: int,
                 bias: bool = False,
                 dropout: float = 0.2,
                 activation: str = 'relu',
                 jk_pool: str = None):
        super(WLNConv, self).__init__(aggr='mean') # We use mean here because the node embeddings started to explode otherwise
        self.hsize = hsize
        self.bias = bias
        self.depth = depth
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.dropout_p = dropout
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'lrelu':
            self.activation_fn = F.leaky_relu
        self.jk_pool = jk_pool
        self._build_components()

    def _build_components(self):
        self.node_emb = nn.Linear(self.node_fdim, self.hsize, self.bias)
        self.mess_emb = nn.Linear(self.edge_fdim, self.hsize, self.bias)

        self.U1 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.U2 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.V = nn.Linear(2 * self.hsize, self.hsize, self.bias)

        self.dropouts = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)
        self.conv_last = WLNConvLast(hsize=self.hsize, bias=self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        if x.size(-1) != self.hsize:
            x = self.node_emb(x)
        edge_attr = self.mess_emb(edge_attr)

        x_depths = []
        for i in range(self.depth):
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            x = self.dropouts[i](x)
            x_depth = self.conv_last(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x_depths.append(x_depth)

        x_final = x_depths[-1]
        if self.jk_pool == 'max':
            x_final = torch.stack(x_depths, dim=-1).max(dim=-1)[0]

        elif self.jk_pool == "concat":
            x_final = torch.cat(x_depths, dim=-1)
        return x_final

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        x = self.activation_fn(self.U1(x) + self.U2(inputs))
        return x

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        nei_mess = self.activation_fn(self.V(torch.cat([x_j, edge_attr], dim=-1)))
        return nei_mess












import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (FiLMConv, GATConv, GATv2Conv, GCNConv,
                                GraphConv, LEConv, LGConv, Linear,
                                MessagePassing, ResGatedGraphConv, SAGEConv,
                                TransformerConv)
from torch_geometric.utils import degree


class NGCFConv(MessagePassing):
    def __init__(self, latent_dim, dropout=0.1, bias=True, **kwargs):
        super(NGCFConv, self).__init__(aggr="add", **kwargs)
        self.dropout = dropout
        self.lin_1 = nn.Linear(latent_dim, latent_dim, bias=bias)
        self.lin_2 = nn.Linear(latent_dim, latent_dim, bias=bias)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.lin_1.weight)
        nn.init.xavier_uniform_(self.lin_2.weight)

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        # Start propagating messages
        out = self.propagate(edge_index, x=(x, x), norm=norm)
        out += self.lin_1(x)
        out = F.dropout(out, self.dropout, self.training)
        return F.leaky_relu(out)

    def message(self, x_j, x_i, norm):
        return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))


def op(layer, hidden_dim, aggr="max"):
    if layer == "linear":
        return Linear(-1, hidden_dim)
    elif layer == "gcn":
        return GCNConv(-1, hidden_dim)
    elif layer == "sage":
        return SAGEConv(-1, hidden_dim, aggr=aggr)
    elif layer == "gat":
        return GATConv(-1, hidden_dim, add_self_loops=False)
    elif layer == "gatv2":
        return GATv2Conv(-1, hidden_dim, add_self_loops=False)
    elif layer == "graph":
        return GraphConv(-1, hidden_dim, aggr=aggr)
    elif layer == "le":
        return LEConv(-1, hidden_dim)
    elif layer == "film":
        return FiLMConv(-1, hidden_dim, aggr=aggr)
    elif layer == "resgated":
        return ResGatedGraphConv(-1, hidden_dim)
    elif layer == "transformer":
        return TransformerConv(-1, hidden_dim)
    elif layer == "lightgcn":
        return LGConv(True)
    elif layer == "ngcf":
        return NGCFConv(hidden_dim)
    else:
        raise ValueError(f"layer type {layer} is not supported")

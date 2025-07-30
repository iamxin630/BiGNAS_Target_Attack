import torch
import torch.nn.functional as F
from torch.nn import Identity
from torch_geometric.nn import BatchNorm

from ops import op


class Conv(torch.nn.Module):
    def __init__(self, space, output_dim, dropout=0.0, bn=True, aggr="max") -> None:
        super().__init__()
        self.bn = bn
        self.aggr = aggr
        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = Identity()

        self.core = torch.nn.ModuleDict(
            {name: op(name, output_dim, aggr) for name in space}
        )

        self.bns = torch.nn.ModuleDict(
            {name: BatchNorm(output_dim) if self.bn else Identity() for name in space}
        )

        self.alpha = torch.nn.Parameter(
            torch.randn(len(space)) * 1e-3, requires_grad=True
        )

    def forward(self, x, edge_index):
        res = []
        for key in self.core:
            tmp = (
                self.core[key](x, edge_index) if key != "linear" else self.core[key](x)
            )
            tmp = self.bns[key](tmp)
            tmp = self.dropout(tmp)
            res.append(tmp)
        res = torch.stack(res, dim=0)
        alpha_shape = [-1] + [1] * (len(res.size()) - 1)
        res = torch.sum(
            res * F.softmax(self.alpha, -1).view(*alpha_shape).to(res.device), 0
        )
        return res

    def reset_parameters(self):
        for key in self.core:
            self.core[key].reset_parameters()
            if hasattr(self.bns[key], "reset_parameters"):
                self.bns[key].reset_parameters()

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(Conv, self).named_parameters():
            if name == "alpha":
                continue
            yield name, p

import logging

import torch

from conv import Conv


class Supernet(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        dropout=0.5,
        space=["gcn", "gat", "gatv2", "sage", "graph", "lightgcn", "linear"],
        bn=True,
        aggr="max",
    ) -> None:
        super().__init__()
        self.space = [space] * num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            out_c = hidden_dim
            self.convs.append(Conv(self.space[i], out_c, dropout, bn, aggr))

    def forward(self, x, edge_index):
        embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embs.append(x)
        x = torch.cat(embs, dim=1)
        return x

    def print_alpha(self):
        for idx, conv in enumerate(self.convs):
            logging.info(f"layer {idx}, alphas softmax: {conv.alpha.softmax(dim=-1)}")

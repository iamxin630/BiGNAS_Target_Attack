import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from ops import op
from supernet import Supernet


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.space = args.space
        self.bn = args.bn
        self.aggr = args.aggr
        self.num_users = args.num_users
        self.num_source_items = args.num_source_items
        self.num_target_items = args.num_target_items

        self.embedding_dim = args.embedding_dim
        self.user_embedding = nn.Embedding(self.num_users, args.embedding_dim)
        self.source_item_embedding = nn.Embedding(
            self.num_source_items, args.embedding_dim
        )
        self.target_item_embedding = nn.Embedding(
            self.num_target_items, args.embedding_dim
        )

        self.source_supernet = Supernet(
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            self.space,
            self.bn,
            self.aggr,
        )
        self.target_supernet = Supernet(
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            self.space,
            self.bn,
            self.aggr,
        )

        self.user_mix_linear = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                for _ in range(self.num_layers)
            ]
        )

        self.pred_input_dim = (
            self.embedding_dim * 2 + self.hidden_dim * self.num_layers * 2
        )
        self.source_preds = nn.ModuleList([nn.Linear(self.pred_input_dim, 1)])
        self.target_preds = nn.ModuleList([nn.Linear(self.pred_input_dim, 1)])
        self.replace_target_preds = nn.ModuleList([nn.Linear(self.pred_input_dim, 1)])

        self.init_parameters()

    def forward(self, source_edge_index, target_edge_index, link, is_source=True):
        source_x = torch.cat(
            [self.user_embedding.weight, self.source_item_embedding.weight], dim=0
        )
        target_x = torch.cat(
            [self.user_embedding.weight, self.target_item_embedding.weight], dim=0
        )

        source_embs = [source_x]
        target_embs = [target_x]
        for i in range(self.num_layers):
            source_x = self.source_supernet.convs[i](source_x, source_edge_index)
            target_x = self.target_supernet.convs[i](target_x, target_edge_index)
            user_emb = self.user_mix_linear[i](
                torch.cat(
                    [source_x[: self.num_users], target_x[: self.num_users]], dim=1
                )
            )
            source_x = torch.cat([user_emb, source_x[self.num_users :]], dim=0)
            target_x = torch.cat([user_emb, target_x[self.num_users :]], dim=0)

            source_embs.append(source_x)
            target_embs.append(target_x)

        source_embs = torch.cat(source_embs, dim=1)
        target_embs = torch.cat(target_embs, dim=1)

        user_embs = source_embs[link[0, :]]
        item_embs = source_embs[link[1, :]] if is_source else target_embs[link[1, :]]
        x = torch.cat([user_embs, item_embs], dim=1)

        preds = self.source_preds if is_source else self.target_preds
        for lin in preds:
            x = lin(x)
            x = F.leaky_relu(x)
        out = x.sigmoid()

        return out

    def meta_prediction(self, source_edge_index, target_edge_index, link):
        source_x = torch.cat(
            [self.user_embedding.weight, self.source_item_embedding.weight]
        )
        target_x = torch.cat(
            [self.user_embedding.weight, self.target_item_embedding.weight]
        )

        source_embs = [source_x]
        target_embs = [target_x]
        for i in range(self.num_layers):
            source_x = self.source_supernet.convs[i](source_x, source_edge_index)
            target_x = self.target_supernet.convs[i](target_x, target_edge_index)
            user_emb = self.user_mix_linear[i](
                torch.cat(
                    [source_x[: self.num_users], target_x[: self.num_users]], dim=1
                )
            )
            source_x = torch.cat([user_emb, source_x[self.num_users :]], dim=0)
            target_x = torch.cat([user_emb, target_x[self.num_users :]], dim=0)

            source_embs.append(source_x)
            target_embs.append(target_x)

        source_embs = torch.cat(source_embs, dim=1)
        target_embs = torch.cat(target_embs, dim=1)

        user_embs = source_embs[link[0, :]]
        item_embs = target_embs[link[1, :]]
        x = torch.cat([user_embs, item_embs], dim=1)

        preds = self.replace_target_preds
        for lin in preds:
            x = lin(x)
            x = F.leaky_relu(x)
        out = x.sigmoid()

        return out

    def init_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.source_item_embedding.weight, std=0.01)
        nn.init.normal_(self.target_item_embedding.weight, std=0.01)

    def print_alpha(self):
        logging.info("source supernet alpha:")
        self.source_supernet.print_alpha()
        logging.info("target supernet alpha:")
        self.target_supernet.print_alpha()


class Perceptor(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.domain_prior = nn.Parameter(torch.ones(1, 1))
        self.item_prior_convs = nn.ModuleList()
        for _ in range(args.meta_num_layers):
            self.item_prior_convs.append(op(args.meta_op, args.meta_hidden_dim))
        self.item_prior_linear = nn.Linear(args.meta_hidden_dim, 1)

    def forward(self, item, edge_index, ref_model):
        x = torch.cat(
            [ref_model.user_embedding.weight, ref_model.source_item_embedding.weight]
        )
        for conv in self.item_prior_convs:
            x = conv(x, edge_index)
        item_prior = self.item_prior_linear(x)[item]
        return torch.relu(self.domain_prior) * torch.softmax(item_prior, dim=0)

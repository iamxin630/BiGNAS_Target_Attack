import gzip
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch

from model import Model


def link_split(data):
    # source graph
    source_link = data.source_link
    source_label = data.source_label
    source_edge_index = torch.cat([source_link, source_link[[1, 0]]], dim=1)

    # target graph
    split_mask = data.split_mask

    target_train_link = data.target_link[:, split_mask["train"]]
    target_train_label = data.target_label[split_mask["train"]]
    
    target_train_edge_index = torch.cat(
        [target_train_link, target_train_link[[1, 0]]], dim=1
    )

    target_valid_link = data.target_link[:, split_mask["valid"]]
    target_valid_label = data.target_label[split_mask["valid"]]

    target_test_link = data.target_link[:, split_mask["test"]]
    target_test_label = data.target_label[split_mask["test"]]
    target_test_edge_index = torch.cat(
        [target_test_link, target_test_link[[1, 0]]], dim=1
    )
    # print(f"target_test_link shape: {target_test_link.shape}")
    # print("Test users:", torch.unique(target_test_link[0]))

    return (
        source_edge_index,
        source_label,
        source_link,
        target_train_edge_index,
        target_train_label,
        target_train_link,
        target_valid_link,
        target_valid_label,
        target_test_link,
        target_test_label,
        target_test_edge_index, # ✅ 加上這行
    )


def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def set_logging():
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt=DATE_FORMAT,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_model(args):
    model = Model(args)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.to(args.device)
    return model

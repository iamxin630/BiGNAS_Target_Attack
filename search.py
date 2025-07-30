import argparse
import logging
import os
import time

import wandb

from dataset import CrossDomain
from model import Model, Perceptor
from train import train
from utils import set_logging, set_seed


def search(args):
    args.search = True

    wandb.init(project="BiGNAS", config=args)
    set_seed(args.seed)
    set_logging()

    logging.info(f"args: {args}")

    dataset = CrossDomain(
        root=args.root,
        categories=args.categories,
        target=args.target,
        use_source=args.use_source,
    )

    data = dataset[0]
    args.num_users = data.num_users
    args.num_source_items = data.num_source_items
    args.num_target_items = data.num_target_items
    logging.info(f"data: {data}")

    DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"
    args.model_path = os.path.join(
        args.model_dir,
        f'{time.strftime(DATE_FORMAT, time.localtime())}_{"_".join(args.categories)}.pt',
    )

    model = Model(args)
    perceptor = Perceptor(args)
    logging.info(f"model: {model}")
    train(model, perceptor, data, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device & mode settings
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--search", default=False, action="store_true")
    parser.add_argument("--use-meta", default=False, action="store_true")
    parser.add_argument("--use-source", default=False, action="store_true")

    # dataset settings
    parser.add_argument(
        "--categories", type=str, nargs="+", default=["Electronic", "Clothing"]
    )
    parser.add_argument("--target", type=str, default="Clothing")
    parser.add_argument("--root", type=str, default="data/")

    # model settings
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--model-dir", type=str, default="./save/")

    # supernet settings
    parser.add_argument(
        "--space",
        type=str,
        nargs="+",
        default=["gcn", "gatv2", "sage", "lightgcn", "linear"],
    )
    parser.add_argument("--warm-up", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--entropy", type=float, default=0.0)

    # training settings
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eta-min", type=float, default=0.001)
    parser.add_argument("--T-max", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=15, help="Top-K for hit ratio evaluation") #新增top k


    # meta settings
    parser.add_argument("--meta-interval", type=int, default=50)
    parser.add_argument("--meta-num-layers", type=int, default=2)
    parser.add_argument("--meta-hidden-dim", type=int, default=32)
    parser.add_argument("--meta-batch-size", type=int, default=512)
    parser.add_argument("--conv-lr", type=float, default=1)
    parser.add_argument("--hpo-lr", type=float, default=0.01)
    parser.add_argument("--descent-step", type=int, default=10)
    parser.add_argument("--meta-op", type=str, default="gat")
    
    parser.add_argument('--user_attack_ratio', type=float, default=1.0,
    help='Proportion of users to attack with fake edges, value between 0 and 1')
    parser.add_argument('--attack_strategy',type=str,default='related',  # 給預設值避免無此屬性錯誤
    choices=['related', 'unrelated', 'random'],help='Fake edge attack strategy')



    args = parser.parse_args()
    search(args)

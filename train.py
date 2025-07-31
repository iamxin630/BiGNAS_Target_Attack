import logging
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from auxilearn.optim import MetaOptimizer
from dataset import Dataset
from pytorchtools import EarlyStopping
from utils import link_split, load_model

import random
from collections import Counter


def meta_optimizeation(
    target_meta_loader,
    replace_optimizer,
    model,
    args,
    criterion,
    replace_scheduler,
    source_edge_index,
    target_edge_index,
):
    device = args.device
    for batch, (target_link, target_label) in enumerate(target_meta_loader):
        if batch < args.descent_step:
            target_link, target_label = target_link.to(device), target_label.to(device)

            replace_optimizer.zero_grad()
            out = model.meta_prediction(
                source_edge_index, target_edge_index, target_link
            ).view(-1)  # 用 view(-1) 避免 squeeze 問題
            loss_target = criterion(out, target_label).mean()
            loss_target.backward()
            replace_optimizer.step()
        else:
            break
    replace_scheduler.step()


@torch.no_grad()
def evaluate(name, model, source_edge_index, target_edge_index, link, label):
    model.eval()
    out = model(source_edge_index, target_edge_index, link, is_source=False).view(-1)
    try:
        auc = roc_auc_score(label.tolist(), out.tolist())
    except Exception as e:
        logging.warning(f"roc_auc_score failed: {e}")
        auc = 0.5  # 0.5 表示隨機猜測
    logging.info(f"{name} AUC: {auc:4f}")
    model.train()
    return auc


def get_test_positive_dict(data):
    """取得測試集用戶正向互動字典，格式：{ user_id: [item1, item2, ...] }"""
    test_user_item_dict = {}
    test_link = data.target_test_link.cpu()
    for u, i in zip(test_link[0], test_link[1]):
        u, i = u.item(), i.item()
        test_user_item_dict.setdefault(u, []).append(i)
    return test_user_item_dict


def evaluate_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    top_k, num_candidates=99,
    device=None
):
    model.eval()
    hit_count = 0
    all_target_items = set(range(data.num_target_items))

    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())
    logging.info(f"Test set user count: {len(sim_users)}")

    total_users = 0
    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    with torch.no_grad():
        for user_id in sim_users:
            pos_items = user_interactions.get(user_id, [])
            if len(pos_items) > 1:
                logging.warning(f"User {user_id} has {len(pos_items)} positives in test set.")

            if not pos_items:
                continue

            pos_item = pos_items[0]
            negative_pool = list(all_target_items - set(pos_items))
            if len(negative_pool) < num_candidates:
                continue

            sampled_negatives = random.sample(negative_pool, num_candidates)
            candidate_items = sampled_negatives + [pos_item]
            random.shuffle(candidate_items)

            user_tensor = torch.tensor([user_id] * len(candidate_items), device=device)
            item_tensor = torch.tensor(candidate_items, device=device)
            link = torch.stack([user_tensor, item_tensor], dim=0)

            scores = model(source_edge_index, target_edge_index, link, is_source=False).view(-1)
            top_k_indices = torch.topk(scores, k=top_k).indices.tolist()
            top_k_items = [candidate_items[i] for i in top_k_indices]

            if pos_item in top_k_items:
                hit_count += 1
            total_users += 1

    hit_ratio = hit_count / total_users if total_users > 0 else 0.0
    logging.info(f"[HIT_RATIO@{top_k}] Users={total_users}, Hits={hit_count}, Hit Ratio={hit_ratio:.4f}")
    return hit_ratio


def evaluate_er_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    cold_item_set,
    top_k, num_candidates=99,
    device=None
):
    import random
    model.eval()

    all_target_items = set(range(data.num_target_items))
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())

    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    total_users = 0
    cold_item_hit_count = 0
    cold_item_ranks = []

    with torch.no_grad():
        for user_id in sim_users:
            negative_pool = list(all_target_items - cold_item_set)
            if len(negative_pool) < num_candidates:
                continue

            sampled_items = random.sample(negative_pool, num_candidates)
            sampled_items += list(cold_item_set)
            sampled_items = list(set(sampled_items))
            random.shuffle(sampled_items)

            user_tensor = torch.tensor([user_id] * len(sampled_items), device=device)
            item_tensor = torch.tensor(sampled_items, device=device)
            link = torch.stack([user_tensor, item_tensor], dim=0)

            scores = model(source_edge_index, target_edge_index, link, is_source=False).view(-1)
            item_score_pairs = list(zip(sampled_items, scores.tolist()))
            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            sorted_items = [item for item, _ in item_score_pairs]

            top_k_items = sorted_items[:top_k]

            cold_hits = [item for item in top_k_items if item in cold_item_set]
            if cold_hits:
                cold_item_hit_count += 1
                for cold_item in cold_hits:
                    rank = top_k_items.index(cold_item) + 1
                    cold_item_ranks.append(rank)

            total_users += 1

    er_ratio = cold_item_hit_count / total_users if total_users > 0 else 0.0
    avg_rank = sum(cold_item_ranks) / len(cold_item_ranks) if cold_item_ranks else -1
    median_rank = sorted(cold_item_ranks)[len(cold_item_ranks) // 2] if cold_item_ranks else -1

    logging.info(f"[ER@{top_k}] Users={total_users}, Cold Item Hits={cold_item_hit_count}, ER Ratio={er_ratio:.4f}")
    logging.info(f"[ER@{top_k}] Cold item avg rank: {avg_rank:.2f}, median rank: {median_rank}")

    return er_ratio


def find_cold_items_by_frequency(data, target_train_edge_index, target_test_edge_index, freq_threshold=0):
    num_users = data.num_users
    num_items = data.num_target_items

    item_indices = target_train_edge_index[1]

    # Debug
    print(f"target_train_edge_index[1] shape: {item_indices.shape}")
    print(f"min val: {item_indices.min()}, max val: {item_indices.max()}")
    print(f"num_users: {num_users}")

    # 扣除 offset
    train_items = item_indices - num_users

    # 過濾負值（若有）
    train_items = train_items[train_items >= 0]

    print(f"Filtered train_items min: {train_items.min()}, max: {train_items.max()}, shape: {train_items.shape}")

    item_freq = torch.bincount(train_items, minlength=num_items)

    cold_items_train = set((item_freq < freq_threshold).nonzero(as_tuple=False).view(-1).tolist())
    test_items = set((target_test_edge_index[1] - num_users).tolist())


    cold_items = [2286]#list(cold_items_train & test_items) 
    logging.info(f"Found {len(cold_items)} cold items by frequency threshold {freq_threshold}")
    
    # # 🔽 統計出現次數
    # print("Cold item ID:", cold_items)
    # print("ASIN:", data.target_id2asin.get(cold_items, "N/A"))

    return cold_items


# def select_high_score_fake_edges(
#     model, data,
#     cold_items, all_users,
#     source_edge_index, target_edge_index,
#     device,
#     strategy="related",
#     user_attack_ratio=1.0,
#     num_fake_edges_per_item=None,
# ):
#     model.eval()
#     fake_edges = []
#     with torch.no_grad():
#         for item in cold_items:
#             if strategy == "random":
#                 # random 從所有用戶隨機選
#                 num_attack_users = max(1, int(len(all_users) * user_attack_ratio))
#                 selected_users = random.sample(all_users, num_attack_users)

#                 for u in selected_users:
#                     fake_edges.append((u, item + data.num_users))

#             else:
#                 # related 及 unrelated 先對「所有用戶」計分
#                 user_tensor = torch.tensor(all_users, device=device)
#                 item_tensor = torch.tensor([item + data.num_users] * len(all_users), device=device)
#                 links = torch.stack([user_tensor, item_tensor], dim=0)

#                 scores = model(source_edge_index, target_edge_index, links, is_source=False).view(-1).tolist()
#                 user_score_pairs = list(zip(all_users, scores))

#                 if strategy == "related":
#                     sorted_pairs = sorted(user_score_pairs, key=lambda x: x[1], reverse=True)
#                 elif strategy == "unrelated":
#                     sorted_pairs = sorted(user_score_pairs, key=lambda x: x[1])
#                 else:
#                     raise ValueError(f"Unsupported strategy: {strategy}")

#                 num_attack_users = max(1, int(len(all_users) * user_attack_ratio))
#                 sorted_pairs = sorted_pairs[:num_attack_users]

#                 fake_edges.extend([(u, item + data.num_users) for u, _ in sorted_pairs])

#     logging.info(f"Selected {len(fake_edges)} fake edges with strategy={strategy}, user_attack_ratio={user_attack_ratio}")
#     return fake_edges
def select_high_score_fake_edges(
    model, data,
    cold_items, all_users,
    source_edge_index, target_edge_index,
    device,
    strategy="related",
    user_attack_ratio=1.0,
    num_fake_edges_per_item=None,
):
    model.eval()
    fake_edges = []

    logging.info(f"=== [DEBUG] Start Selecting Fake Edges ===")
    logging.info(f"Strategy: {strategy}, Total Cold Items: {len(cold_items)}, Total Users: {len(all_users)}")

    with torch.no_grad():
        for idx, item in enumerate(cold_items):
            logging.info(f"\n🎯 Processing Cold Item {idx+1}/{len(cold_items)} → Global ID: {item + data.num_users}")

            if strategy == "random":
                num_attack_users = max(1, int(len(all_users) * user_attack_ratio))
                selected_users = random.sample(all_users, num_attack_users)
                logging.debug(f"[Random] Selected {len(selected_users)} users for cold item {item}")

                for u in selected_users:
                    # fake_edges.append((u, item + data.num_users))
                    fake_edges.append((u, item))

            else:
                user_tensor = torch.tensor(all_users, device=device)
                # item_tensor = torch.tensor([item + data.num_users] * len(all_users), device=device)
                item_tensor = torch.tensor([item] * len(all_users), device=device)
                links = torch.stack([user_tensor, item_tensor], dim=0)

                scores = model(source_edge_index, target_edge_index, links, is_source=False).view(-1).tolist()
                user_score_pairs = list(zip(all_users, scores))

                if strategy == "related":
                    sorted_pairs = sorted(user_score_pairs, key=lambda x: x[1], reverse=True)
                elif strategy == "unrelated":
                    sorted_pairs = sorted(user_score_pairs, key=lambda x: x[1])
                else:
                    raise ValueError(f"Unsupported strategy: {strategy}")

                num_attack_users = max(1, int(len(all_users) * user_attack_ratio))
                selected_pairs = sorted_pairs[:num_attack_users]

                for u, s in selected_pairs:
                    # fake_edges.append((u, item + data.num_users))
                    fake_edges.append((u, item))
                    logging.debug(f"[{strategy}] Select User {u} (Score: {s:.4f}) for Cold Item {item}")

    logging.info(f"\n✅ Selected Total Fake Edges: {len(fake_edges)} with strategy={strategy}, user_attack_ratio={user_attack_ratio}")
    return fake_edges


# def inject_fake_edges(data, target_train_edge_index, fake_edges):
#     device = target_train_edge_index.device
#     if not fake_edges:
#         logging.info("No fake edges to inject.")
#         return target_train_edge_index

#     fake_edge_tensor = torch.tensor(fake_edges, dtype=torch.long).T.to(device)  # shape: [2, num_fake_edges]
#     new_edge_index = torch.cat([target_train_edge_index, fake_edge_tensor], dim=1)
#     logging.info(f"Injected {fake_edge_tensor.shape[1]} fake edges. New train edge count: {new_edge_index.shape[1]}")
#     return new_edge_index

def inject_fake_edges(data, target_train_edge_index, fake_edges):
    device = target_train_edge_index.device
    if not fake_edges:
        logging.info("❌ No fake edges to inject.")
        return target_train_edge_index

    fake_edge_tensor = torch.tensor(fake_edges, dtype=torch.long).T.to(device)  # shape: [2, num_fake_edges]

    logging.info(f"\n=== [DEBUG] Injecting Fake Edges ===")
    logging.info(f"Original edge count: {target_train_edge_index.shape[1]}")
    logging.info(f"Fake edge tensor shape: {fake_edge_tensor.shape}")
    logging.debug(f"Sample fake edges (first 10): {fake_edges[:10]}")

    # 檢查是否有非法 index
    max_user_idx = data.num_users - 1
    max_item_idx = data.num_users + data.num_target_items - 1
    invalid_edges = [(u, i) for u, i in fake_edges if u > max_user_idx or i < data.num_users or i > max_item_idx]
    if invalid_edges:
        logging.warning(f"⚠️ Found invalid fake edges: {invalid_edges[:5]} (showing first 5)")
    else:
        logging.info("✅ All fake edges are within valid user/item index range.")

    new_edge_index = torch.cat([target_train_edge_index, fake_edge_tensor], dim=1)
    logging.info(f"Injected {fake_edge_tensor.shape[1]} fake edges. New train edge count: {new_edge_index.shape[1]}")
    return new_edge_index
def train(model, perceptor, data, args):
    device = args.device
    data = data.to(device)
    model = model.to(device)
    perceptor = perceptor.to(device)

    (
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
        target_test_edge_index,
    ) = link_split(data)
    data.target_test_link = target_test_link

    source_set_size = source_link.shape[1]
    train_set_size = target_train_link.shape[1]
    val_set_size = target_valid_link.shape[1]
    test_set_size = target_test_link.shape[1]

    logging.info(f"Train set size: {train_set_size}")
    logging.info(f"Valid set size: {val_set_size}")
    logging.info(f"Test set size: {test_set_size}")

    target_train_set = Dataset(target_train_link.to("cpu"), target_train_label.to("cpu"))
    target_train_loader = DataLoader(
        target_train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=target_train_set.collate_fn)

    source_batch_size = int(args.batch_size * train_set_size / source_set_size)
    source_train_set = Dataset(source_link.to("cpu"), source_label.to("cpu"))
    source_train_loader = DataLoader(
        source_train_set, batch_size=source_batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=source_train_set.collate_fn)

    target_meta_loader = DataLoader(
        target_train_set, batch_size=args.meta_batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=target_train_set.collate_fn)
    target_meta_iter = iter(target_meta_loader)
    source_meta_batch_size = int(args.meta_batch_size * train_set_size / source_set_size)
    source_meta_loader = DataLoader(
        source_train_set, batch_size=source_meta_batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=source_train_set.collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    perceptor_optimizer = torch.optim.Adam(perceptor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta_optimizer = MetaOptimizer(
        meta_optimizer=perceptor_optimizer,
        hpo_lr=args.hpo_lr,
        truncate_iter=3,
        max_grad_norm=10,
    )

    model_param = [p for name, p in model.named_parameters() if "preds" not in name]
    replace_param = [p for name, p in model.named_parameters() if name.startswith("replace")]
    replace_optimizer = torch.optim.Adam(replace_param, lr=args.lr)
    replace_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        replace_optimizer, T_max=getattr(args, "T_max", args.epochs))

    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=args.model_path,
        trace_func=logging.info,
    )

    criterion = nn.BCELoss(reduction="none")
    iteration = 0

    ##### Step 1: FULL Training WITHOUT fake edges #####
    logging.info(f"=== Step 1: Full training WITHOUT fake edges for {args.epochs} epochs ===")
    for epoch in range(args.epochs):
        model.train()
        for (source_link_batch, source_label_batch), (target_link_batch, target_label_batch) in zip(source_train_loader, target_train_loader):
            source_link_batch = source_link_batch.to(device)
            source_label_batch = source_label_batch.to(device)
            target_link_batch = target_link_batch.to(device)
            target_label_batch = target_label_batch.to(device)
            weight_source = perceptor(source_link_batch[1], source_edge_index, model)

            optimizer.zero_grad()
            source_out = model(source_edge_index, target_train_edge_index, source_link_batch, is_source=True).view(-1)
            target_out = model(source_edge_index, target_train_edge_index, target_link_batch, is_source=False).view(-1)
            source_loss = (criterion(source_out, source_label_batch).reshape(-1,1) * weight_source).sum()
            target_loss = criterion(target_out, target_label_batch).mean()
            loss = source_loss + target_loss if args.use_meta else target_loss
            loss.backward()
            optimizer.step()

            iteration += 1

            if args.use_source and args.use_meta and iteration % args.meta_interval == 0:
                logging.info(f"Meta optimization at iteration {iteration}")
                try:
                    meta_optimizeation(
                        target_meta_loader, replace_optimizer, model, args,
                        criterion, replace_scheduler,
                        source_edge_index, target_train_edge_index)
                except Exception as e:
                    logging.warning(f"Meta optimization failed: {e}")

        train_auc = evaluate("Train", model, source_edge_index, target_train_edge_index, target_train_link, target_train_label)
        val_auc = evaluate("Valid", model, source_edge_index, target_train_edge_index, target_valid_link, target_valid_label)
        logging.info(f"[Epoch {epoch}] Train AUC: {train_auc:.4f}, Valid AUC: {val_auc:.4f}")
        wandb.log({"loss": loss.item(), "train_auc": train_auc, "val_auc": val_auc}, step=epoch)
        lr_scheduler.step()

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

    ##### Step 2: 找冷門商品，並注入假邊 #####
    cold_items = find_cold_items_by_frequency(data, target_train_edge_index, target_test_edge_index, freq_threshold=1)
    logging.info(f"Cold items found: {cold_items}")

    target_cold_item = cold_items[0] if cold_items else None

    if target_cold_item is not None:
        logging.info(f"Injecting fake edges for cold item id: {target_cold_item}")
        all_users = list(range(data.num_users))
        fake_edges = select_high_score_fake_edges(
            model, data, [target_cold_item], all_users,
            source_edge_index, target_train_edge_index,
            device,
            strategy=args.attack_strategy,
            user_attack_ratio=args.user_attack_ratio,
            num_fake_edges_per_item=getattr(args, "max_fake_edges_per_item", None)
        )
        logging.info(f"Number of fake edges generated: {len(fake_edges)}")

        target_train_edge_index = inject_fake_edges(data, target_train_edge_index, fake_edges)

        fake_labels = torch.ones(len(fake_edges), dtype=target_train_label.dtype).to(target_train_label.device)
        target_train_label = torch.cat([target_train_label, fake_labels], dim=0)

        # 更新 Dataset 和 DataLoader
        target_train_set = Dataset(target_train_edge_index.to("cpu"), target_train_label.to("cpu"))
        target_train_loader = DataLoader(
            target_train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=target_train_set.collate_fn)

    else:
        logging.info("No cold item found for fake edge injection.")

    ##### Step 3: Retrain with fake edges #####
    logging.info(f"=== Step 3: Retraining WITH injected fake edges for {args.epochs} epochs ===")
    iteration = 0
    model.train()  # 確保模型在train模式
    # （可選）若要從頭訓練，請加模型重置，這裡維持接著訓練方式保持原參數
    for epoch in range(args.epochs):
        for (source_link_batch, source_label_batch), (target_link_batch, target_label_batch) in zip(source_train_loader, target_train_loader):
            source_link_batch = source_link_batch.to(device)
            source_label_batch = source_label_batch.to(device)
            target_link_batch = target_link_batch.to(device)
            target_label_batch = target_label_batch.to(device)
            weight_source = perceptor(source_link_batch[1], source_edge_index, model)

            optimizer.zero_grad()
            source_out = model(source_edge_index, target_train_edge_index, source_link_batch, is_source=True).view(-1)
            target_out = model(source_edge_index, target_train_edge_index, target_link_batch, is_source=False).view(-1)
            source_loss = (criterion(source_out, source_label_batch).reshape(-1,1) * weight_source).sum()
            target_loss = criterion(target_out, target_label_batch).mean()
            loss = source_loss + target_loss if args.use_meta else target_loss
            loss.backward()
            optimizer.step()

            iteration += 1

            if args.use_source and args.use_meta and iteration % args.meta_interval == 0:
                logging.info(f"Meta optimization at iteration {iteration} (retrain)")
                try:
                    meta_optimizeation(
                        target_meta_loader, replace_optimizer, model, args,
                        criterion, replace_scheduler,
                        source_edge_index, target_train_edge_index)
                except Exception as e:
                    logging.warning(f"Meta optimization failed: {e}")

        train_auc = evaluate("Train_with_fake", model, source_edge_index, target_train_edge_index, target_train_link, target_train_label)
        val_auc = evaluate("Valid_with_fake", model, source_edge_index, target_train_edge_index, target_valid_link, target_valid_label)
        logging.info(f"[Epoch {epoch} retrain] Train AUC: {train_auc:.4f}, Valid AUC: {val_auc:.4f}")
        wandb.log({"loss": loss.item(), "train_auc_with_fake": train_auc, "val_auc_with_fake": val_auc}, step=epoch)

    ##### Step 4: 測試評估 #####
    model = load_model(args).to(device)  # Load best模型或最後模型

    logging.info("=== Step 4: Evaluating Hit Ratio and ER ===")
    evaluate_hit_ratio(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,
        top_k=args.top_k,
        num_candidates=99,
        device=device,
    )

    if target_cold_item is not None:
        evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_train_edge_index,
            cold_item_set={target_cold_item},
            top_k=args.top_k,
            num_candidates=99,
            device=device,
        )
    else:
        logging.info("No cold item found for ER evaluation.")

    test_auc = evaluate("Test", model, source_edge_index, target_train_edge_index, target_test_link, target_test_label)
    logging.info(f"Test AUC: {test_auc:.4f}")
    wandb.log({"Test AUC": test_auc})


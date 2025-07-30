import logging
import os
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset as BaseDataset
from torch_geometric.data import Data, Dataset, download_url

from utils import get_df

# Using Amazon 5-core: https://jmcauley.ucsd.edu/data/amazon/
root_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"

category_file_names = {
    "Book": "reviews_Books_5.json.gz",
    "Electronic": "reviews_Electronics_5.json.gz",
    "Movie": "reviews_Movies_and_TV_5.json.gz",
    "CD": "reviews_CDs_and_Vinyl_5.json.gz",
    "Clothing": "reviews_Clothing_Shoes_and_Jewelry_5.json.gz",
    "Kitchen": "reviews_Home_and_Kitchen_5.json.gz",
    "Kindle": "reviews_Kindle_Store_5.json.gz",
    "Sports": "reviews_Sports_and_Outdoors_5.json.gz",
    "Phone": "reviews_Cell_Phones_and_Accessories_5.json.gz",
    "Health": "reviews_Health_and_Personal_Care_5.json.gz",
    "Toy": "reviews_Toys_and_Games_5.json.gz",
    "Game": "reviews_Video_Games_5.json.gz",
    "Tool": "reviews_Tools_and_Home_Improvement_5.json.gz",
    "Beauty": "reviews_Beauty_5.json.gz",
    "App": "reviews_Apps_for_Android_5.json.gz",
    "Office": "reviews_Office_Products_5.json.gz",
    "Pet": "reviews_Pet_Supplies_5.json.gz",
    "Automotive": "reviews_Automotive_5.json.gz",
    "Grocery": "reviews_Grocery_and_Gourmet_Food_5.json.gz",
    "Patio": "reviews_Patio_Lawn_and_Garden_5.json.gz",
    "Baby": "reviews_Baby_5.json.gz",
    "Music": "reviews_Digital_Music_5.json.gz",
    "Instrument": "reviews_Musical_Instruments_5.json.gz",
    "Video": "reviews_Amazon_Instant_Video_5.json.gz",
}


class CrossDomain(Dataset):
    def __init__(
        self,
        # root,
        root="data/",
        categories=["Music", "Instrument"],
        target="Music",
        use_source=True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.root = root
        self.categories = categories
        self.target = categories[-1]
        self.use_source = use_source
        self.name = categories[0]
        for category in categories[1:]:
            self.name += "_" + category
        self.name += "_for_" + target

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> str:
        return [category_file_names[category] for category in self.categories]

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self) -> str:
        return ["data.pt"]

    def download(self):
        for category in self.categories:
            download_url(root_url + category_file_names[category], self.raw_dir)

    def process(self):
        logging.info("Processing...")
        df_list = []
        for category in self.categories:
            path = os.path.join(self.raw_dir, category_file_names[category])
            df = get_df(path)[[
                'reviewerID', 'asin', 'overall', 'unixReviewTime'
            ]]
            df.columns = ["user", "item", "rating", "timestamp"]
            df["is_target"] = category == self.target
            df_list.append(df)

        logging.info(f"cates: {self.categories}")

        user_sets = [set(df["user"].unique()) for df in df_list]
        common_users = set.intersection(*user_sets)
        logging.info(f"Common users: {len(common_users)}")

        df = pd.concat(df_list)
        df = df[df["user"].isin(common_users)]
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["click"] = (df["rating"] > 3).astype(float)
        df.drop(columns=["rating"], inplace=True)
        logging.info(df)

        target_df = df[df["is_target"] == 1]
        training_time = dict()
        for user, group in target_df.groupby("user"):
            training_time[user] = group["timestamp"].values[-2]

        remove_index = []
        source_df = df[df["is_target"] == 0]
        for user, group in source_df.groupby("user"):
            indexs = group[group["timestamp"] >= training_time[user]].index
            remove_index.extend(indexs)
        df.drop(remove_index, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)

        user_index = {}
        for idx, user in enumerate(df["user"].unique()):
            user_index[user] = idx
        df["user"] = df["user"].apply(lambda x: user_index[x])

        source_df = df[df["is_target"] == 0].copy()
        source_df.reset_index(drop=True, inplace=True)
        source_item_index = {}
        for idx, item in enumerate(source_df["item"].unique()):
            source_item_index[item] = idx
        source_df["item"] = source_df["item"].apply(lambda x: source_item_index[x])
        source_df["item"] += len(user_index)

        source_label = torch.tensor(source_df["click"].values, dtype=torch.float)
        source_link = torch.tensor(
            source_df[["user", "item"]].values, dtype=torch.long
        ).t()

        target_df = df[df["is_target"] == 1].copy()
        target_df.reset_index(drop=True, inplace=True)
        target_item_index = {}
        for idx, item in enumerate(target_df["item"].unique()):
            target_item_index[item] = idx
        target_df["item"] = target_df["item"].apply(lambda x: target_item_index[x])
        target_df["item"] += len(user_index)

        target_label = torch.tensor(target_df["click"].values, dtype=torch.float)
        target_link = torch.tensor(
            target_df[["user", "item"]].values, dtype=torch.long
        ).t()

        # print statistics
        logging.info(
            f'source item num: {len(source_df["item"].unique())}, record num: {source_df.shape[0]}'
        )
        logging.info(
            f'source sparsity: {source_df.shape[0] / len(source_df["item"].unique()) / len(source_df["user"].unique()) * 100:3f}%'
        )

        logging.info(
            f'target item num: {len(target_df["item"].unique())}, record num: {target_df.shape[0]}'
        )
        logging.info(
            f'target sparsity: {target_df.shape[0] / len(target_df["item"].unique()) / len(target_df["user"].unique()) * 100:3f}%'
        )

        train_mask = torch.zeros(target_df.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(target_df.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(target_df.shape[0], dtype=torch.bool)

        for user, group in target_df.groupby("user"):
            test_mask[group.index[-1]] = 1
            val_mask[group.index[-2]] = 1
            train_mask[group.index[:-2]] = 1

        record_tot = dict()
        for user, group in df.groupby("user"):
            tot = group.shape[0]
            if tot not in record_tot:
                record_tot[tot] = 0
            record_tot[tot] += 1
        logging.info(record_tot)

        # 轉成 PyG 所需的 edge_index 格式
        target_edge_index = torch.tensor(
            target_df[["user", "item"]].values, dtype=torch.long
        ).t()

        # 製作 train/valid/test 的 edge index
        target_train_edge_index = target_edge_index[:, train_mask]
        target_valid_edge_index = target_edge_index[:, val_mask]
        target_test_edge_index  = target_edge_index[:, test_mask]

        # 建立 PyG 的 Data 物件
        data = Data(
            source_label=source_label,
            source_link=source_link,
            target_label=target_label,
            target_link=target_link,
            split_mask={"train": train_mask, "valid": val_mask, "test": test_mask},
            num_users=len(user_index),
            num_source_items=len(source_item_index),
            num_target_items=len(target_item_index),

            # ✅ 補上 overlap user list
            raw_overlap_users=torch.tensor(
                [user_index[u] for u in common_users],
                dtype=torch.long
            ),

            # ✅ 補上 target domain 的分割後 edge_index
            target_train_edge_index=target_train_edge_index,
            target_valid_edge_index=target_valid_edge_index,
            target_test_edge_index=target_test_edge_index,
        )


        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(data_list, self.processed_paths[0])

    def len(self):
        data_list = torch.load(self.processed_paths[0])
        return len(data_list)

    def get(self, idx):
        data_list = torch.load(self.processed_paths[0])
        data = data_list[idx]
        return data


class Dataset(BaseDataset):
    def __init__(self, link, label):
        self.link = link
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        link = self.link[:, index]
        label = self.label[index]
        return link, label

    def collate_fn(self, batch):
        link, label = zip(*batch)
        link = torch.stack(link, dim=1)
        label = torch.tensor(label, dtype=torch.float)
        return link, label

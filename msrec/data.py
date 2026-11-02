"""读取 MovieLens（ml-latest-small 格式）数据，并划分训练集和测试集。

data/raw/ 下需要的文件：
    ratings.csv : userId, movieId, rating, timestamp
    movies.csv  : movieId, title, genres（"Action|Comedy" 格式）
    links.csv   : movieId, imdbId, tmdbId
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Dataset:
    ratings: pd.DataFrame          # userId, movieId, rating
    movies: pd.DataFrame           # movieId, title, genres
    links: pd.DataFrame            # movieId, tmdbId
    user_index: dict[int, int]     # userId  -> 矩阵的行号
    item_index: dict[int, int]     # movieId -> 矩阵的列号

    @property
    def n_users(self) -> int:
        return len(self.user_index)

    @property
    def n_items(self) -> int:
        return len(self.item_index)


def load_movielens(raw_dir: str | Path) -> Dataset:
    raw_dir = Path(raw_dir)
    ratings = pd.read_csv(raw_dir / "ratings.csv", usecols=["userId", "movieId", "rating"])
    movies = pd.read_csv(raw_dir / "movies.csv")
    links_path = raw_dir / "links.csv"
    if links_path.exists():
        links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"])
    else:
        links = pd.DataFrame({"movieId": movies["movieId"], "tmdbId": np.nan})

    # 没有任何评分的电影不放进矩阵
    item_ids = sorted(ratings["movieId"].unique())
    user_ids = sorted(ratings["userId"].unique())
    movies = movies[movies["movieId"].isin(item_ids)].reset_index(drop=True)

    return Dataset(
        ratings=ratings,
        movies=movies,
        links=links,
        user_index={u: i for i, u in enumerate(user_ids)},
        item_index={m: j for j, m in enumerate(item_ids)},
    )


def split_per_user(ratings: pd.DataFrame, test_ratio: float, seed: int,
                   min_train: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按用户随机把评分分成 train 和 test。

    评分很少的用户也至少在 train 里保留 min_train 条。
    """
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, grp in ratings.groupby("userId"):
        idx = rng.permutation(len(grp))
        n_test = int(round(len(grp) * test_ratio))
        n_test = min(n_test, max(len(grp) - min_train, 0))
        test_parts.append(grp.iloc[idx[:n_test]])
        train_parts.append(grp.iloc[idx[n_test:]])
    return pd.concat(train_parts), pd.concat(test_parts)


def sparsify(train: pd.DataFrame, keep_ratio: float, seed: int,
             min_keep: int = 2) -> pd.DataFrame:
    """只保留 train 中 keep_ratio 比例的评分，人为制造数据稀疏的情况。"""
    if keep_ratio >= 1.0:
        return train
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in train.groupby("userId"):
        n_keep = max(min_keep, int(round(len(grp) * keep_ratio)))
        n_keep = min(n_keep, len(grp))
        parts.append(grp.iloc[rng.permutation(len(grp))[:n_keep]])
    return pd.concat(parts)


def to_matrix(df: pd.DataFrame, ds: Dataset) -> sparse.csr_matrix:
    """把评分表转换成 用户 × 电影 的稀疏矩阵。"""
    rows = df["userId"].map(ds.user_index).to_numpy()
    cols = df["movieId"].map(ds.item_index).to_numpy()
    return sparse.csr_matrix(
        (df["rating"].to_numpy(dtype=np.float32), (rows, cols)),
        shape=(ds.n_users, ds.n_items),
    )

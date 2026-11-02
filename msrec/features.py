"""把电影（物品）转换成特征向量。

使用两种信息来源：
  * internal：MovieLens 自带的类型信息（movies.csv）
  * tmdb    ：从 TMDB 收集的外部元数据（类型、导演、主演、关键词、年代）

另外可以人为模拟"采集失败导致的数据缺失"，
并比较是否用 internal 的类型信息去补全缺失的电影（fallback）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Dataset

NO_GENRE = "(no genres listed)"


def _split(value) -> list[str]:
    """把 'A|B|C' 格式的字符串拆成列表，空值返回空列表。"""
    if not isinstance(value, str) or not value:
        return []
    return [v for v in value.split("|") if v and v != NO_GENRE]


def internal_tokens(movies: pd.DataFrame) -> dict[int, list[str]]:
    return {int(m): [f"g:{g}" for g in _split(gs)]
            for m, gs in movies[["movieId", "genres"]].itertuples(index=False)}


def tmdb_tokens(meta: pd.DataFrame) -> dict[int, list[str]]:
    """把每部电影的信息变成带前缀的标签，例如 'd:John Lasseter'（导演）、'k:toy'（关键词）。
    加前缀是为了区分同名但含义不同的标签。"""
    out: dict[int, list[str]] = {}
    for row in meta.itertuples(index=False):
        toks = [f"g:{g}" for g in _split(row.genres)]
        toks += [f"d:{d}" for d in _split(row.directors)]
        toks += [f"c:{c}" for c in _split(row.cast)]
        toks += [f"k:{k}" for k in _split(row.keywords)]
        if pd.notna(row.release_year):
            toks.append(f"y:{int(row.release_year) // 10 * 10}s")
        if toks:
            out[int(row.movieId)] = toks
    return out


@dataclass
class ItemFeatures:
    X: sparse.csr_matrix      # 电影数 × 特征数（每行做了 L2 归一化）
    coverage: float           # 至少有一个特征的电影所占比例
    source: str


def build_item_features(ds: Dataset, source: str, tmdb_meta: pd.DataFrame | None = None,
                        missing_ratio: float = 0.0, fallback: bool = False,
                        seed: int = 0) -> ItemFeatures:
    """source：'internal' 或 'tmdb'。

    missing_ratio：在有外部元数据的电影中，随机删掉这个比例，模拟采集失败。
    fallback     ：没有外部元数据的电影，改用 internal 的类型信息补上。
    """
    internal = internal_tokens(ds.movies)
    if source == "internal":
        tokens = internal
    elif source == "tmdb":
        if tmdb_meta is None:
            raise ValueError("tmdb_meta is required")
        tokens = tmdb_tokens(tmdb_meta)
        if missing_ratio > 0:
            rng = np.random.default_rng(seed)
            ids = np.array(sorted(tokens))
            drop = rng.choice(ids, size=int(round(len(ids) * missing_ratio)), replace=False)
            for m in drop:
                tokens.pop(int(m), None)
        if fallback:
            for m, toks in internal.items():
                tokens.setdefault(m, toks)
    else:
        raise ValueError(f"unknown source: {source}")

    item_ids = sorted(ds.item_index, key=ds.item_index.get)
    docs = [tokens.get(m, []) for m in item_ids]
    # TF-IDF：常见的标签（比如 Drama）权重低，少见的标签（比如某位导演）权重高
    vec = TfidfVectorizer(analyzer=lambda toks: toks, min_df=2, norm="l2")
    try:
        X = vec.fit_transform(docs).tocsr()
    except ValueError:                       # 所有电影都没有特征（词表为空）的情况
        X = sparse.csr_matrix((len(item_ids), 1))
    coverage = float(np.mean(X.getnnz(axis=1) > 0))
    return ItemFeatures(X=X, coverage=coverage, source=source)

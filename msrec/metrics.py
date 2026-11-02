"""Top-N 推荐的评估指标：Precision@N、Recall@N、NDCG@N、覆盖率。

正确答案（relevant）= 测试集中评分 >= threshold 的电影。
训练集中已经评过分的电影不参与推荐。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def top_n(scores: np.ndarray, R_train: sparse.csr_matrix, n: int) -> np.ndarray:
    """返回每个用户得分最高的 n 部电影（按得分从高到低排列）。"""
    scores = scores.astype(np.float64, copy=True)
    rows, cols = R_train.nonzero()
    scores[rows, cols] = -np.inf                    # 排除已经看过的电影
    idx = np.argpartition(-scores, n, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def evaluate(scores: np.ndarray, R_train: sparse.csr_matrix, R_test: sparse.csr_matrix,
             n: int = 10, threshold: float = 4.0) -> dict[str, float]:
    recs = top_n(scores, R_train, n)
    R_test = R_test.tocsr()
    # NDCG 的折扣：排名越靠后，命中的价值越低
    discounts = 1.0 / np.log2(np.arange(2, n + 2))

    prec, rec, ndcg = [], [], []
    for u in range(R_test.shape[0]):
        start, end = R_test.indptr[u], R_test.indptr[u + 1]
        items = R_test.indices[start:end][R_test.data[start:end] >= threshold]
        if len(items) == 0:
            continue                       # 没有正确答案的用户不参与评估
        relevant = set(items.tolist())
        hits = np.array([i in relevant for i in recs[u]], dtype=float)
        prec.append(hits.sum() / n)
        rec.append(hits.sum() / len(relevant))
        ideal = discounts[: min(len(relevant), n)].sum()
        ndcg.append((hits * discounts).sum() / ideal)

    return {
        f"precision@{n}": float(np.mean(prec)),
        f"recall@{n}": float(np.mean(rec)),
        f"ndcg@{n}": float(np.mean(ndcg)),
        "coverage": float(len(np.unique(recs)) / scores.shape[1]),
        "n_eval_users": len(prec),
    }

"""推荐模型。

* Popularity     ：按评分人数从多到少推荐（作为下限参考）
* UserCF         ：基于用户的协同过滤（把毕业论文的方法重新实现，作为基线）
* ContentBased   ：用用户的口味画像和电影特征的相似度推荐
* HybridUserCF   ：用户相似度 = 评分相似度和口味画像相似度的加权和。
                   评分很稀疏时，用外部信息补充用户之间的相似度。

    sim(u, v) = λ * cos(r_u, r_v) + (1 - λ) * cos(p_u, p_v)

所有模型先调用 fit(R, X)，再用 scores() 返回 用户数 × 电影数 的打分矩阵。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def _cosine(M: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """计算所有行两两之间的余弦相似度，对角线（自己和自己）设为 0。"""
    Mn = normalize(M, norm="l2", axis=1)
    S = Mn @ Mn.T
    S = S.toarray() if sparse.issparse(S) else np.asarray(S)
    np.fill_diagonal(S, 0.0)
    return S


def _keep_top_k(S: np.ndarray, k: int) -> np.ndarray:
    """每一行只保留相似度最高的 k 个邻居，其余设为 0。"""
    if k >= S.shape[1]:
        return S
    out = np.zeros_like(S)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    out[rows, idx] = S[rows, idx]
    return out


def user_profiles(R: sparse.csr_matrix, X: sparse.csr_matrix) -> sparse.csr_matrix:
    """用户的口味画像：把打分高于自己平均分的电影特征加上，低于平均分的减掉。"""
    R = R.tocsr().astype(np.float32)
    counts = np.diff(R.indptr)
    means = np.asarray(R.sum(axis=1)).ravel() / np.maximum(counts, 1)
    W = R.copy()
    W.data = W.data - np.repeat(means, counts)      # 减去用户平均分（不喜欢的电影权重为负）
    return normalize(W @ X, norm="l2", axis=1)


class Popularity:
    name = "Popularity"

    def fit(self, R, X=None):
        self.pop = np.asarray((R > 0).sum(axis=0)).ravel().astype(np.float32)
        self.n_users = R.shape[0]
        return self

    def scores(self):
        return np.tile(self.pop, (self.n_users, 1))


class UserCF:
    name = "UserCF"

    def __init__(self, k: int = 50):
        self.k = k

    def similarity(self, R, X):
        return _cosine(R)

    def fit(self, R, X=None):
        self.R = R.tocsr()
        self.S = _keep_top_k(self.similarity(self.R, X), self.k)
        return self

    def scores(self):
        # 打分 = 邻居的相似度 × 邻居的评分，再求和（即 S @ R）
        return np.asarray(self.R.T.dot(self.S.T).T)


class HybridUserCF(UserCF):
    def __init__(self, k: int = 50, lam: float = 0.5):
        super().__init__(k)
        self.lam = lam
        self.name = f"HybridUserCF(λ={lam})"

    def similarity(self, R, X):
        if X is None:
            raise ValueError("HybridUserCF needs item features X")
        s_rating = _cosine(R)                       # 评分相似度
        s_profile = _cosine(user_profiles(R, X))    # 口味画像相似度
        return self.lam * s_rating + (1 - self.lam) * s_profile


class ContentBased:
    name = "ContentBased"

    def fit(self, R, X):
        self.P = user_profiles(R, X)
        self.X = X
        return self

    def scores(self):
        S = self.P @ self.X.T
        return S.toarray() if sparse.issparse(S) else np.asarray(S)

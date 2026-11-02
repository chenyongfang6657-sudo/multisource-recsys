"""推荐模型和评估指标的测试（使用可以手算验证的小例子）。"""

import numpy as np
from scipy import sparse

from msrec.metrics import evaluate, top_n
from msrec.models import HybridUserCF, Popularity, UserCF


def small():
    # 用户 0 和用户 1 口味相同，用户 2 口味不同
    R = sparse.csr_matrix(np.array([
        [5, 4, 0, 0, 1],
        [5, 0, 5, 0, 1],
        [0, 0, 1, 5, 5],
    ], dtype=np.float32))
    X = sparse.csr_matrix(np.array([
        [1, 0], [1, 0], [1, 0], [0, 1], [0, 1],
    ], dtype=np.float32))
    return R, X


def test_usercf_recommends_neighbor_item():
    R, _ = small()
    scores = UserCF(k=1).fit(R).scores()
    # 给用户 0 推荐的应该是相似用户 1 打了高分的电影 2
    assert top_n(scores, R, 1)[0, 0] == 2


def test_hybrid_lambda_one_equals_usercf():
    # λ = 1 时只用评分相似度，结果应该和 UserCF 完全相同
    R, X = small()
    a = UserCF(k=2).fit(R).scores()
    b = HybridUserCF(k=2, lam=1.0).fit(R, X).scores()
    assert np.allclose(a, b)


def test_top_n_excludes_train_items():
    # 已经看过的电影不应该再被推荐
    R, _ = small()
    recs = top_n(Popularity().fit(R).scores(), R, 2)
    for u in range(3):
        assert not set(recs[u]) & set(R[u].indices)


def test_metrics_perfect_ranking():
    R_train = sparse.csr_matrix((1, 4))
    R_test = sparse.csr_matrix(np.array([[5, 0, 0, 0]], dtype=np.float32))
    scores = np.array([[0.9, 0.1, 0.2, 0.3]])
    m = evaluate(scores, R_train, R_test, n=2)
    assert m["ndcg@2"] == 1.0 and m["recall@2"] == 1.0 and m["precision@2"] == 0.5

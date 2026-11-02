"""运行全部实验。

实验 A（稀疏程度）：把训练集评分减少到 100% / 50% / 20% / 10%，比较各方法的精度
实验 B（数据缺失）：在稀疏条件下（默认 20%）人为删除一部分外部元数据，
                    比较补全（fallback）与不补全时精度的变化
实验 C（λ 敏感度）：改变 HybridUserCF 的权重 λ，观察精度的变化

用法：
    python -m msrec.experiment --raw data/raw --tmdb data/external/tmdb_metadata.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                # 不弹出窗口，直接把图保存成文件
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import load_movielens, sparsify, split_per_user, to_matrix  # noqa: E402
from .features import build_item_features  # noqa: E402
from .metrics import evaluate  # noqa: E402
from .models import ContentBased, HybridUserCF, Popularity, UserCF  # noqa: E402

log = logging.getLogger("experiment")

# 每个方法固定颜色和标记，保证所有图里同一个方法的样子都一样
STYLE = {
    "Popularity": ("#8a8a85", "x"),
    "UserCF": ("#2a78d6", "o"),
    "Hybrid (internal genres)": ("#1baf7a", "s"),
    "Hybrid (TMDB)": ("#eb6834", "D"),
    "Content (TMDB)": ("#eda100", "^"),
    "Hybrid (TMDB, no fallback)": ("#eb6834", "D"),
    "Hybrid (TMDB + fallback)": ("#2a78d6", "o"),
}


def run_one(model, R_train, R_test, X, n):
    """训练一个模型并评估，同时记录耗时。"""
    t0 = time.perf_counter()
    model.fit(R_train, X)
    res = evaluate(model.scores(), R_train, R_test, n=n)
    res["seconds"] = round(time.perf_counter() - t0, 2)
    return res


def summarize(df: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    """把多个随机种子的结果汇总成 平均值 ± 标准差。"""
    g = df.groupby(keys)[metric].agg(["mean", "std"]).reset_index()
    g["text"] = g.apply(lambda r: f"{r['mean']:.4f} ± {0 if np.isnan(r['std']) else r['std']:.4f}",
                        axis=1)
    return g


def plot_lines(summary: pd.DataFrame, x: str, out: Path, title: str, xlabel: str,
               metric_label: str, invert_x: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for method, grp in summary.groupby("method", sort=False):
        grp = grp.sort_values(x)
        color, marker = STYLE.get(method, ("#555555", "o"))
        ax.errorbar(grp[x], grp["mean"], yerr=grp["std"].fillna(0), label=method,
                    color=color, marker=marker, markersize=6, linewidth=2, capsize=3)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_label)
    if invert_x:
        ax.invert_xaxis()                # 从左到右表示数据越来越稀疏
    ax.grid(True, color="#e5e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Experiments on integrating external data into UserCF")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--tmdb", type=Path, default=Path("data/external/tmdb_metadata.csv"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--keep", type=float, nargs="+", default=[1.0, 0.5, 0.2, 0.1],
                   help="fractions of training ratings to keep")
    p.add_argument("--missing", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9],
                   help="fractions of TMDB metadata to remove artificially")
    p.add_argument("--missing-keep", type=float, default=0.2,
                   help="fraction of training ratings used in experiments B and C")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--lam", type=float, default=0.5, help="λ used in experiments A and B")
    p.add_argument("--k", type=int, default=50, help="number of neighbor users")
    p.add_argument("--n", type=int, default=10, help="length of recommendation list")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    ds = load_movielens(args.raw)
    log.info("users=%d items=%d ratings=%d", ds.n_users, ds.n_items, len(ds.ratings))

    tmdb = pd.read_csv(args.tmdb) if args.tmdb.exists() else None
    if tmdb is None:
        log.warning("%s not found, skipping methods that use TMDB", args.tmdb)

    metric = f"ndcg@{args.n}"
    rows_a, rows_b, rows_c = [], [], []
    tmdb_coverage = None

    for seed in range(args.seeds):
        # 每个随机种子重新划分一次训练集和测试集
        train_full, test = split_per_user(ds.ratings, args.test_ratio, seed)
        R_test = to_matrix(test, ds)
        X_int = build_item_features(ds, "internal").X
        X_tmdb = None
        if tmdb is not None:
            f_tmdb = build_item_features(ds, "tmdb", tmdb)
            X_tmdb, tmdb_coverage = f_tmdb.X, f_tmdb.coverage

        # ---- 实验 A：稀疏程度 ---------------------------------------------
        for keep in args.keep:
            R_train = to_matrix(sparsify(train_full, keep, seed), ds)
            models = [("Popularity", Popularity(), None),
                      ("UserCF", UserCF(args.k), None),
                      ("Hybrid (internal genres)", HybridUserCF(args.k, args.lam), X_int)]
            if X_tmdb is not None:
                models += [("Hybrid (TMDB)", HybridUserCF(args.k, args.lam), X_tmdb),
                           ("Content (TMDB)", ContentBased(), X_tmdb)]
            for name, model, X in models:
                res = run_one(model, R_train, R_test, X, args.n)
                rows_a.append({"seed": seed, "keep": keep, "method": name, **res})
                log.info("[A] seed=%d keep=%.2f %-26s %s=%.4f", seed, keep, name,
                         metric, res[metric])

        if X_tmdb is None:
            continue
        R_sparse = to_matrix(sparsify(train_full, args.missing_keep, seed), ds)

        # ---- 实验 B：外部元数据缺失 ----------------------------------------
        for miss in args.missing:
            for fb in (False, True):
                feats = build_item_features(ds, "tmdb", tmdb, missing_ratio=miss,
                                            fallback=fb, seed=seed)
                name = "Hybrid (TMDB + fallback)" if fb else "Hybrid (TMDB, no fallback)"
                res = run_one(HybridUserCF(args.k, args.lam), R_sparse, R_test, feats.X, args.n)
                rows_b.append({"seed": seed, "missing": miss, "method": name,
                               "feature_coverage": feats.coverage, **res})
                log.info("[B] seed=%d missing=%.1f %-26s %s=%.4f", seed, miss, name,
                         metric, res[metric])

        # ---- 实验 C：λ 的敏感度 ---------------------------------------------
        for lam in args.lambdas:
            res = run_one(HybridUserCF(args.k, lam), R_sparse, R_test, X_tmdb, args.n)
            rows_c.append({"seed": seed, "lambda": lam, "method": "Hybrid (TMDB)", **res})

    # ---- 保存结果并画图 ----------------------------------------------------
    md = ["# Results", "",
          f"Metric: {metric.upper()} (mean ± std over {args.seeds} seeds)", ""]
    if tmdb_coverage is not None:
        md += [f"Movies with TMDB metadata: {tmdb_coverage:.1%}", ""]

    df_a = pd.DataFrame(rows_a)
    df_a.to_csv(args.out / "exp_a_sparsity.csv", index=False)
    s_a = summarize(df_a, ["keep", "method"], metric)
    plot_lines(s_a, "keep", args.out / "exp_a_sparsity.png", "A. Sparsity vs. accuracy",
               "Fraction of training ratings kept", metric.upper(), invert_x=True)
    md += ["## Experiment A: rating sparsity", "",
           s_a.pivot(index="method", columns="keep", values="text")
              .reindex(columns=sorted(args.keep, reverse=True)).to_markdown(), "",
           "![](exp_a_sparsity.png)", ""]

    if rows_b:
        df_b = pd.DataFrame(rows_b)
        df_b.to_csv(args.out / "exp_b_missing.csv", index=False)
        s_b = summarize(df_b, ["missing", "method"], metric)
        plot_lines(s_b, "missing", args.out / "exp_b_missing.png",
                   f"B. Missing external metadata (keep={args.missing_keep})",
                   "Fraction of TMDB metadata removed", metric.upper())
        md += [f"## Experiment B: missing external metadata "
               f"({args.missing_keep:.0%} of training ratings)", "",
               s_b.pivot(index="method", columns="missing", values="text").to_markdown(), "",
               "![](exp_b_missing.png)", ""]

        df_c = pd.DataFrame(rows_c)
        df_c.to_csv(args.out / "exp_c_lambda.csv", index=False)
        s_c = summarize(df_c, ["lambda", "method"], metric)
        plot_lines(s_c, "lambda", args.out / "exp_c_lambda.png",
                   f"C. Sensitivity to λ (keep={args.missing_keep})",
                   "λ (1.0 = rating similarity only)", metric.upper())
        md += [f"## Experiment C: sensitivity to λ "
               f"({args.missing_keep:.0%} of training ratings)", "",
               s_c.pivot(index="method", columns="lambda", values="text").to_markdown(), "",
               "![](exp_c_lambda.png)", ""]

    (args.out / "results.md").write_text("\n".join(md), encoding="utf-8")
    (args.out / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Results saved to %s", args.out)


if __name__ == "__main__":
    main()

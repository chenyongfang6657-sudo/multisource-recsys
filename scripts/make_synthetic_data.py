"""生成用于检查程序能否运行的模拟数据（文件格式和 MovieLens ml-latest-small 相同）。

这不是真实数据，得到的数字不能当作研究结果。
真实实验请按 README 的步骤准备 MovieLens 和 TMDB 数据。

生成方式：
  * 每部电影有 1〜2 个隐藏的"主题"（代表某种口味）
  * 类型（internal）和主题的关系较弱，导演和关键词（外部数据）和主题的关系较强
  * 每个用户喜欢几个主题，对喜欢主题的电影更容易打高分

用法：
    python scripts/make_synthetic_data.py --out data/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western", "Film-Noir"]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic data in MovieLens format")
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--users", type=int, default=400)
    p.add_argument("--items", type=int, default=2000)
    p.add_argument("--topics", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    T, U, I = args.topics, args.users, args.items
    raw = args.out / "raw"
    ext = args.out / "external"
    raw.mkdir(parents=True, exist_ok=True)
    ext.mkdir(parents=True, exist_ok=True)

    # 每部电影的主题，以及主题对应的类型、导演、关键词
    item_topics = [rng.choice(T, size=rng.integers(1, 3), replace=False) for _ in range(I)]
    topic_genres = [rng.choice(len(GENRES), size=3, replace=False) for _ in range(T)]
    topic_directors = [[f"Director {t}-{d}" for d in range(4)] for t in range(T)]
    topic_keywords = [[f"keyword {t}-{w}" for w in range(10)] for t in range(T)]
    popularity = rng.pareto(1.2, size=I) + 1      # 少数电影非常热门（长尾分布）

    movies, links, meta = [], [], []
    for m in range(I):
        ts = item_topics[m]
        genres = {GENRES[g] for t in ts for g in rng.choice(topic_genres[t], 1)}
        if rng.random() < 0.5:                                   # 给类型加入噪声
            genres.add(GENRES[rng.integers(len(GENRES))])
        movie_id = m + 1
        movies.append({"movieId": movie_id, "title": f"Movie {movie_id} ({1950 + m % 70})",
                       "genres": "|".join(sorted(genres))})
        tmdb_id = 100000 + movie_id if rng.random() > 0.01 else np.nan
        links.append({"movieId": movie_id, "imdbId": 1000000 + movie_id, "tmdbId": tmdb_id})
        if np.isnan(tmdb_id) or rng.random() < 0.03:             # 模拟真实采集中的少量缺失
            continue
        main_t = ts[0]
        meta.append({
            "movieId": movie_id, "tmdbId": int(tmdb_id), "release_year": 1950 + m % 70,
            "genres": "|".join(sorted(genres)),
            "directors": rng.choice(topic_directors[main_t]),
            "cast": "|".join(f"Actor {rng.integers(300)}" for _ in range(5)),
            "keywords": "|".join(sorted({rng.choice(topic_keywords[t]) for t in ts
                                         for _ in range(3)})),
            "fetched_at": "synthetic",
        })

    # 生成用户评分：热门电影更容易被看到，喜欢的主题更容易打高分
    ratings = []
    for u in range(U):
        liked = set(rng.choice(T, size=rng.integers(2, 4), replace=False).tolist())
        n = int(np.clip(rng.lognormal(3.6, 0.8), 20, 600))
        prob = popularity / popularity.sum()
        items = rng.choice(I, size=min(n, I), replace=False, p=prob)
        for m in items:
            match = len(liked & set(item_topics[m].tolist())) > 0
            mu = 4.2 if match else 2.6
            r = float(np.clip(np.round((mu + rng.normal(0, 0.8)) * 2) / 2, 0.5, 5.0))
            ratings.append({"userId": u + 1, "movieId": int(m) + 1, "rating": r,
                            "timestamp": 1_500_000_000 + int(rng.integers(10**7))})

    pd.DataFrame(ratings).to_csv(raw / "ratings.csv", index=False)
    pd.DataFrame(movies).to_csv(raw / "movies.csv", index=False)
    pd.DataFrame(links).to_csv(raw / "links.csv", index=False)
    pd.DataFrame(meta).to_csv(ext / "tmdb_metadata.csv", index=False)
    print(f"ratings={len(ratings)} movies={I} metadata={len(meta)} -> {args.out}")


if __name__ == "__main__":
    main()

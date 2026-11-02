"""通过 TMDB API 收集电影的外部元数据。

特点：
  * 可断点续传：每取一条就追加写入 JSONL 缓存，中断后重新运行会从断点继续
  * 自动重试：429（限流）按 Retry-After 等待；5xx 和网络错误按指数退避重试
  * 质量检查：统计必要字段的缺失情况，输出采集报告（JSON）

用法：
    export TMDB_API_KEY=xxxxx        # 或者 TMDB_READ_TOKEN（v4 的 Read Access Token）
    python -m msrec.collect_tmdb --raw data/raw --out data/external

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

API_BASE = "https://api.themoviedb.org/3"
REQUIRED_FIELDS = ["genres", "directors", "cast", "keywords", "release_year"]

log = logging.getLogger("collect_tmdb")


@dataclass
class CollectorConfig:
    api_key: str | None = None
    read_token: str | None = None
    min_interval: float = 0.1      # 两次请求之间的最小间隔（秒）
    max_retries: int = 5
    backoff_base: float = 1.0      # 指数退避的基准时间（秒）
    timeout: float = 10.0
    top_cast: int = 5              # 只取前几名主演


@dataclass
class FetchResult:
    tmdb_id: int
    status: str                    # ok / not_found / failed
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class TMDBCollector:
    def __init__(self, cfg: CollectorConfig, session: requests.Session | None = None,
                 sleep=time.sleep):
        if not cfg.api_key and not cfg.read_token:
            raise ValueError("Set TMDB_API_KEY or TMDB_READ_TOKEN")
        self.cfg = cfg
        self.session = session or requests.Session()
        self._sleep = sleep            # 测试时可以替换成不等待的函数
        self._last_request = 0.0
        if cfg.read_token:
            self.session.headers["Authorization"] = f"Bearer {cfg.read_token}"

    # ---- HTTP 请求 --------------------------------------------------------
    def _wait_interval(self) -> None:
        """控制请求频率，避免过快触发限流。"""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.cfg.min_interval:
            self._sleep(self.cfg.min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, path: str, params: dict[str, Any]) -> requests.Response | None:
        """发送 GET 请求。遇到限流、服务器错误、网络错误时自动重试，超过次数返回 None。"""
        if self.cfg.api_key:
            params = {**params, "api_key": self.cfg.api_key}
        for attempt in range(self.cfg.max_retries + 1):
            self._wait_interval()
            try:
                resp = self.session.get(f"{API_BASE}{path}", params=params,
                                        timeout=self.cfg.timeout)
            except requests.RequestException as exc:
                wait = self.cfg.backoff_base * 2 ** attempt
                log.warning("Network error on %s (%s), retrying in %.1fs", path, exc, wait)
                self._sleep(wait)
                continue

            if resp.status_code == 429:
                # 被限流：服务器会在 Retry-After 里告诉我们要等多久
                wait = float(resp.headers.get("Retry-After", self.cfg.backoff_base * 2 ** attempt))
                log.warning("Rate limited (429), waiting %.1fs", wait)
                self._sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                # 服务器临时错误：等待时间每次翻倍
                wait = self.cfg.backoff_base * 2 ** attempt
                log.warning("Server error %d, retrying in %.1fs", resp.status_code, wait)
                self._sleep(wait)
                continue
            return resp
        return None

    # ---- 获取并整理数据 ----------------------------------------------------
    def fetch(self, tmdb_id: int) -> FetchResult:
        # append_to_response 可以一次请求同时拿到演职员表和关键词
        resp = self._get(f"/movie/{tmdb_id}",
                         {"append_to_response": "credits,keywords", "language": "en-US"})
        if resp is None:
            return FetchResult(tmdb_id, "failed", error="max retries exceeded")
        if resp.status_code == 404:
            return FetchResult(tmdb_id, "not_found")
        if resp.status_code != 200:
            return FetchResult(tmdb_id, "failed", error=f"HTTP {resp.status_code}")
        try:
            return FetchResult(tmdb_id, "ok", data=self.normalize(resp.json()))
        except (ValueError, KeyError, TypeError) as exc:
            # 返回的结构和预期不同时，记录下来继续处理下一条，不让整个程序停下
            return FetchResult(tmdb_id, "failed", error=f"parse error: {exc}")

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        """从 API 返回的原始 JSON 中只取出实验需要的字段。字段缺失时用空值代替。"""
        credits = raw.get("credits") or {}
        release = raw.get("release_date") or ""
        return {
            "title": raw.get("title"),
            "release_year": int(release[:4]) if release[:4].isdigit() else None,
            "genres": [g["name"] for g in raw.get("genres") or []],
            # 导演要从工作人员列表中筛选 job 为 Director 的人
            "directors": [c["name"] for c in credits.get("crew") or []
                          if c.get("job") == "Director"],
            "cast": [c["name"] for c in (credits.get("cast") or [])[: self.cfg.top_cast]],
            "keywords": [k["name"] for k in (raw.get("keywords") or {}).get("keywords") or []],
            "runtime": raw.get("runtime"),
            "vote_average": raw.get("vote_average"),
        }


# ---- 缓存 ------------------------------------------------------------------
def load_cache(path: Path) -> dict[int, dict[str, Any]]:
    """读取 JSONL 缓存。ok 和 not_found 的不再重新获取，failed 的会重试。"""
    done: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["status"] in ("ok", "not_found"):
                done[int(rec["tmdb_id"])] = rec
    return done


def append_cache(path: Path, res: FetchResult) -> dict[str, Any]:
    """每取到一条就立刻追加写入，这样中途停止也不会丢失已获取的数据。"""
    rec = {
        "tmdb_id": res.tmdb_id,
        "status": res.status,
        "data": res.data,
        "error": res.error,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


# ---- 质量报告 ----------------------------------------------------------------
def build_report(links: pd.DataFrame, records: dict[int, dict[str, Any]],
                 failed: Iterable[int]) -> dict[str, Any]:
    """统计采集的覆盖率和各字段的缺失率。"""
    n_items = len(links)
    n_no_id = int(links["tmdbId"].isna().sum())
    ok = [r for r in records.values() if r["status"] == "ok"]
    missing = {f: 0 for f in REQUIRED_FIELDS}
    for r in ok:
        for f in REQUIRED_FIELDS:
            v = r["data"].get(f)
            if v is None or v == []:
                missing[f] += 1
    return {
        "items_total": n_items,
        "items_without_tmdb_id": n_no_id,
        "fetched_ok": len(ok),
        "not_found": sum(r["status"] == "not_found" for r in records.values()),
        "failed": len(set(failed)),
        "coverage": round(len(ok) / n_items, 4) if n_items else 0.0,
        "field_missing_rate": {f: round(c / len(ok), 4) if ok else None
                               for f, c in missing.items()},
    }


def to_table(links: pd.DataFrame, records: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """整理成以 movieId 为单位的表格。列表类型的字段用 '|' 连接成字符串保存。"""
    rows = []
    for movie_id, tmdb_id in links[["movieId", "tmdbId"]].itertuples(index=False):
        if pd.isna(tmdb_id) or int(tmdb_id) not in records:
            continue
        rec = records[int(tmdb_id)]
        if rec["status"] != "ok":
            continue
        d = rec["data"]
        rows.append({
            "movieId": int(movie_id),
            "tmdbId": int(tmdb_id),
            "release_year": d.get("release_year"),
            "genres": "|".join(d.get("genres") or []),
            "directors": "|".join(d.get("directors") or []),
            "cast": "|".join(d.get("cast") or []),
            "keywords": "|".join(d.get("keywords") or []),
            "fetched_at": rec.get("fetched_at"),
        })
    return pd.DataFrame(rows)


def run(raw_dir: Path, out_dir: Path, cfg: CollectorConfig, limit: int | None = None,
        collector: TMDBCollector | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "tmdb_cache.jsonl"
    links = pd.read_csv(raw_dir / "links.csv", usecols=["movieId", "tmdbId"])
    records = load_cache(cache_path)
    collector = collector or TMDBCollector(cfg)

    # 只获取缓存里还没有的电影
    todo = [int(t) for t in links["tmdbId"].dropna().unique() if int(t) not in records]
    if limit is not None:
        todo = todo[:limit]
    log.info("Already cached: %d / to fetch now: %d", len(records), len(todo))

    failed: list[int] = []
    for k, tmdb_id in enumerate(todo, 1):
        res = collector.fetch(tmdb_id)
        rec = append_cache(cache_path, res)
        if res.status == "failed":
            failed.append(tmdb_id)
        else:
            records[tmdb_id] = rec
        if k % 200 == 0:
            log.info("Processed %d / %d", k, len(todo))

    table = to_table(links, records)
    table.to_csv(out_dir / "tmdb_metadata.csv", index=False)
    report = build_report(links, records, failed)
    (out_dir / "collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Done: %s", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Collect movie metadata from the TMDB API")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/external"))
    p.add_argument("--limit", type=int, default=None,
                   help="max number of movies to fetch (0 = rebuild outputs from cache only)")
    p.add_argument("--min-interval", type=float, default=0.1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = CollectorConfig(api_key=os.getenv("TMDB_API_KEY"),
                          read_token=os.getenv("TMDB_READ_TOKEN"),
                          min_interval=args.min_interval)
    run(args.raw, args.out, cfg, limit=args.limit)


if __name__ == "__main__":
    main()

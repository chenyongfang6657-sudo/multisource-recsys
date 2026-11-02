"""TMDB 采集功能的测试（不访问真实的 API）。"""

import json

import pandas as pd

from msrec.collect_tmdb import CollectorConfig, TMDBCollector, load_cache, run

SAMPLE = {
    "title": "Toy Story", "release_date": "1995-10-30", "runtime": 81, "vote_average": 8.0,
    "genres": [{"id": 16, "name": "Animation"}, {"id": 35, "name": "Comedy"}],
    "credits": {"crew": [{"job": "Director", "name": "John Lasseter"},
                         {"job": "Producer", "name": "Someone"}],
                "cast": [{"name": f"Actor {i}"} for i in range(8)]},
    "keywords": {"keywords": [{"name": "toy"}, {"name": "friendship"}]},
}


class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeSession:
    """模拟的 HTTP 会话：每次调用按顺序返回事先准备好的响应。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


def make(responses):
    cfg = CollectorConfig(api_key="dummy", min_interval=0, backoff_base=0, max_retries=3)
    return TMDBCollector(cfg, session=FakeSession(responses), sleep=lambda s: None)


def test_normalize_extracts_fields():
    res = make([FakeResp(200, SAMPLE)]).fetch(862)
    assert res.status == "ok"
    assert res.data["directors"] == ["John Lasseter"]
    assert res.data["release_year"] == 1995
    assert len(res.data["cast"]) == 5
    assert res.data["keywords"] == ["toy", "friendship"]


def test_retry_on_429_and_500():
    col = make([FakeResp(429, headers={"Retry-After": "0"}), FakeResp(500), FakeResp(200, SAMPLE)])
    res = col.fetch(862)
    assert res.status == "ok"
    assert col.session.calls == 3


def test_gives_up_after_max_retries():
    res = make([FakeResp(500)] * 4).fetch(1)
    assert res.status == "failed"


def test_not_found():
    assert make([FakeResp(404)]).fetch(1).status == "not_found"


def test_missing_fields_do_not_crash():
    res = make([FakeResp(200, {"title": "No info"})]).fetch(1)
    assert res.status == "ok"
    assert res.data["genres"] == [] and res.data["release_year"] is None


def test_run_is_resumable(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"movieId": [1, 2, 3], "tmdbId": [10, 20, None]}).to_csv(raw / "links.csv",
                                                                          index=False)
    out = tmp_path / "ext"
    cfg = CollectorConfig(api_key="dummy", min_interval=0)

    # 第 1 次运行：10 成功，20 失败
    col1 = make([FakeResp(200, SAMPLE), FakeResp(500), FakeResp(500), FakeResp(500),
                 FakeResp(500)])
    report1 = run(raw, out, cfg, collector=col1)
    assert report1["fetched_ok"] == 1 and report1["failed"] == 1
    assert report1["items_without_tmdb_id"] == 1

    # 第 2 次运行：只重新获取失败的 20
    col2 = make([FakeResp(200, SAMPLE)])
    report2 = run(raw, out, cfg, collector=col2)
    assert col2.session.calls == 1
    assert report2["fetched_ok"] == 2
    assert len(load_cache(out / "tmdb_cache.jsonl")) == 2
    table = pd.read_csv(out / "tmdb_metadata.csv")
    assert sorted(table["movieId"]) == [1, 2]
    assert json.loads((out / "collection_report.json").read_text())["coverage"] == round(2 / 3, 4)

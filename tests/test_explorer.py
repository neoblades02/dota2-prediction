import re

import pytest
from sqlalchemy import create_engine, func, select

import dota_predictor.ingest.explorer as ex
from dota_predictor.settings import get_settings
from dota_predictor.storage.models import pro_match_index
from dota_predictor.storage.repository import init_db

TIMEOUT = {"err": "Error: Query read timeout"}


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


class _Explorer:
    """OpenDota explorer stand-in that times out on big queries, like the real one under load."""

    def __init__(self, n_matches=4500, max_limit=1000, max_ids=50):
        self.n, self.max_limit, self.max_ids = n_matches, max_limit, max_ids
        self.limits: list[int] = []

    def get(self, path, params=None):
        sql = params["sql"]
        if "FROM matches" in sql:
            limit = int(re.search(r"LIMIT (\d+)", sql).group(1))
            cursor = int(re.search(r"m\.match_id > (\d+)", sql).group(1))
            self.limits.append(limit)
            if limit > self.max_limit:
                return _Resp(400, TIMEOUT)
            ids = range(cursor + 1, min(cursor + limit, self.n) + 1)
            return _Resp(200, {"rows": [self._match(i) for i in ids], "err": None})
        ids = [int(x) for x in re.search(r"IN \(([^)]*)\)", sql).group(1).split(",")]
        if len(ids) > self.max_ids:
            return _Resp(400, TIMEOUT)
        rows = [{"match_id": i, "account_id": k, "hero_id": k + 1, "player_slot": k} for i in ids for k in range(10)]
        return _Resp(200, {"rows": rows, "err": None})

    def close(self):
        pass

    @staticmethod
    def _match(i):
        return {
            "match_id": i,
            "start_time": 1700000000 + i,
            "duration": 2000,
            "radiant_win": True,
            "radiant_team_id": 1,
            "dire_team_id": 2,
            "radiant_score": 30,
            "dire_score": 20,
            "leagueid": 15000,
            "league_name": "L",
            "tier": "premium",
            "series_id": None,
            "series_type": 0,
        }


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "raw_store_dir", tmp_path / "raw")
    eng = create_engine(f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
    init_db(eng)
    return eng


def test_index_page_shrinks_after_a_query_timeout_and_completes(engine, monkeypatch):
    fake = _Explorer(n_matches=4500, max_limit=1000)
    monkeypatch.setattr(ex, "RateLimitedClient", lambda **kw: fake)
    res = ex.sync_explorer_match_index(engine)
    with engine.connect() as c:
        assert c.execute(select(func.count()).select_from(pro_match_index)).scalar() == 4500
    assert res["indexed"] == 4500
    assert fake.limits[:2] == [2000, 1000] and set(fake.limits[2:]) == {1000}


def test_timeout_below_the_smallest_page_is_raised(engine, monkeypatch):
    monkeypatch.setattr(ex, "RateLimitedClient", lambda **kw: _Explorer(max_limit=10))
    with pytest.raises(ex.ExplorerTimeout, match="Query read timeout"):
        ex.sync_explorer_match_index(engine)


def test_player_rows_split_the_id_list_on_timeout():
    rows = ex._player_rows(_Explorer(max_ids=50), list(range(1, 121)))
    assert sorted({r["match_id"] for r in rows}) == list(range(1, 121))
    assert len(rows) == 1200


def test_rejections_carry_opendota_message():
    class Rejecting:
        def get(self, path, params=None):
            return _Resp(400, {"error": "Invalid API key format"})

    with pytest.raises(ex.ExplorerError, match="Invalid API key format") as info:
        ex._query(Rejecting(), "SELECT 1")
    assert not isinstance(info.value, ex.ExplorerTimeout)

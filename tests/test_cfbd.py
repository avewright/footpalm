import json
from pathlib import Path

from datetime import datetime
from zoneinfo import ZoneInfo

from footpalm.cfbd import (
    cached,
    cache_age_s,
    cfbd_dir,
    dataset_path,
    load_dotenv,
    load_usage,
    note_calls,
    should_refresh_cfbd,
)
from footpalm.fetch import pbp_path

ET = ZoneInfo("America/New_York")


def test_cfbd_does_not_share_pbp_paths(tmp_path: Path):
    cfbd = dataset_path(tmp_path, "games", 2024)
    pbp = pbp_path(tmp_path, 2024)
    assert cfbd != pbp
    assert cfbd.is_relative_to(cfbd_dir(tmp_path))
    assert pbp.parent == tmp_path / "data" / "raw"
    assert cfbd.parent == tmp_path / "data" / "raw" / "cfbd" / "games"


def test_load_dotenv_does_not_override_real_env(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("CFBD_API_KEY=from-file\nOTHER=1\n")
    monkeypatch.setenv("CFBD_API_KEY", "from-env")
    monkeypatch.delenv("OTHER", raising=False)
    load_dotenv(tmp_path)
    import os

    assert os.environ["CFBD_API_KEY"] == "from-env"
    assert os.environ["OTHER"] == "1"


def test_cached_requires_a_real_file(tmp_path: Path):
    dest = tmp_path / "games.json"
    assert not cached(dest)
    dest.write_text("[]\n")
    assert not cached(dest)
    dest.write_text(json.dumps([{"id": i} for i in range(10)]) + "\n")
    assert cached(dest)


def test_usage_rolls_to_a_new_month(tmp_path: Path):
    dest = tmp_path / "data" / "processed" / "cfbd-usage.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps({"month": "2026-07", "calls": 400, "cap": 1000}) + "\n")
    row = note_calls(tmp_path, 2, now=datetime(2026, 8, 30, tzinfo=ET))
    assert row["month"] == "2026-08"
    assert row["calls"] == 2
    assert row["prior"]["calls"] == 400


def test_should_refresh_cfbd_throttles():
    tue = datetime(2026, 9, 1, 12, tzinfo=ET)
    sat = datetime(2026, 9, 5, 12, tzinfo=ET)
    assert should_refresh_cfbd(tue, None, 0) is True
    assert should_refresh_cfbd(tue, 5 * 3600, 0) is False
    assert should_refresh_cfbd(tue, 7 * 3600, 0) is True
    assert should_refresh_cfbd(sat, 2 * 3600, 0) is False
    assert should_refresh_cfbd(sat, 4 * 3600, 0) is True
    assert should_refresh_cfbd(sat, 20 * 3600, 950) is False


def test_cache_age_missing(tmp_path: Path):
    assert cache_age_s(tmp_path, "games", 2026) is None
    assert load_usage(tmp_path)["calls"] == 0

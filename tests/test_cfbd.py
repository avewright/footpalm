import json
from pathlib import Path

from footpalm.cfbd import cached, cfbd_dir, dataset_path, load_dotenv
from footpalm.fetch import pbp_path


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

from pathlib import Path

from footpalm.ask import Session

from tests.test_ask import _warehouse


def test_submit_locks_market_line(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    out = session.submit_picks(items=[{"query": "Indiana", "kind": "ats"}])
    assert len(out["matched"]) == 1
    assert out["unmatched"] == []
    pick = next(iter(session.book["picks"].values()))
    assert pick["kind"] == "ats"
    assert pick["side"] == "home"
    assert pick["line"] == -40.5


def test_submit_away_line_and_list(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    session.submit_picks(items=[{"query": "North Texas", "kind": "ats", "line": 40.5}])
    pick = next(iter(session.book["picks"].values()))
    assert pick["side"] == "away"
    assert pick["line"] == 40.5
    listed = session.list_picks()
    assert listed["n"] == 1
    assert listed["picks"][0]["result"] == "pending"
    assert listed["picks"][0]["yours"].startswith("North Texas")


def test_submit_unmatched_name(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    out = session.submit_picks(items=[{"query": "Tatooine", "kind": "ats"}])
    assert out["matched"] == []
    assert out["unmatched"]


def test_submit_parses_line_off_query(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    out = session.submit_picks(items=[{"query": "North Texas +3"}])
    assert out["unmatched"] == []
    pick = next(iter(session.book["picks"].values()))
    assert pick["side"] == "away"
    assert pick["line"] == 3


def test_submit_merges_into_existing_book(tmp_path: Path):
    session = Session(
        _warehouse(tmp_path),
        2025,
        picks={"name": "Mine", "picks": {"keep": {"kind": "ml", "side": "home", "home_win_prob": None}}},
    )
    session.submit_picks(items=[{"query": "Nebraska", "kind": "ml"}])
    assert "keep" in session.book["picks"]
    assert any(p.get("kind") == "ml" and p.get("side") == "home" for p in session.book["picks"].values())
    dumped = session.dump_book()
    assert dumped["name"] == "Mine"
    assert dumped["matched"] >= 2

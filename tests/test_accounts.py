import json
from pathlib import Path

from footpalm.accounts import Accounts, Store, game_key, score_picks


def _root(tmp_path: Path, games: list[dict] | None = None) -> Path:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    payload = {
        "season": 2026,
        "games": games
        or [
            {
                "season": 2026,
                "week": 1,
                "game_id": 1,
                "home": "Ohio State",
                "away": "Akron",
                "fbs_fbs": True,
                "pred_home": 52.0,
                "pred_away": 10.0,
                "pred_margin": 42.0,
                "home_win_prob": 0.99,
                "actual_home": 48.0,
                "actual_away": 7.0,
                "actual_margin": 41.0,
                "home_won": 1,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": 2,
                "home": "Ohio State",
                "away": "Michigan",
                "fbs_fbs": True,
                "pred_home": 28.0,
                "pred_away": 21.0,
                "pred_margin": 7.0,
                "home_win_prob": 0.7,
                "actual_home": 17.0,
                "actual_away": 24.0,
                "actual_margin": -7.0,
                "home_won": 0,
            },
        ],
    }
    (processed / "predictions-2026.json").write_text(json.dumps(payload) + "\n")
    return tmp_path


def test_claim_reuses_name_and_active_model(tmp_path: Path):
    store = Store(_root(tmp_path))
    user, token = store.claim("jackson")
    again, _ = store.claim("Jackson")
    assert again["id"] == user["id"]
    assert store.user_from_token(token)["username"] == "jackson"

    row = store.create_model(
        user,
        "Home brew",
        2026,
        "slate.csv",
        {
            "1": {"pred_away": 9, "pred_home": 45, "home_win_prob": 0.95},
            "2": {"pred_away": 24, "pred_home": 20, "home_win_prob": 0.4},
        },
    )
    active = store.active_model(user, 2026)
    assert active is not None
    assert active["id"] == row["id"]
    assert active["picks"]["1"]["pred_home"] == 45
    cards = store.catalog(2026, user)
    kinds = {c["kind"] for c in cards}
    assert "you" in kinds
    assert "admin" in kinds
    yours = next(c for c in cards if c["kind"] == "you")
    assert yours["score"]["n"] == 2
    assert yours["score"]["suW"] == 2


def test_second_model_becomes_active_and_can_switch(tmp_path: Path):
    store = Store(_root(tmp_path))
    user, _ = store.claim("jordan")
    first = store.create_model(user, "A", 2026, "a.csv", {"1": {"pred_away": 10, "pred_home": 40, "home_win_prob": 0.9}})
    second = store.create_model(user, "B", 2026, "b.csv", {"1": {"pred_away": 12, "pred_home": 38, "home_win_prob": 0.8}})
    assert store.active_model(user, 2026)["id"] == second["id"]
    store.patch_model(user, first["id"], {"active": True, "name": "A renamed"})
    active = store.active_model(user, 2026)
    assert active["id"] == first["id"]
    assert active["name"] == "A renamed"
    assert len(store.mine(user, 2026)) == 2


def test_unpublished_hidden_from_others(tmp_path: Path):
    store = Store(_root(tmp_path))
    alice, _ = store.claim("alice")
    bob, _ = store.claim("bob")
    store.create_model(
        alice,
        "Secret",
        2026,
        "s.csv",
        {"1": {"pred_away": 3, "pred_home": 40, "home_win_prob": 0.9}},
        published=False,
    )
    public = store.create_model(
        alice,
        "Public",
        2026,
        "p.csv",
        {"1": {"pred_away": 4, "pred_home": 41, "home_win_prob": 0.91}},
        published=True,
        active=False,
    )
    names = {c["name"] for c in store.catalog(2026, bob)}
    assert "Secret" not in names
    assert "Public" in names
    assert public["id"] in {c["id"] for c in store.catalog(2026, bob)}


def test_score_picks_matches_winner_count():
    games = [
        {"game_id": 1, "actual_home": 20, "actual_away": 10, "actual_margin": 10, "home_won": 1},
        {"game_id": 2, "actual_home": 10, "actual_away": 20, "actual_margin": -10, "home_won": 0},
    ]
    picks = {
        "1": {"pred_away": 14, "pred_home": 24, "home_win_prob": 0.8},
        "2": {"pred_away": 17, "pred_home": 21, "home_win_prob": 0.6},
    }
    card = score_picks(games, picks)
    assert card["n"] == 2
    assert card["suW"] == 1
    assert card["suL"] == 1
    assert game_key(games[0]) == "1"


def test_accounts_http_login_roundtrip(tmp_path: Path):
    accounts = Accounts(_root(tmp_path))

    class Fake:
        path = "/api/auth/login"
        headers = {}

    result = accounts.dispatch(Fake(), "POST", {"username": "casey"})
    assert result["code"] == 200
    assert result["payload"]["user"]["username"] == "casey"
    assert "footpalm=" in result["cookie"]
    again = accounts.dispatch(Fake(), "POST", {"username": "casey"})
    assert again["payload"]["user"]["id"] == result["payload"]["user"]["id"]

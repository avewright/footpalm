from footpalm.score import et_day, grade, scorecard, stamp_payload, summarize


def test_et_day_uses_new_york():
    assert et_day("2026-08-29T23:00:00.000Z") == "2026-08-29"
    assert et_day("2026-08-30T03:00:00.000Z") == "2026-08-29"


def test_grade_tcu_miss_and_memphis_ats():
    tcu = grade(
        {
            "home_win_prob": 0.8183,
            "pred_margin": 15.64,
            "spread": -8.5,
            "actual_home": 10,
            "actual_away": 15,
            "start": "2026-08-29T16:00:00.000Z",
        }
    )
    assert tcu is not None
    assert tcu["su"] is False
    assert tcu["ats"] is False
    assert tcu["take_home"] is True

    memphis = grade(
        {
            "home_win_prob": 0.5195,
            "pred_margin": 0.81,
            "spread": -4.5,
            "actual_home": 21,
            "actual_away": 27,
            "start": "2026-08-29T23:00:00.000Z",
        }
    )
    assert memphis is not None
    assert memphis["su"] is False
    assert memphis["ats"] is True
    assert memphis["take_home"] is False


def test_summarize_saturday_shape():
    games = [
        {"home_win_prob": 0.82, "pred_margin": 16, "spread": -8.5, "actual_home": 10, "actual_away": 15, "start": "2026-08-29T16:00:00Z"},
        {"home_win_prob": 0.94, "pred_margin": 29, "spread": -38.5, "actual_home": 42, "actual_away": 26, "start": "2026-08-29T19:00:00Z"},
    ]
    card = summarize(games)
    assert card["n"] == 2
    assert card["su_w"] == 1
    assert card["ats_w"] == 1


def test_stamp_does_not_move_probabilities():
    payload = {
        "season": 2026,
        "games": [
            {
                "game_id": 1,
                "home": "A",
                "away": "B",
                "home_win_prob": 0.7,
                "pred_margin": 7.0,
                "spread": -3.5,
                "actual_home": None,
                "completed": False,
            }
        ],
    }
    slate = [
        {
            "game_id": 1,
            "start": "2026-08-29T16:00:00.000Z",
            "actual_home": 24.0,
            "actual_away": 17.0,
            "actual_margin": 7.0,
            "home_won": 1,
            "spread": -3.0,
            "completed": True,
        }
    ]
    out, stamped = stamp_payload(payload, slate)
    game = out["games"][0]
    assert stamped == 1
    assert game["home_win_prob"] == 0.7
    assert game["pred_margin"] == 7.0
    assert game["spread"] == -3.5
    assert game["actual_home"] == 24.0
    assert game["completed"] is True
    card = scorecard(out["games"])
    assert card["last_day"] == "2026-08-29"
    assert card["to_date"]["su_w"] == 1

from footpalm.draft import list_picks


def test_list_picks_filters_year_round_pos_nfl():
    picks = [
        {
            "name": "A",
            "year": 2026,
            "round": 1,
            "overall": 1,
            "college": "Indiana",
            "nfl": "Las Vegas",
            "position": "Quarterback",
        },
        {
            "name": "B",
            "year": 2026,
            "round": 5,
            "overall": 161,
            "college": "Nebraska",
            "nfl": "Kansas City",
            "position": "Running Back",
        },
        {
            "name": "C",
            "year": 2025,
            "round": 1,
            "overall": 1,
            "college": "Miami",
            "nfl": "Tennessee",
            "position": "Quarterback",
        },
    ]
    assert [p["name"] for p in list_picks(picks, 2026)] == ["A", "B"]
    assert [p["name"] for p in list_picks(picks, 2026, rnd=1)] == ["A"]
    assert [p["name"] for p in list_picks(picks, 2026, position="qb")] == ["A"]
    assert [p["name"] for p in list_picks(picks, 2026, nfl="Kansas City")] == ["B"]
    assert [p["name"] for p in list_picks(picks, 2025)] == ["C"]

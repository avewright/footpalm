from footpalm.nil import estimate_football


def test_texas_is_football_not_all_sports():
    value, quality = estimate_football("Texas", "SEC", 46_485_000)
    assert quality == "blended"
    assert 47_000_000 <= value <= 50_000_000
    assert value < 60_000_000


def test_lsu_is_top_spender_without_the_50m_rumor():
    value, _ = estimate_football("LSU", "SEC", 40_966_000)
    assert 44_000_000 <= value < 50_000_000


def test_miami_clears_the_40m_club():
    value, _ = estimate_football("Miami", "ACC", 38_629_000)
    assert value >= 40_000_000


def test_indiana_title_does_not_make_them_texas():
    value, quality = estimate_football("Indiana", "B1G", 27_253_000)
    assert quality == "blended"
    assert 29_000_000 <= value <= 32_000_000


def test_memphis_uses_the_g6_ceiling():
    value, quality = estimate_football("Memphis", "AAC", None)
    assert quality == "reported"
    assert value == 10_500_000


def test_mac_default_is_not_2023_money():
    value, quality = estimate_football("Bowling Green", "MAC", None)
    assert quality == "modeled"
    assert value == 2_500_000

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from footpalm.cron import project_due

ET = ZoneInfo("America/New_York")


def _pred(root: Path, generated_at: str) -> None:
    dest = root / "web" / "public" / "data"
    dest.mkdir(parents=True)
    dest.joinpath("predictions-2026.json").write_text(
        f'{{"generated_at": "{generated_at}", "games": []}}\n'
    )


def test_project_due_tuesday_after_ten(tmp_path: Path):
    _pred(tmp_path, "2026-08-25T14:00:00+00:00")
    morning = datetime(2026, 9, 1, 9, tzinfo=ET)
    afternoon = datetime(2026, 9, 1, 11, tzinfo=ET)
    wednesday = datetime(2026, 9, 2, 12, tzinfo=ET)
    thursday = datetime(2026, 9, 3, 12, tzinfo=ET)
    assert project_due(tmp_path, 2026, morning) is False
    assert project_due(tmp_path, 2026, afternoon) is True
    assert project_due(tmp_path, 2026, wednesday) is True
    assert project_due(tmp_path, 2026, thursday) is False


def test_project_due_false_if_already_ran_this_week(tmp_path: Path):
    _pred(tmp_path, "2026-09-01T14:05:00+00:00")
    assert project_due(tmp_path, 2026, datetime(2026, 9, 1, 16, tzinfo=ET)) is False
    assert project_due(tmp_path, 2026, datetime(2026, 9, 2, 12, tzinfo=ET)) is False

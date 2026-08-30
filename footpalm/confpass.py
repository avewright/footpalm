from __future__ import annotations

from footpalm.form import ALL_NAMES, CONF_ALL, CONF_NAMES
from footpalm.losopass import score_sets

SETS = (
    ("extras", "X_full", ALL_NAMES),
    ("extras+conf", "X_conf", CONF_ALL),
)


def run() -> dict:
    return score_sets(
        SETS,
        candidate="extras+conf",
        feature_names=CONF_NAMES,
        heading="Conference (diagnostic, not live)",
        blurb=(
            "Menu locked from the LOSO permutation before this score. Conference axis only. "
            "Leave-one-season-out on 2014–2025 FBS–FBS. Walk-forward features only."
        ),
        stem="confpass",
        protocol=(
            "Conference menu locked before the score. Fit on all other seasons, score each "
            "2014–2025 FBS–FBS season once. Trees + logistic. Not live TabPFN."
        ),
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()

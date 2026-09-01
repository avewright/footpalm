from __future__ import annotations

from footpalm.form import ALL_NAMES
from footpalm.qb import QB_ALL, QB_NAMES
from footpalm.walkpass import score_sets

SETS = (
    ("extras", "X_full", ALL_NAMES),
    ("extras+qb", "X_qb", QB_ALL),
)


def run() -> dict:
    return score_sets(
        SETS,
        candidate="extras+qb",
        feature_names=QB_NAMES,
        heading="QB (diagnostic, not live)",
        blurb=(
            "Menu locked before this score. QB state only: this-season EPA/play with the "
            "expected starter, change vs last season modal, log1p starts, prior EPA any team. "
            "Garbage-filtered, walk-forward PBP, prior slates only. No spread / NIL / week dummy. "
            "Expected starter = modal passer from most recent completed game, else last season modal. "
            "Expanding-year on 2014–2025 FBS–FBS."
        ),
        stem="qbpass",
        protocol=(
            "QB menu locked before the score. Walk-forward PBP only. Prior slates only. "
            "No spread. No NIL. No week dummy. No Massey/SP+/talent. Do not peek at this game’s passer. "
            "Expanding-year: train season < hold. Score each 2015–2025 FBS–FBS season once. "
            "Trees + logistic. Not live TabPFN."
        ),
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()

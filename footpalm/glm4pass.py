from __future__ import annotations

from footpalm.form import ALL_NAMES, GLM4_ALL, GLM4_NAMES
from footpalm.walkpass import score_sets

SETS = (
    ("extras", "X_full", ALL_NAMES),
    ("extras+glm4", "X_glm4", GLM4_ALL),
)


def run() -> dict:
    return score_sets(
        SETS,
        candidate="extras+glm4",
        feature_names=GLM4_NAMES,
        heading="GLM4 (diagnostic, not live)",
        blurb=(
            "Menu locked before this score. Same ridge Bradley–Terry as LOSO, "
            "but each number is its own column: glm_home, glm_away, glm_diff, glm_sum. "
            "Expanding-year on 2014–2025 FBS–FBS. Walk-forward features only."
        ),
        stem="glm4pass",
        protocol=(
            "GLM4 menu locked before the score. Expanding-year: train season < hold. "
            "Score each 2015–2025 FBS–FBS season once. Trees + logistic. Not live TabPFN."
        ),
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()

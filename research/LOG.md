# Research log

## Shipped 2026-08-29: extras into live TabPFN

Walk-forward rebuild 2014–2025. History is 9947 × 20 (locked Pom + extras). Logistic fallback still uses the locked 10.

| 2025 FBS–FBS n=807 | Brier | log loss |
|---|---|---|
| locked TabPFN (old live) | 0.1926 | — |
| extras walk, raw | 0.1865 | 0.5529 |
| extras walk + T=0.7167 | **0.1817** | **0.5367** |

Δ vs locked: −0.0109 Brier. T refit on 2014–2024 extras walk (was 0.6913 on locked). 2026 preseason projected from 2025 Pom + extras, TabPFN on all 9947 rows (fit uses last 8000), T=0.7167. Do not add craft/signal without a new locked menu.

---

Generated 2026-08-29T21:27:22+00:00. Fit on 2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024 (n=7865), scored on 2025 (n=807).

Promoted: **temperature**.

Promoted temperature. 2025 Brier 0.1865 → 0.1817.

| id | params | 2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024 Brier | 2025 Brier | 2025 logloss | Δ Brier | pass |
|---|---|---|---|---|---|---|
| identity | — | 0.1864 | 0.1865 | 0.5529 | — | baseline |
| clip_05 | — | 0.1864 | 0.1865 | 0.5529 | 0.0 | False |
| clip_10 | — | 0.1865 | 0.1866 | 0.5535 | 0.0001 | False |
| temperature | {'T': 0.7167} | 0.1837 | 0.1817 | 0.5367 | -0.0048 | True |
| sigma | {'s': 10.392} | 0.1837 | 0.1817 | 0.5367 | -0.0048 | True |
| shrink | {'w': 0.0, 'rate': 0.5797} | 0.1864 | 0.1865 | 0.5529 | 0.0 | False |
| platt | {'a': -0.0223, 'b': 1.4031} | 0.1837 | 0.1818 | 0.5369 | -0.0047 | True |
| oracle_temperature | {'T': 0.6003} | 0.1847 | 0.1814 | 0.5339 | -0.0051 | False |
| oracle_platt | {'a': 0.0441, 'b': 1.6505} | 0.1848 | 0.1814 | 0.5337 | -0.0051 | False |

2025 expected Brier if calibrated: 0.214.
Promotion rule: holdout Brier must fall by at least 0.002 and log loss must not rise.
oracle_* rows are a ceiling. They fit on the holdout year. Do not ship them.
No NIL. No market line.

## Trees (diagnostic)

| family | locked Brier | set Brier | Δ Brier | pass |
|---|---|---|---|---|
| lightgbm / extras | 0.1909 | 0.1845 | -0.0064 | True |
| lightgbm / signal | 0.1909 | 0.1836 | -0.0073 | True |
| xgboost / extras | 0.1924 | 0.1834 | -0.009 | True |
| xgboost / signal | 0.1924 | 0.1851 | -0.0073 | True |

- lightgbm: 2025 Brier 0.1909. perm pom_diff +0.0519, adjd_diff +0.0035, adjo_diff +0.0023
- lightgbm-full: 2025 Brier 0.1845. perm pom_diff +0.0247, elo_diff +0.0063, avg_margin_diff +0.0055
- lightgbm-signal: 2025 Brier 0.1836. perm pom_diff +0.0255, elo_diff +0.0061, form4_margin_diff +0.0009
- xgboost: 2025 Brier 0.1924. perm pom_diff +0.0609, adjd_diff +0.0013, away_games +0.0005
- xgboost-full: 2025 Brier 0.1834. perm pom_diff +0.0228, elo_diff +0.0056, avg_margin_diff +0.0056
- xgboost-signal: 2025 Brier 0.1851. perm pom_diff +0.0215, elo_diff +0.0056, sos_diff +0.0015

Trees are not the live model.


## LOSO (diagnostic, not live)

Menu locked from ten 2025/2026 March Madness writeups before this score. Leave-one-season-out on 2014–2025 FBS–FBS. Walk-forward features only.

| family | extras | extras+loso | Δ | pass |
|---|---|---|---|---|
| logistic | 0.1835 | 0.1817 | -0.0018 | False |
| lightgbm | 0.1854 | 0.184 | -0.0014 | False |
| xgboost | 0.1843 | 0.1831 | -0.0012 | False |

Pooled permutation on extras+loso (2025 fold):

| feature | LightGBM | XGBoost |
|---|---|---|
| glm_quality_diff | +0.0004 | +0.0007 |
| glm_sum | +0.0001 | -0.0008 |
| elo_ratio | +0.0018 | +0.0010 |
| log_margin_diff | +0.0003 | +0.0003 |
| conf_pom_diff | +0.0026 | +0.0036 |
| tier_win_diff | -0.0001 | -0.0000 |
| form10_win_diff | +0.0004 | -0.0000 |
| close_x_margin | -0.0001 | -0.0001 |
| late_win_diff | +0.0001 | +0.0000 |
| yoy_margin_diff | -0.0001 | -0.0001 |

| model | pooled Brier | pooled logloss | mean season Brier |
|---|---|---|---|
| logistic/extras | 0.1835 | 0.544 | 0.1836 |
| lightgbm/extras | 0.1854 | 0.5492 | 0.1857 |
| xgboost/extras | 0.1843 | 0.5455 | 0.1844 |
| logistic/extras+loso | 0.1817 | 0.5376 | 0.1817 |
| lightgbm/extras+loso | 0.184 | 0.5439 | 0.1842 |
| xgboost/extras+loso | 0.1831 | 0.541 | 0.1832 |

would_promote=False live_promoted=False. Do not carve a subset after seeing LOSO.

## Conference (diagnostic, not live)

Menu locked from the LOSO permutation before this score. Conference axis only. Leave-one-season-out on 2014–2025 FBS–FBS. Walk-forward features only.

| family | extras | extras+conf | Δ | pass |
|---|---|---|---|---|
| logistic | 0.1835 | 0.1822 | -0.0013 | False |
| lightgbm | 0.1854 | 0.1836 | -0.0018 | False |
| xgboost | 0.1843 | 0.1826 | -0.0017 | False |

Permutation on extras+conf (2025 fold):

| feature | LightGBM | XGBoost |
|---|---|---|
| conf_pom_diff | +0.0054 | +0.0063 |
| conf_elo_diff | -0.0007 | -0.0002 |
| p4_diff | -0.0004 | -0.0004 |
| same_conf | -0.0001 | -0.0000 |
| conf_win_diff | +0.0001 | +0.0002 |
| conf_margin_diff | +0.0003 | +0.0012 |
| ooc_win_diff | +0.0001 | +0.0000 |
| ooc_margin_diff | +0.0009 | +0.0006 |

| model | pooled Brier | pooled logloss | mean season Brier |
|---|---|---|---|
| logistic/extras | 0.1835 | 0.544 | 0.1836 |
| lightgbm/extras | 0.1854 | 0.5492 | 0.1857 |
| xgboost/extras | 0.1843 | 0.5455 | 0.1844 |
| logistic/extras+conf | 0.1822 | 0.5391 | 0.1823 |
| lightgbm/extras+conf | 0.1836 | 0.5427 | 0.1838 |
| xgboost/extras+conf | 0.1826 | 0.54 | 0.1827 |

would_promote=False live_promoted=False. Do not carve a subset after seeing LOSO.

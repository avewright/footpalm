# FootPalm research protocol

Goal: lower Brier without overfitting.

## Rules

1. Features are walk-forward (prior slates only). Fits are too: never train on a later season or a later week of the hold year.
2. **Screen** on expanding-year (`train = season < Y`, score Y, 2015–2025, 2014 is train-only). Logistic + locked trees.
3. **Confirm** live TabPFN only with slate walk-forward (`train = season < Y or (season == Y and slate < this)`).
4. **Audit** peek-LOSO (`season != Y`) is an upper bound. If it beats walk-forward, the gap is leakage. Never promote from it.
5. Do not add NIL, spreads, or week dummies to the live model.
6. Do not grid-search a kitchen sink and pick the winner after seeing any hold year.
7. Candidates are listed below before the run. That is the whole menu.
8. Promote only if pooled Brier drops by at least **0.002**, pooled log loss does not rise, and the drop is not one year (median season Δ < 0, or better in at least **8** scored years). Never promote from a slice (early / mid / late / 2025-only).
9. If two candidates pass, keep the one with fewer parameters.
10. **2026 is live.** Project it from 2025 Pom + TabPFN trained on 2014–2025. Do not fit calibration on 2026. Refresh the slate weekly.

## Why more seasons

One year of calibration can look perfect and still miss a year shift. Expanding-year scores every season after 2014. A 2025-only number is a slice, not the promote.

## Why Brier, not ATS

ATS is a holdout. Tuning it is how you overfit a market. Brier and log loss are proper scoring rules on our own probabilities.

## Candidates

| id | params | what it does |
|---|---|---|
| identity | 0 | raw TabPFN-3 / logistic p |
| clip_05 | 0 | lock p into [0.05, 0.95] |
| clip_10 | 0 | lock p into [0.10, 0.90] |
| temperature | 1 | `p' = σ(logit(p) / T)`, T fit on train log loss |
| sigma | 1 | `p' = σ(margin / s)`, s fit on train log loss |
| shrink | 1 | `p' = (1-w)p + w * train home-win rate` |
| platt | 2 | `p' = σ(a + b · logit(p))` |

Temperature / sigma / Platt are calibration. They do not change who is favored unless a probability crosses 0.5, which Platt can do. We still report accuracy so we can see if that happened.

## Diagnostics, not candidates

`oracle_*` rows fit on 2025 and score on 2025. They are a ceiling. They cannot be promoted.

LightGBM and XGBoost are fit on the locked 10-feature vector, 2014–2024 train / 2025 holdout. They exist to rank features (gain + permutation). They are not the live model and cannot be promoted from this pass. Hyperparameters are locked (depth 4, 200 trees, lr 0.05). Do not grid-search them on 2025.

## Extra features (diagnostic)

Menu locked from March Madness winning writeups (2024 8th, 2026 2nd/3rd): all extras are home−away diffs, built walk-forward from prior slates only. 538-style MOV Elo (K=20, HFA=55, 75% revert to 1500). Last-4 form, season win% / margin, close-game (≤8) win%, SOS as mean opponent Pom, rest as slate gap, luck as win% minus expected from margins, quality-win% vs opponents with Pom > 0.

Fit the same locked trees on locked-10 vs locked-10+extras. Score 2025 FBS–FBS once. Promote extras into the live TabPFN vector only if 2025 Brier drops by at least **0.002** and log loss does not get worse. If they fail, leave `FEATURE_NAMES` at 10.

**Promoted 2026-08-29.** Walk-forward TabPFN on extras was 0.1865 vs locked 0.1926 (Δ −0.0061). Live TabPFN is locked Pom + extras. Logistic fallback stays on the locked 10. Do not add craft/signal on top of this without a new locked menu.

No Massey/SP+/talent ordinals. Those are external ratings and this protocol forbids them.

## Signal expansion (diagnostic)

Second menu. Locked from the extras permutation *before this score*: Elo, season margin, last-4 margin, and SOS had signal. Close-game, luck, rest, and last-4 win% did not. Expand those four axes only. Still walk-forward, prior slates only. No spread. No NIL. No week dummy.

| id | axis | what it is |
|---|---|---|
| elo_diff | Elo | same 538 MOV Elo as extras |
| elo_momentum_diff | Elo | Elo change over the last 4 games |
| avg_margin_diff | margin | season mean scoring margin |
| median_margin_diff | margin | season median scoring margin |
| form2_margin_diff | margin | last-2 mean margin |
| form4_margin_diff | margin | last-4 mean margin |
| form8_margin_diff | margin | last-8 mean margin |
| ewma_margin_diff | margin | EWMA margin, decay 0.45 |
| pf_diff | margin | season points scored / game |
| pa_diff | margin | season points allowed / game |
| form4_pf_diff | margin | last-4 points scored / game |
| form4_pa_diff | margin | last-4 points allowed / game |
| pythag_diff | margin | pf² / (pf² + pa²) |
| venue_margin_diff | margin | home team's home margins − away team's road margins |
| sos_diff | SOS | mean opponent Pom |
| residual_margin_diff | SOS | mean(margin − opponent Pom) |
| form4_residual_diff | SOS | same, last 4 |
| h2h_margin | Elo/margin | last meeting, either season. 0 if none |

Fit locked trees on locked-10 vs locked-10+signal. Same promotion bar. Same rule: do not carve a subset after seeing 2025. Live TabPFN stays on the locked 10 unless this whole vector clears the bar *and* a walk-forward rebuild is run.

## Model grid (diagnostic)

Fit LightGBM, XGBoost, and TabPFN-3 on locked-10, extras, and signal. Same split: 2014–2024 train, 2025 FBS–FBS once. TabPFN uses the locked v3 checkpoint and the last 8000 train rows, same as live.

Also score, listed before the run:

- temperature: one T fit on train log loss, applied to that model's holdout
- blend_extras: mean of the three extras probabilities. No extra fit.

Walk-forward TabPFN on extras for 2025 slates (fit only on earlier rows) is the live-shaped number. Batch TabPFN is the tree-shaped number.

Do not carve a feature subset after seeing 2025. Do not promote extras into live TabPFN without that walk-forward score clearing the 0.002 Brier bar.

## Time features (diagnostic)

Not week dummies. Two numbers, known at kickoff, locked before this score:

| id | what |
|---|---|
| year_idx | `season - 2014`. 2014 is 0. Linear calendar, not a year dummy. |
| week52 | `week / 52`. Week 0 is 0. Bowls stay on the CFBD week, not remapped. |

Score extras vs extras+time on 2025 FBS–FBS once. Same trees. Promote onto the board only if Brier drops ≥ 0.002 vs extras and log loss does not rise. Do not add these to live TabPFN from this pass. 2026 year_idx is 12: that is extrapolation past the train max (11).

## Next pass (diagnostic)

Locked before this score. Whole menu. Do not add anything after seeing 2025. Do not promote extras into live TabPFN in the same pass. ATS is still a holdout.

| id | params | what it is |
|---|---|---|
| tabpfn_margin | 0 | Fit TabPFN-3 **regressor** on `y_margin`. Classifier still owns P(win). Us uses the regressor margin instead of `14.5 × logit(p)`. |
| blend_train | 2 | `p = w_l LGBM + w_x XGB + w_t TabPFN` on extras. Weights ≥ 0, sum to 1, fit on **train** log loss only. Baseline is equal-mean extras. |
| thin | 1 col | `1` if `min(home_games, away_games) < 3`, else `0`. Early-season sample-size bit. Not a week dummy. Score extras vs extras+thin on the locked trees. |

Rules for this pass:

- `blend_train` and `thin` promote on the usual Brier ≥ 0.002 drop and no worse log loss, vs their named baseline (equal blend, extras trees).
- `tabpfn_margin` does not change P(win). Promote into **Us / display margin** only if 2025 MAE drops by at least **0.3** points vs the derived margin and Brier does not get worse (it should be identical).
- Fail = stop. Do not try a 12th idea after the score.

## Craft (diagnostic)

Locked before this score. Whole menu. Walk-forward, prior slates only. No spread. No NIL. No week dummy. No Massey/SP+/talent ordinals.

Inspiration is March Madness 2025 1st (GLM quality, T1+T2 sums, opponent box) and 2026 2nd/3rd (Colley, SRS, ncsos, interactions, difference+abs). Translated to CFB. Not a kitchen sink of box averages we already have as Pom.

| id | source | what it is |
|---|---|---|
| srs_diff | 2026 3rd | Simple Rating System. Linear solve of `r = mean(margin + opp_r)`, centered. Orthogonal to EPA Pom and MOV Elo. |
| colley_diff | 2026 3rd | Colley Matrix on the season W/L graph. Ignores margin. Default 0.5 if no games. |
| ncsos_diff | 2026 3rd | Mean opponent Pom in games where both conferences are known and differ. 0 if none. |
| margin_std_diff | craft | Season std of scoring margin. Consistency vs volatility. 0 if fewer than 2 games. |
| pom_sum | 2025 / kesiee | `home_pom + away_pom`. Game quality, not gap. |
| pom_abs | 2025 / kesiee | `\|pom_diff\|`. Competitiveness. |
| tempo_abs | Four Factors analog | `\|tempo_diff\|`. Pace mismatch. |
| log_pom_diff | nonlinear | `sign(pom) · log1p(\|pom\|)`. Compresses blowout gaps. |
| tanh_elo_diff | nonlinear | `tanh(elo_diff / 200)`. Saturating Elo gap. |
| pom_elo_prod | 2026 3rd interactions | `pom_diff · elo_diff / 400`. EPA and Elo agreement. |
| form_sos_prod | 2026 3rd interactions | `form4_margin_diff · sos_diff / 20`. Hot team that played quality. |
| srs_elo_prod | 2026 3rd srs×rank | `srs_diff · elo_diff / 400`. SRS and Elo agreement. |

Score, listed before the run:

- locked vs locked+craft
- extras vs extras+craft
- LightGBM, XGBoost, TabPFN-3 batch on both craft sets
- temperature on each, T fit on train log loss only
- TabPFN walk-forward on extras+craft (fit only on earlier rows)

Do not carve a subset after seeing 2025. Promote extras+craft on the usual Brier ≥ 0.002 drop and no worse log loss vs extras, on the trees. Promote into live TabPFN only if the walk-forward extras+craft number also clears that bar vs extras-walk. Fail = stop.

## LOSO menu (diagnostic)

Locked before this score. Whole menu. Walk-forward features, prior slates only. No spread. No NIL. No week dummy. No Massey/SP+/talent ordinals.

**Promote on expanding-year**, not peek-LOSO. For each season 2015–2025, fit on earlier seasons only, score that year’s FBS–FBS once. Report pooled Brier, mean of season Briers, and slices (weeks ≤3 / 4–8 / ≥9). Peek-LOSO (`fit on the other seasons`, including the future) is an audit only — March Madness copied a fold that leaks in weekly CFB.

One idea from each of ten March Madness writeups, translated to CFB:

| id | source | what it is |
|---|---|---|
| glm_quality_diff | 2025 1st (Odeh / modeh7) | Ridge logistic Bradley–Terry on the season W/L graph. MLE team strength, centered. |
| glm_sum | 2025 raddar (vilnius-ncaa) | `glm_home + glm_away`. T1+T2 absolute quality, not the gap. |
| elo_ratio | 2025 7th (wakama1994 / raddar hard) | `home_elo / away_elo − 1`. Scale-free Elo mismatch. |
| log_margin_diff | 2025 Tyser | Mean `sign(m)·log1p(\|m\|)`. Log MOV form, not raw points. |
| conf_pom_diff | 2025 bualimov | Conference mean of last-known team Pom. |
| tier_win_diff | 2026 1st (Horan) | Quality-win points: opp Pom >10 → 6, >0 → 4, >−15 → 2, else 0.25. Sum of wins. |
| form10_win_diff | 2026 2nd (Carlin) | Last-10 win percentage. |
| close_x_margin | 2026 3rd (Miller) | `close_winpct_diff × avg_margin_diff`. Close-game skill times dominance. |
| late_win_diff | 2026 4th (Xroxa) | Win% on slates ≥ 8. Late-season form. 0.5 if none. |
| yoy_margin_diff | 2026 ledmaster | This-season mean margin minus last season’s final mean margin. 0 if no prior year. |

Score, listed before the run:

- extras vs extras+loso on LightGBM, XGBoost, and logistic, expanding-year
- pooled Brier, mean-of-seasons Brier, early/mid/late slices
- per fold: temperature (T fit on earlier seasons only) and clip `[0.02, 0.98]` (2026 2nd)
- peek-LOSO as an audit, not a promote

Do not carve a subset after seeing the score. Promote extras+loso on the expanding-year bar (pooled Δ ≤ −0.002, log loss not worse, not one year). Do not add these to live TabPFN from the tree screen. Live still needs slate walk-forward.

## Conference (diagnostic)

Locked from the LOSO permutation *before this score*. `conf_pom` was the only column with holdout signal. Expand that axis only. Still walk-forward. No spread. No NIL. No week dummy. Do not pull GLM, Elo ratio, or the dead LOSO columns back in.

| id | what it is |
|---|---|
| conf_pom_diff | Conference mean of last-known team Pom |
| conf_elo_diff | Conference mean Elo (2026 4th Xroxa) |
| p4_diff | 1 if ACC/B12/B1G/SEC/P12, else 0. Home minus away. |
| same_conf | 1 if both conferences known and equal. Known at kickoff. |
| conf_win_diff | Win% in conference games. 0.5 if none. |
| conf_margin_diff | Mean margin in conference games. 0 if none. |
| ooc_win_diff | Win% out of conference. 0.5 if none. |
| ooc_margin_diff | Mean margin out of conference. 0 if none. |

Incoming Pom and conference are registered for the current slate before features are read. That is not a result leak.

Score extras vs extras+conf on logistic, LightGBM, and XGBoost, **expanding-year**, 2015–2025 FBS–FBS. Same promotion bar on the expanding-year number. Peek-LOSO conference numbers are an audit. Do not carve. Do not add to live TabPFN from this pass.

## Pace (diagnostic)

Locked before this score. Whole menu. Walk-forward from prior slates' play-by-play only. Garbage-filtered. No spread. No NIL. No week dummy. No Massey/SP+/talent ordinals.

cfbfastR has no blitz flag. This menu is turnovers, rushing efficiency, and two clocks.

| id | what |
|---|---|
| to_margin_diff | Season mean takeaways − giveaways per game. `turnover==1`. |
| ypc_diff | Offense yards per carry. `yds_rushed` on `rush==1`, not sacks. |
| play_speed_diff | Mean wallclock seconds between snaps, same possession and period, gaps in [3, 55]. Faster is lower. |
| sec_per_play_diff | Mean game-clock seconds consumed per scrimmage play, same period, dt in [1, 40]. |

Score extras vs extras+pace on LightGBM, XGBoost, and TabPFN-3 batch (2014–2024 train, 2025 FBS–FBS once). Walk-forward TabPFN on extras+pace. Promote extras+pace on the trees if Brier drops ≥ 0.002 and log loss does not rise vs extras. Promote into live TabPFN only if that walk-forward also clears the bar vs extras-walk (0.1865 / 0.5529). Do not carve. Fail = stop.

## Specials (diagnostic)

Locked before this score. New menu. Do not fold into pace after seeing pace trees. Walk-forward, prior slates only. Garbage-filtered PBP. No spread. No NIL. No week dummy.

| id | what |
|---|---|
| margin_momentum_diff | Last-4 mean margin minus season mean margin. 0 if fewer than 2 games. |
| win_streak_diff | Signed streak: +n wins or −n losses. Resets on a result of the other kind. |
| fg_avg_make_diff | Mean made field-goal distance (`yds_fg`). |
| fg_make_adj_diff | FG make rate minus `fg_make_prob` (distance-adjusted residual). Probabilities >1 treated as percents. |
| punt_rate_diff | Punts / (scrimmage plays + punts). |
| punt_yds_diff | Mean `yds_punted`. |
| plays_pg_diff | Mean offensive scrimmage plays per game. Not the locked tempo rating. |

Score extras vs extras+specials on LightGBM, XGBoost, and TabPFN-3 batch. Walk-forward TabPFN on extras+specials. Same promote bar vs extras / extras-walk. Do not carve. Fail = stop.

## Subsets (diagnostic)

Not a promotion pass. Pace + specials columns only. Selection year is **2024** (train 2014–2023). 2025 is scored after the 2024 ranking is written. A 2025 winner that was not the 2024 pick cannot be promoted.

Locked groups, before either year is scored:

| id | columns |
|---|---|
| pace | to_margin, ypc, play_speed, sec_per_play |
| specials | momentum, streak, fg_avg_make, fg_make_adj, punt_rate, punt_yds, plays_pg |
| clocks | play_speed, sec_per_play |
| rush | ypc |
| ball | to_margin |
| punts | punt_rate, punt_yds |
| kicks | fg_avg_make, fg_make_adj |
| form | momentum, streak |
| plays | plays_pg |
| all | pace + specials |

Also score each column alone, and drop-one from `all`, on 2024 then 2025. LightGBM + XGBoost. TabPFN batch only on extras, `all`, and the 2024-best group. Do not promote from this pass.

## GLM4 (diagnostic)

Locked before this score. Whole menu. Walk-forward, prior slates only. No spread. No NIL. No week dummy.

LOSO already had `glm_quality_diff` and `glm_sum` inside a ten-column sink. That failed. This menu is GLM only, and each number is its own column. Same ridge logistic Bradley–Terry as LOSO (`λ=1`, 25 IRLS, mean-centered). 0 if the team has no games yet.

| id | what |
|---|---|
| glm_home | Home team's GLM strength |
| glm_away | Away team's GLM strength |
| glm_diff | `glm_home − glm_away` |
| glm_sum | `glm_home + glm_away` |

Score extras vs extras+glm4 on logistic, LightGBM, and XGBoost, **expanding-year**, 2015–2025 FBS–FBS. Same promotion bar on the expanding-year number. Do not carve. Do not add to live TabPFN from this pass.

## Note

2025 calibration buckets were inspected before the first protocol was written. The menu is still the standard 0–2 parameter calibration set, not a 2025-tuned list. 2025 is used only as a frozen score.

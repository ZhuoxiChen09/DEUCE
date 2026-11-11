# Tennis Prediction Model — Diagnostic Report

> Historical report, archived 2026-11-10. Its dataset, split procedure, metrics,
> and recommendations are not a statement of the current release. See the
> [research plan](../research/RESEARCH_PLAN.md) and [roadmap](../roadmap.md).

This document is a structured audit of the tennis match outcome prediction model in this project, for external review.

---

## 1. MODEL ARCHITECTURE

| Item | Detail |
|------|--------|
| **Model type** | Logistic regression (binary classifier) |
| **Target variable** | Binary: 1 = player A wins, 0 = player B wins (player A is the first-named player in the matchup) |
| **Training objective** | Maximize log-likelihood of the binary outcome (cross-entropy / log loss) |
| **Loss function** | Log loss (binary cross-entropy). Validation log loss is reported at training time. |
| **Hyperparameters** | `max_iter=1000`, `random_state=42` (sklearn `LogisticRegression`). No regularization strength specified (default L2, C=1.0). |
| **Probability calibration** | **Yes.** Platt scaling is applied: a second logistic regression is fit on the validation set to map raw model outputs to calibrated probabilities. The saved artifact includes a `calibrated_model` wrapper used at inference. |

---

## 2. TRAINING DATA

| Item | Detail |
|------|--------|
| **Total number of matches** | 558 (for the sample run on `data/historical_2026_v3.csv`). Configurable by CSV. |
| **Time range** | Depends on the CSV. For `historical_2026_v3.csv`: 2026-01-02 to 2026-03-02. |
| **Tours included** | ATP Tour and Challenger (inferred from tournament name / level in the CSV). No WTA; no Futures in the default sample. |
| **Train / validation split** | 80% train / 20% validation, stratified by outcome. `val_frac=0.2`, `random_state=42`. |
| **Test set** | No held-out test set; only train/validation split. Validation metrics are used as the reported performance. |
| **Cross-validation** | Not used. Single stratified split. |

---

## 3. FEATURE SET

The **Core** model uses **57 features** (no ATP leaderboard stats; no score-derived features). For each feature:

| feature_name | description | data_source | percent_missing (sample) | missing handling | scaled/normalized | available for Challenger |
|--------------|-------------|-------------|--------------------------|------------------|-------------------|---------------------------|
| rank_diff | B − A rank (positive ⇒ A better) | ATP rankings / CSV rank_a, rank_b | 1.08% | Median imputation + rank_diff_missing=1 | Yes (StandardScaler) | Yes (if rank present) |
| rank_diff_missing | 1 if rank missing for either player | — | — | — | Yes | — |
| log_points_diff | log(1+points_A) − log(1+points_B) | ATP points / CSV points_a, points_b | 1.08% | Median imputation + log_points_diff_missing=1 | Yes | Yes |
| log_points_diff_missing | 1 if points missing | — | — | — | Yes | — |
| surface_speed_rating | Court speed (e.g. 1–10) | Tournament catalog / surface | 0% | Default from context | Yes | Yes |
| indoor_flag | 1 if indoor | Tournament / CSV indoor | 0% | Default | Yes | Yes |
| altitude_factor | Altitude factor | Tournament catalog | 0% | Default | Yes | Yes |
| climate_stress_index | Climate stress | Tournament catalog | 0% | Default | Yes | Yes |
| round_importance | Round importance | Tournament context / round | 0% | Default | Yes | Yes |
| court_type_indoor | Indoor court type | Context | 0% | Default | Yes | Yes |
| tournament_weight | Tournament tier weight | Catalog | 0% | Default | Yes | Yes |
| best_of | Best-of-3 or 5 | CSV / context | 0% | Default 3 | Yes | Yes |
| draw_size | Draw size | CSV / context | 0% | Default 64 | Yes | Yes |
| age_diff | Age A − B | CSV age_a, age_b | 0% | Median + age_diff_missing=1 | Yes | Yes |
| age_diff_missing | 1 if age missing | — | — | — | Yes | — |
| height_diff | Height A − B (cm) | CSV ht_a, ht_b | 0% | Median + height_diff_missing=1 | Yes | Yes |
| height_diff_missing | 1 if height missing | — | — | — | Yes | — |
| hand_lefty_matchup | 1 if one lefty, one righty | CSV hand_a, hand_b | 0% | 0 + hand_lefty_missing=1 if missing | Yes | Yes |
| hand_lefty_missing | 1 if hand missing | — | — | — | Yes | — |
| int_lefty_surface | lefty_matchup × surface_speed_rating | Derived | 0% | — | Yes | Yes |
| seed_diff | Seed B − A | CSV seed_a, seed_b | 27.6% | Median + seed_diff_missing=1 | Yes | Yes (when seeded) |
| seed_diff_missing | 1 if seed missing | — | — | — | Yes | — |
| seeded_flag_a | 1 if A seeded 1–32 | CSV | 0.18% | seeded_flag_a_missing=1 | Yes | Yes |
| seeded_flag_a_missing | 1 if seed unknown | — | — | — | Yes | — |
| seeded_flag_b | 1 if B seeded 1–32 | CSV | 0.18% | seeded_flag_b_missing=1 | Yes | Yes |
| seeded_flag_b_missing | 1 if seed unknown | — | — | — | Yes | — |
| both_seeded_flag | 1 if both seeded | Derived | — | — | Yes | Yes |
| unseeded_vs_seeded_flag | 1 if one seeded, one not | Derived | — | — | Yes | Yes |
| slam_flag | Grand Slam | tourney_level / catalog | 0% | 0/1 | Yes | Yes |
| masters_flag | Masters 1000 | tourney_level | 0% | 0/1 | Yes | Yes |
| atp500_flag | ATP 500 | tourney_level | 0% | 0/1 | Yes | Yes |
| atp250_flag | ATP 250 | tourney_level | 0% | 0/1 | Yes | Yes |
| challenger_flag | Challenger | tourney_level | 0% | 0/1 | Yes | Yes |
| qualifier_flag_a/b | Qualifier entry | CSV entry_a, entry_b | — | 0 if missing | Yes | Yes |
| wildcard_flag_a/b | Wildcard | CSV | — | 0 | Yes | Yes |
| lucky_loser_flag_a/b | Lucky loser | CSV | — | 0 | Yes | Yes |
| special_exempt_flag_a/b | Special exempt | CSV | — | 0 | Yes | Yes |
| qualifier_vs_direct_flag | Qualifier vs direct | Derived | — | — | Yes | Yes |
| wildcard_vs_seeded_flag | WC vs seeded | Derived | — | — | Yes | Yes |
| entry_missing_a | 1 if entry_a unknown | CSV | — | — | Yes | Yes |
| entry_missing_b | 1 if entry_b unknown | CSV | — | — | Yes | Yes |
| event_id_hash | Hash of event/tourney id | tourney_id / context | 50.36% | 0.5 + event_id_missing=1 | Yes | Yes |
| event_id_missing | 1 if event id missing | — | — | — | Yes | — |
| home_flag_a, home_flag_b | Player plays at home country | tournament_country, ioc | 0% | 0 + _missing=1 | Yes | Yes |
| home_flag_a_missing, home_flag_b_missing | 1 if country missing | — | — | — | Yes | — |
| same_country_flag | Same country | ioc_a, ioc_b | 0% | 0 | Yes | Yes |
| home_advantage_diff | Home advantage delta | Derived from home flags | 0% | Median + _missing=1 | Yes | Yes |
| home_advantage_diff_missing | 1 if missing | — | — | — | Yes | — |
| match_num | Match number in event | CSV match_num | 100% (in sample) | Median + match_num_missing=1 | Yes | Yes (if present) |
| match_num_missing | 1 if match_num missing | — | — | — | Yes | — |
| in_event_fatigue_proxy | Proxy for matches played in event | Derived | — | Default | Yes | Yes |

**Scaling:** All features are scaled with `sklearn.preprocessing.StandardScaler` (zero mean, unit variance) fit on the training set only.

**Challenger:** All Core features are defined so they can be populated for Challenger matches (rank, points, tournament level, round, draw, entry, seed when available, demographics, home/country). No feature depends on ATP-only leaderboards (serve/return/pressure/TDI), so Challenger players are fully supported.

---

## 4. MISSING DATA ANALYSIS

Table sorted by **percent_missing** descending (value features; missingness from corresponding `*_missing` flag in the training sample).

| feature_name | percent_missing | num_missing | num_total |
|--------------|----------------|-------------|-----------|
| match_num | 100.00 | 558 | 558 |
| event_id_hash | 50.36 | 281 | 558 |
| seed_diff | 27.60 | 154 | 558 |
| rank_diff | 1.08 | 6 | 558 |
| log_points_diff | 1.08 | 6 | 558 |
| seeded_flag_a | 0.18 | 1 | 558 |
| seeded_flag_b | 0.18 | 1 | 558 |
| age_diff | 0.00 | 0 | 558 |
| height_diff | 0.00 | 0 | 558 |
| hand_lefty_matchup | 0.00 | 0 | 558 |
| home_flag_a | 0.00 | 0 | 558 |
| home_flag_b | 0.00 | 0 | 558 |
| home_advantage_diff | 0.00 | 0 | 558 |

*Generated from `scripts/model_diagnostic_report.py` on `data/historical_2026_v3.csv`.*

---

## 5. FEATURE IMPORTANCE (COEFFICIENTS)

Coefficients from the trained Core model (sorted by absolute coefficient descending). Positive coefficient ⇒ higher value increases P(A wins).

| rank | feature_name | coefficient |
|------|--------------|-------------|
| 1 | seeded_flag_b | 2.669 |
| 2 | both_seeded_flag | 2.119 |
| 3 | challenger_flag | -1.807 |
| 4 | atp500_flag | 1.706 |
| 5 | slam_flag | -1.263 |
| 6 | seeded_flag_a | 1.105 |
| 7 | same_country_flag | 0.456 |
| 8 | masters_flag | 0.450 |
| 9 | log_points_diff | 0.435 |
| 10 | seed_diff_missing | 0.416 |
| 11 | qualifier_flag_b | 0.363 |
| 12 | atp250_flag | 0.304 |
| 13 | seeded_flag_a_missing | 0.211 |
| 14 | seeded_flag_b_missing | 0.211 |
| 15 | int_lefty_surface | 0.203 |
| 16 | event_id_missing | 0.194 |
| 17 | qualifier_flag_a | -0.193 |
| 18 | unseeded_vs_seeded_flag | -0.178 |
| 19 | lucky_loser_flag_b | -0.175 |
| 20 | rank_diff_missing | 0.171 |
| 21 | log_points_diff_missing | 0.171 |
| 22 | wildcard_flag_a | -0.171 |
| 23 | special_exempt_flag_a | -0.161 |
| 24 | special_exempt_flag_b | 0.092 |
| 25 | event_id_hash | -0.092 |
| 26 | seed_diff | 0.084 |
| 27 | qualifier_vs_direct_flag | 0.063 |
| 28 | wildcard_vs_seeded_flag | -0.063 |
| 29–57 | (remaining features) | 0.0 or small |

*Full list is produced by `scripts/model_diagnostic_report.py`.*

---

## 6. MODEL PERFORMANCE

| Metric | Value |
|--------|--------|
| **Train accuracy** | Not saved at training time. Can be reproduced by re-running the diagnostic script with the same sklearn version as training. |
| **Validation accuracy** | 0.8750 (87.50%) |
| **Validation log loss** | 0.3067 |
| **Validation Brier score** | Not computed in default training run. Can be computed by `scripts/model_diagnostic_report.py` when the pickled model loads successfully (same sklearn version as training). |
| **ROC AUC** | Not computed in default training run. Same as above. |
| **Confusion matrix (validation)** | Not saved. Reproducible via the diagnostic script. |

*Note: If the diagnostic script is run in an environment with a different sklearn version than the one used to train the model, the calibrated predictor may fail to run; in that case only coefficients and missing-data stats are produced. Retrain with the same environment to get consistent metrics.*

---

## 7. DATA COVERAGE PROBLEMS

- **ATP leaderboard stats (serve, return, under pressure, TDI, win/loss index):** Intentionally **not** used in the Core model. They are not available for many players (especially Challenger and lower). The model does not depend on them, so no coverage issue for prediction.
- **Rank / points:** Sometimes missing (e.g. 1.08% in the sample). Handled by median imputation and `rank_diff_missing` / `log_points_diff_missing` so the model can downweight or adjust when data is missing.
- **Seeds:** Often missing in early-round or Challenger data (27.6% in sample). Handled by median imputation and `seed_diff_missing` and seeded-flag missing indicators.
- **Event ID:** Missing in about half of the sample when `tourney_id` or catalog event_id is absent; `event_id_hash` is imputed and `event_id_missing=1` is set.
- **Match number in event:** Often missing (100% in the sample CSV). Imputed with median; `match_num_missing=1` is set so the model does not treat it as observed.

**Summary:** Predictions are still produced when features are missing. Missing values are replaced by medians (or 0/0.5 for flags), and corresponding `*_missing` indicators are set to 1. The model uses both the imputed value and the missing flag, so it can adapt to partial data. No feature is required to be present for inference.

---

## 8. PIPELINE SUMMARY

End-to-end flow from **raw player names** to **predicted win probability**:

1. **Resolve player stats**  
   For each of player A and player B: look up rank, points (and optionally other fields) from the **player database** or from in-memory datasets (e.g. ATP rankings, matchups CSV). Build a `PlayerStats` object per player. Missing fields remain `None`.

2. **Tournament context (optional)**  
   If a tournament name is provided, look up surface, draw size, best-of, round, tier (slam/masters/500/250/challenger), and other context from the **tournament catalog**. Used for surface_speed_rating, indoor_flag, round_importance, tournament_weight, etc.

3. **Match context (optional)**  
   If available: demographics (age, height, hand), seed, entry type (qualifier, wildcard, etc.), country (for home advantage), match number, draw/tourney id. Comes from the matchup record or CSV row.

4. **Feature construction**  
   `build_match_features(player_a, player_b, tournament_context, medians, match_context, feature_set="core")`:  
   - Compute deltas (e.g. rank_diff, log_points_diff, age_diff, seed_diff).  
   - For any missing value: substitute median (from training or default) and set the corresponding `*_missing` flag to 1.  
   - Compute context features (surface, indoor, round, best_of, draw_size, tourney-level flags).  
   - Compute entry and seed flags, event_id_hash, home advantage, match_num, in_event_fatigue_proxy.  
   - Return a **57-dimensional Core feature vector** and a feature dict.

5. **Model selection**  
   The app/CLI loads up to three artifacts: Core, Enhanced, and legacy. Enhanced currently uses the same feature set as Core. The active model is chosen (e.g. Core if present); no ATP-coverage gating is used for model selection.

6. **Scaling**  
   The 57-dim vector is transformed with the **scaler** saved in the model bundle (fit on training data only).

7. **Probability**  
   The **calibrated model** (Platt-scaled logistic regression) is applied to the scaled vector to produce P(A wins). P(B wins) = 1 − P(A wins).

8. **Post-processing (optional)**  
   If enabled: extreme probabilities can be capped when data completeness is low; single-bucket dominance can trigger shrinkage toward 0.5 and a warning. Warnings and evidence quality are returned with the prediction.

9. **Output**  
   `PredictionResult`: `model_prob_a`, `model_prob_b`, `model_used`, `evidence_quality`, `warnings`, and optional contribution breakdown.

---

*Report generated for the DEUCE tennis prediction model. Empirical numbers (missing %, coefficients, validation metrics) can be regenerated by running `python scripts/model_diagnostic_report.py` from the project root.*

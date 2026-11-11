# DEUCE research questions and experiment protocol

Status: proposed experiments, organized 2026-11-10. New court-speed features,
datasets, and performance improvements have not been implemented or measured.

## Motivation

Do historically available court conditions change the predictive value of
players' serve and return strengths beyond ranking and broad surface type?
DEUCE connects tennis knowledge, supervised learning, and statistical modeling.

## Questions

| ID | Question | Controlled comparison | Issue |
|---|---|---|---|
| RQ1 | Do learned probabilities outperform ranking and Elo? | Constant/prior, rank/points LR, overall Elo, surface Elo, current Core | [#3](https://github.com/ZhuoxiChen09/DEUCE/issues/3) |
| RQ2 | Does speed moderate serve/return advantages? | Baseline; plus player strength; plus speed interactions | [#5](https://github.com/ZhuoxiChen09/DEUCE/issues/5) |
| RQ3 | Does CPI add information beyond adjusted-ace speed? | TA, CPI, and both on identical matches | [#6](https://github.com/ZhuoxiChen09/DEUCE/issues/6) |
| RQ4 | Does partial pooling stabilize sparse condition estimates? | Hierarchical server/returner/venue model versus lagged estimates | [#7](https://github.com/ZhuoxiChen09/DEUCE/issues/7), later |

Primary initial speed hypothesis: speed x player-strength interactions improve
future-match log loss beyond ranking, surface, and player strength alone.
Null or negative findings remain reportable outcomes.

## Foundation

- [Evaluation parity #1](https://github.com/ZhuoxiChen09/DEUCE/issues/1): share
  weighting, mirroring, calibration, symmetry, and shrinkage policy with serving.
  Split original matches before augmentation.
- [Reproducible CI #2](https://github.com/ZhuoxiChen09/DEUCE/issues/2): diagnose the
  failed default-branch run without guessing its cause.
- [Source audit #4](https://github.com/ZhuoxiChen09/DEUCE/issues/4): verify usable
  historical observations and past-only player statistics before feature joins.

The existing compact demo has 1,232 matches and 1,072 OOS predictions: log loss
0.6778698683, Brier 0.2419111202, accuracy 0.6156716418, AUC 0.6190595550.
These describe the existing evaluation procedure, not new feature performance or
the full served probability pipeline. See [metrics](../../artifacts/demo/demo_metrics.json)
and [fold report](../../artifacts/demo/walk_forward_report.json).

## Sources

| Source | Meaning | Proposed role |
|---|---|---|
| [Tennis Abstract](https://tennisabstract.com/reports/atp_surface_speed.html) | Server/returner-adjusted ace-rate proxy; 1.0 is average | Primary historical conditions estimate after timing verification |
| [CourtSpeed CPI](https://courtspeed.com/) | Compiled pace readings, commonly stadium court | Separate measurement and incremental-information comparison |
| [CourtSpeed methodology](https://courtspeed.com/method-notes/) | Composite also uses ace rate, rally length, and air density | Later ablation, not independent evidence |

TA 1.25 means 25% more aces than expected under average conditions, not 25% greater
ball velocity. Keep TA and CPI in separate fields with original units.

## Data contract and availability

Record tournament_id, venue_id, edition, surface/indoor status, measurement_type,
value, unit, measured_period_end, available_at with evidence, retrieved_at,
source_url, court_scope, missing_reason, and known revisions.

- Resolve venue/edition identity rather than joining only on tournament name.
  Handle relocations/alternating venues and stadium-versus-event court scope.
- Inputs must be available before the prediction cutoff. Completed-event ace,
  rally, and weather summaries cannot predict earlier matches in that event.
- Retrieval today is not proof of historical availability. Verify prior-edition
  publication timing too; label unverified reconstructions as exploratory.
- Use verified prior observations/trailing estimates first. Make surface/indoor
  fallbacks, observation age, and missingness explicit.
- Use conservative date/event boundaries when exact match ordering is unavailable.
- Build player serve/return strengths from past point counts, with sample sizes
  and shrinkage. Do not use current snapshots or the target match's own statistics.
- Document reuse terms/attribution separately from temporal validity. Existing
  public visibility does not establish data or model redistribution clearance.

## Features and first model

Use separate ta_speed_prior and cpi_prior values, standardized within each
training fold. Existing code clips speed to 0-10 and uses 11-minus-speed:
those assumptions must be replaced for the new measurements.

Let DeltaServe and DeltaReturn be A-minus-B past strengths and z be centered speed:

```text
logit P(A wins) = baseline matchup terms
               + beta_s * DeltaServe + beta_r * DeltaReturn
               + gamma_s * z * DeltaServe + gamma_r * z * DeltaReturn
```

Retain lower-order player-strength terms and A/B symmetry. A court value shared by
both players does not alone identify who benefits. Estimate interaction signs.
Use an explicit research feature configuration: current Enhanced routing and
leaderboard snapshots do not supply historically valid player estimates.

## Evaluation contract

1. Define the ATP population, exclusions, match IDs, and date semantics. Use the
   larger history for research; retain compact data for fast demos.
2. Freeze outer chronological windows and comparison populations. Tune features,
   regularization, and calibration in earlier temporal windows. Identify already
   inspected history honestly when reserving a final evaluation period.
3. Keep match orientations together. Fit preprocessing, historical aggregates,
   and calibration without future information.
4. Primary metric: paired OOS log-loss difference. Also report Brier, calibration,
   AUC, accuracy, sample size, coverage, and fallback usage.
5. Use time/event-block uncertainty intervals and examine block-choice sensitivity.
   Retain weak folds and all predeclared primary comparisons.
6. Examine year/event stability and hard-court-only performance to distinguish
   speed information from broad surface categories.
7. Compare CPI models on identical source-overlap matches first; separately report
   full-population results with specified fallbacks.
8. Record source/model/configuration versions and per-match predictions sufficient
   for reproduction within the permitted distribution scope.

Promote models only when evidence supports them; no target log loss is guaranteed.
Observational coefficients and scenario sliders do not establish causal effects.

## Coursework and portfolio evidence

| Topic | Demonstration |
|---|---|
| COGS 118A | Regularization, feature engineering, controlled comparisons, generalization |
| COGS 118D | GLM moderation; later mixed effects, latent variables, parameter recovery, model checking |
| Probability/statistics | Likelihood, calibration, paired uncertainty, measurement error |
| Research methods | Falsifiable hypotheses, confounding, leakage prevention, negative findings |

Deliver a concise report and matchup sensitivity chart across supported speed
ranges. Distinguish observations, estimates, and hypothetical scenarios.
A hierarchical extension should demonstrate parameter recovery on simulations
before interpreting latent effects. Do not label them as unmeasured player
confidence, pressure, or mental fatigue.

See the [roadmap](../roadmap.md) for sequencing and linked GitHub tasks.

# DEUCE research roadmap

The current offline demo is a baseline. Court-speed features and the experiments
below are planned work, not improvements already demonstrated.
See the [research protocol](research/RESEARCH_PLAN.md) for hypotheses and controls.
GitHub issues own execution status.

## 1. Trustworthy evaluation

[Milestone 1](https://github.com/ZhuoxiChen09/DEUCE/milestone/1)

1. [#1 Align training, calibration, and served probabilities](https://github.com/ZhuoxiChen09/DEUCE/issues/1).
2. [#2 Restore reproducible CI](https://github.com/ZhuoxiChen09/DEUCE/issues/2).
3. [#3 RQ1: compare ranking, Elo, and learned probabilities](https://github.com/ZhuoxiChen09/DEUCE/issues/3).

Exit: the evaluated procedure matches serving, checks pass, and baselines use
identical documented temporal evaluation matches.

## 2. Court conditions research

[Milestone 2](https://github.com/ZhuoxiChen09/DEUCE/milestone/2)

1. [#4 Audit speed coverage and historical availability](https://github.com/ZhuoxiChen09/DEUCE/issues/4).
2. [#5 RQ2: test speed x serve/return interactions](https://github.com/ZhuoxiChen09/DEUCE/issues/5).
3. [#6 RQ3: test incremental information from CPI](https://github.com/ZhuoxiChen09/DEUCE/issues/6).

The data audit can run alongside milestone 1. Model conclusions require both.
Exit: reproducible comparisons with coverage, calibration, uncertainty, and
year/event sensitivity, including negative findings.

## 3. Portfolio evidence

[Milestone 3](https://github.com/ZhuoxiChen09/DEUCE/milestone/3)

- [#8 Research walkthrough and matchup sensitivity](https://github.com/ZhuoxiChen09/DEUCE/issues/8).
- [#9 Resolve public distribution scope](https://github.com/ZhuoxiChen09/DEUCE/issues/9).
  The source/permission decision can begin immediately.

Exit: verified claims, reproducible commands, clearly distinguished
observations/estimates/scenarios, and a documented distribution decision.

## Later

- [#7 RQ4: hierarchical court-condition estimates](https://github.com/ZhuoxiChen09/DEUCE/issues/7).
- Live providers, trading tools, broad API expansion, and WTA ingestion are outside
  these milestones. Retained code does not imply supported production capability.

## Branch and cleanup policy

Use short-lived implementation branches. Delete after merge once their changes
are retained on the default branch; check unmerged commits and active worktrees
first. Do not remove compatibility modules or research data solely because they
are outside the current milestone.

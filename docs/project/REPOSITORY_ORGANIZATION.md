# Repository organization record

Date: 2026-11-10.

## Changes

- Added a research protocol, documentation guide, three GitHub milestones, and
  nine scoped issues with research/data/evaluation/foundation labels.
- Replaced the obsolete live-service roadmap with evaluation, court-conditions,
  and portfolio-evidence milestones.
- Archived four superseded audits/handoffs with historical-status banners.
- Removed scripts/project_health_report.py. It used the old project name and flat
  artifact layout, missed the nested demo bundle, and recommended stale work.
  Use python -m deuce.cli demo validate for supported bundle validation.
- Untracked .claude/settings.local.json while preserving its local copy. Ignored
  machine-local settings/worktrees and .codex state.
- Retained runtime modules, tests, data, and model artifacts. Compatibility modules
  still have import/pickle obligations; removal needs a separate migration.

## Branches and work in progress

The local and remote inventories contained only main. No branch qualified for
deletion. Pre-existing uncommitted extension files, exporter, dependency change,
and README extension edits are excluded from the organization commit.

## Release state

GitHub reported public visibility, contrary to the existing private-release notice.
Visibility was not changed. [Issue #9](https://github.com/ZhuoxiChen09/DEUCE/issues/9)
tracks the owner's source/permission decision. This cleanup adds no dataset or
model artifact.

## Verification scope

Check documentation links, issue/milestone references, staged-diff whitespace,
and preservation of pre-existing work. This documentation/metadata cleanup does
not retrain a model or repair the previously failing CI; that work is tracked in
[issue #2](https://github.com/ZhuoxiChen09/DEUCE/issues/2).

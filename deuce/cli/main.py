"""
Structured CLI for DEUCE. Dispatches to existing modules so behavior stays aligned.

Usage:
  python -m deuce.cli train --data data/historical.csv --out artifacts/m.pkl --model-type core
  python -m deuce.cli inspect-model artifacts/tennis_model_core.pkl
  python -m deuce.cli research-report --csv path/to/preds.csv
  python -m deuce.cli walk-forward-report --csv data/historical.csv --model-type core
  python -m deuce.cli feature-tiers
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _legacy_cli_args(argv: list[str]) -> int:
    """Forward to legacy ``cli.py`` with same cwd."""
    root = _root()
    cli_py = root / "cli.py"
    cmd = [sys.executable, str(cli_py), *argv]
    return subprocess.call(cmd, cwd=str(root))


def _dispatch_legacy_fetch(argv: list[str]) -> int:
    """Explicit compatibility path for market-fetching workflows still implemented in ``cli.py``."""
    return _legacy_cli_args(argv)


def _dispatch_legacy_update_db() -> int:
    """Explicit compatibility path for player DB updates still implemented in ``cli.py``."""
    return _legacy_cli_args(["--update-player-db"])


def cmd_train(ns: argparse.Namespace) -> int:
    from deuce.training.train import train_and_save, train_from_data_dir

    try:
        if ns.data_dir:
            summary = train_from_data_dir(
                ns.data_dir,
                ns.out,
                val_frac=ns.val_frac,
                random_state=ns.random_state,
                catalog_path=ns.catalog,
                model_type=ns.model_type,
                split_strategy=ns.split_strategy,
                calibration_method=ns.calibration,
                test_frac=ns.test_frac,
            )
            print(f"Files: {summary['n_files']}  Total: {summary['n_total']}  Train/Val: {summary['n_train']}/{summary['n_val']}")
        else:
            if not ns.data:
                print("Provide --data or --data-dir", file=sys.stderr)
                return 1
            train_and_save(
                ns.data,
                ns.out,
                val_frac=ns.val_frac,
                random_state=ns.random_state,
                catalog_path=ns.catalog,
                model_type=ns.model_type,
                mirror_rows=(False if ns.no_mirror else None),
                split_strategy=ns.split_strategy,
                calibration_method=ns.calibration,
                test_frac=ns.test_frac,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_inspect(ns: argparse.Namespace) -> int:
    import deuce.training.train as _train_mod

    path = Path(ns.path)
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    class _U(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module in ("__main__", "model_train", "deuce.training.train") and name in ("_PlattCalibratedWrapper", "_BundleCalibratedWrapper"):
                return getattr(_train_mod, name)
            return super().find_class(module, name)

    with open(path, "rb") as f:
        obj = _U(f).load()
    if isinstance(obj, dict) and obj.get("version") in (2, 3):
        meta = {k: v for k, v in obj.items() if k not in ("scaler", "model", "calibrated_model")}
        tier = obj.get("model_tier") or obj.get("model_type") or "?"
        feats = obj.get("feature_names") or []
        n_feat = len(feats)
        print("=== Summary ===")
        print(f"  Tier: {tier}  |  feature_set_name: {obj.get('feature_set_name', '?')}")
        print(f"  Features: {n_feat}  |  split_strategy: {obj.get('split_strategy', '?')}")
        print(f"  Calibration: {obj.get('calibration_requested') or (obj.get('calibration') or {}).get('method', '?')}")
        tw = obj.get("train_date_window") or {}
        vw = obj.get("validation_date_window") or {}
        tew = obj.get("test_date_window") or {}
        print(f"  Train dates: {tw.get('min')} → {tw.get('max')}")
        print(f"  Val dates:   {vw.get('min')} → {vw.get('max')}")
        print(f"  Test dates:  {tew.get('min')} → {tew.get('max')}")
        print(f"  Samples: total={obj.get('n_samples_total', '?')} train={obj.get('n_train', '?')} val={obj.get('n_val', '?')} test_holdout={obj.get('n_test_holdout', '?')}")
        print(f"  code_version: {obj.get('code_version', '?')}")
        miss = obj.get("missingness_summary_train") or {}
        if miss:
            top = sorted(miss.items(), key=lambda x: -x[1])[:8]
            print("  Missingness (train, top):", ", ".join(f"{k}={v:.2f}" for k, v in top))
        print("=== Bundle metadata (no estimators) ===")
        print(json.dumps(meta, indent=2, default=str))
    else:
        print("Legacy pickle (no bundle metadata).")
    fj = path.with_suffix(path.suffix + ".features.json")
    if fj.exists():
        print("\n=== Sidecar JSON ===")
        print(fj.read_text(encoding="utf-8"))
    return 0


def cmd_research_report(ns: argparse.Namespace) -> int:
    from deuce.research.report import report_to_json_dict, research_report_from_csv

    p = Path(ns.csv)
    if not p.exists():
        print(f"Not found: {p}", file=sys.stderr)
        return 1
    from deuce.backtest.engine import BacktestConfig

    cfg = BacktestConfig(
        edge_threshold=ns.edge_threshold,
        stake_mode=ns.stake_mode,
        flat_fraction=ns.flat_fraction,
    )
    rep = research_report_from_csv(str(p), backtest_cfg=cfg)
    print(json.dumps(report_to_json_dict(rep), indent=2, default=str))
    return 0


def cmd_feature_tiers(_: argparse.Namespace) -> int:
    from deuce.features.tier_manifest import print_feature_tiers_json

    print(print_feature_tiers_json())
    return 0


def cmd_walk_forward_report(ns: argparse.Namespace) -> int:
    from deuce.backtest.engine import BacktestConfig
    from deuce.research.walk_forward import aggregate_report_to_json, run_walk_forward_from_csv, save_walk_forward_outputs

    p = Path(ns.csv)
    if not p.exists():
        print(f"Not found: {p}", file=sys.stderr)
        return 1
    cfg = BacktestConfig(
        edge_threshold=ns.edge_threshold,
        stake_mode=ns.stake_mode,
        flat_fraction=ns.flat_fraction,
    )
    rep = run_walk_forward_from_csv(
        str(p),
        model_type=ns.model_type,
        catalog_path=ns.catalog,
        mirror_rows=ns.mirror_rows,
        mode=ns.mode,
        val_frac=ns.val_frac,
        calibration_method=ns.calibration,
        random_state=ns.random_state,
        min_pool_rows=ns.min_pool_rows,
        n_oos_folds=ns.n_folds,
        train_rows=ns.train_rows,
        val_rows=ns.val_rows,
        oos_rows=ns.oos_rows,
        rolling_step=ns.rolling_step,
        backtest_cfg=cfg,
    )
    summary = aggregate_report_to_json(rep)
    print(json.dumps(summary, indent=2, default=str))
    if ns.out_json or ns.out_oos_csv:
        save_walk_forward_outputs(rep, out_json=ns.out_json, out_oos_csv=ns.out_oos_csv)
    return 0


def cmd_update_db(ns: argparse.Namespace) -> int:
    """Update player stats DB from ATP datasets."""
    try:
        from deuce.player_db import update_db_from_atp_datasets, DEFAULT_DB_PATH
        db_path = ns.db or str(DEFAULT_DB_PATH)
        update_db_from_atp_datasets(db_path)
        print(f"Player DB updated: {db_path}")
    except Exception as e:
        print(f"Error updating player DB: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_build_dataset(ns: argparse.Namespace) -> int:
    """Build training CSV from TML year CSVs via scripts.build_historical_from_tml."""
    import subprocess
    root = _root()
    script = root / "scripts" / "build_historical_from_tml.py"
    if not script.exists():
        print(f"Script not found: {script}", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script)]
    for f in (ns.inputs or []):
        cmd += ["-i", f]
    if ns.output:
        cmd += ["-o", ns.output]
    if ns.db:
        cmd += ["--db", ns.db]
    return subprocess.call(cmd, cwd=str(root))


def cmd_demo_build(ns: argparse.Namespace) -> int:
    """Build local portfolio-demo artifacts from committed demo data."""
    try:
        from deuce.demo.build import build_demo_artifacts, format_build_summary

        manifest = build_demo_artifacts(rebuild_dataset=bool(ns.rebuild_dataset))
        print(format_build_summary(manifest))
    except Exception as e:
        print(f"Error building demo artifacts: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_demo_validate(_: argparse.Namespace) -> int:
    """Validate the committed portfolio-demo runtime bundle."""
    try:
        from deuce.demo.artifacts import validate_demo_artifacts

        summary = validate_demo_artifacts()
        print("Demo runtime bundle is ready.")
        print(
            "Scores: "
            f"dataset={summary['dataset_rows']}, "
            f"OOS={summary['oos_predictions']}, "
            f"accuracy={summary['accuracy']:.4f}, "
            f"ROC-AUC={summary['roc_auc']:.4f}, "
            f"Brier={summary['brier']:.4f}"
        )
    except Exception as e:
        print(f"Demo runtime validation failed: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_app(_: argparse.Namespace) -> int:
    root = _root()
    print("Run from project root:")
    print(f'  cd "{root}"')
    print(f"  {sys.executable} -m streamlit run app.py")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="deuce", description="DEUCE tennis market CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train model (wraps model_train.py)")
    p_train.add_argument("--data", default=None, help="Single CSV (omit if using --data-dir)")
    p_train.add_argument("--data-dir", dest="data_dir", default=None, help="Directory of TML year CSVs")
    p_train.add_argument("--out", default="artifacts/tennis_model.pkl")
    p_train.add_argument("--val-frac", type=float, default=0.2)
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.add_argument("--catalog", default=None)
    p_train.add_argument("--model-type", choices=("core", "enhanced", "legacy"), default="legacy")
    p_train.add_argument("--split-strategy", choices=("random", "time", "walkforward"), default="random")
    p_train.add_argument("--calibration", choices=("none", "platt", "isotonic"), default="platt")
    p_train.add_argument("--test-frac", type=float, default=0.0)
    p_train.add_argument("--no-mirror", action="store_true")
    p_train.set_defaults(func=cmd_train)

    sub.add_parser("fetch", help="Kalshi / tennis search (legacy cli)")

    p_rr = sub.add_parser("research-report", help="Metrics from CSV of predictions + optional market columns")
    p_rr.add_argument("--csv", required=True, help="CSV with model_prob_a, outcome_a_won, optional market_price_a/b")
    p_rr.add_argument("--edge-threshold", type=float, default=0.0)
    p_rr.add_argument("--stake-mode", choices=("flat", "fixed_fraction", "kelly"), default="flat")
    p_rr.add_argument("--flat-fraction", type=float, default=0.01)
    p_rr.set_defaults(func=cmd_research_report)

    p_in = sub.add_parser("inspect-model", help="Print bundle metadata + features JSON")
    p_in.add_argument("path")
    p_in.set_defaults(func=cmd_inspect)

    p_ft = sub.add_parser("feature-tiers", help="Print Core/Enhanced/legacy feature manifest JSON")
    p_ft.set_defaults(func=cmd_feature_tiers)

    p_wf = sub.add_parser(
        "walk-forward-report",
        help="Walk-forward retrain per fold; aggregate OOS log loss, Brier, ROI-style metrics",
    )
    p_wf.add_argument("--csv", required=True, help="Training-format CSV (same as train --data)")
    p_wf.add_argument("--model-type", choices=("core", "enhanced", "legacy"), default="core")
    p_wf.add_argument("--catalog", default=None)
    p_wf.add_argument("--mirror-rows", action="store_true", help="Mirror A/B rows (same as training)")
    p_wf.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    p_wf.add_argument("--val-frac", type=float, default=0.2, dest="val_frac", help="Validation fraction within pool (expanding)")
    p_wf.add_argument("--calibration", choices=("none", "platt", "isotonic"), default="platt")
    p_wf.add_argument("--random-state", type=int, default=42)
    p_wf.add_argument("--min-pool-rows", type=int, default=40, dest="min_pool_rows")
    p_wf.add_argument("--n-folds", type=int, default=3, dest="n_folds", help="Number of OOS segments (expanding)")
    p_wf.add_argument("--train-rows", type=int, default=30, dest="train_rows")
    p_wf.add_argument("--val-rows", type=int, default=8, dest="val_rows")
    p_wf.add_argument("--oos-rows", type=int, default=6, dest="oos_rows")
    p_wf.add_argument("--rolling-step", type=int, default=4, dest="rolling_step")
    p_wf.add_argument("--edge-threshold", type=float, default=0.0)
    p_wf.add_argument("--stake-mode", choices=("flat", "fixed_fraction", "kelly"), default="flat")
    p_wf.add_argument("--flat-fraction", type=float, default=0.01)
    p_wf.add_argument("--out-json", default=None, dest="out_json")
    p_wf.add_argument("--out-oos-csv", default=None, dest="out_oos_csv")
    p_wf.set_defaults(func=cmd_walk_forward_report)

    p_app = sub.add_parser("app", help="Show how to launch Streamlit")
    p_app.set_defaults(func=cmd_app)

    p_udb = sub.add_parser("update-db", help="Update player stats DB from ATP datasets")
    p_udb.add_argument("--db", default=None, help="Path to player_stats.json (default: deuce default)")
    p_udb.set_defaults(func=cmd_update_db)

    p_ds = sub.add_parser("build-dataset", help="Build training CSV from TML year CSVs")
    p_ds.add_argument("-i", "--inputs", nargs="+", default=None, help="Input TML CSV file(s)")
    p_ds.add_argument("-o", "--output", default=None, help="Output training CSV path")
    p_ds.add_argument("--db", default=None, help="Player DB JSON to enrich with serve/return stats")
    p_ds.set_defaults(func=cmd_build_dataset)

    p_demo = sub.add_parser("demo", help="Portfolio demo utilities")
    demo_sub = p_demo.add_subparsers(dest="demo_cmd", required=True)
    p_demo_build = demo_sub.add_parser(
        "build",
        help="Build local demo model and walk-forward artifacts",
        description="Build local demo model and walk-forward artifacts",
    )
    p_demo_build.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Regenerate data/demo/demo_matches.csv from data/historical_tml_2000_2025.csv before building artifacts.",
    )
    p_demo_build.set_defaults(func=cmd_demo_build)
    p_demo_validate = demo_sub.add_parser(
        "validate",
        help="Validate committed demo files required by Streamlit",
        description="Validate committed demo files required by Streamlit",
    )
    p_demo_validate.set_defaults(func=cmd_demo_validate)

    known = {
        "train",
        "fetch",
        "update-db",
        "build-dataset",
        "backtest",
        "inspect-model",
        "app",
        "research-report",
        "feature-tiers",
        "walk-forward-report",
        "demo",
    }
    if not argv or argv[0] not in known:
        return _legacy_cli_args(argv)

    ns = parser.parse_args(argv)

    if ns.cmd == "fetch":
        return _dispatch_legacy_fetch(argv[1:])
    if ns.cmd == "backtest":
        print(
            "Backtest: build rows with model_prob_a, market_price_a/b, outcome_a_won; "
            "call deuce.backtest.run_backtest or `python -m deuce.cli research-report --csv ...`.",
            file=sys.stderr,
        )
        return 0

    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())

"""
Project health report for kalshi_prediction_market.

Run from project root:
  python kalshi_prediction_market/scripts/project_health_report.py

Prints a plain-text summary you can copy/paste into another chat.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def _list_files(pattern: str) -> List[Path]:
    return sorted(ROOT.glob(pattern))


def main() -> int:
    lines: List[str] = []

    lines.append("=== PROJECT HEALTH REPORT: kalshi_prediction_market ===")
    lines.append(f"Root: {ROOT}")
    lines.append("")

    # 1) Key files / modules
    key_files = [
        "README.md",
        "requirements.txt",
        "app.py",
        "cli.py",
        "model_features.py",
        "model_train.py",
        "model_predict.py",
        "tournament_catalog.py",
        "weather_enrichment.py",
        "player_db.py",
    ]
    lines.append("1. KEY FILES")
    for rel in key_files:
        status = "OK" if _exists(rel) else "MISSING"
        lines.append(f"- {rel}: {status}")
    lines.append("")

    # 2) Data summary
    data_dir = ROOT / "data"
    lines.append("2. DATA FILES (data/)")
    if not data_dir.exists():
        lines.append("- data/: MISSING")
    else:
        csvs = sorted(data_dir.glob("*.csv"))
        if not csvs:
            lines.append("- No .csv files in data/")
        else:
            for p in csvs:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"- {p.name}: {size} bytes")
    lines.append("")

    # 3) Model artifacts
    artifacts_dir = ROOT / "artifacts"
    lines.append("3. MODEL ARTIFACTS (artifacts/)")
    if not artifacts_dir.exists():
        lines.append("- artifacts/: MISSING")
    else:
        pkls = sorted(artifacts_dir.glob("*.pkl"))
        if not pkls:
            lines.append("- No .pkl model bundles in artifacts/")
        else:
            # Determine defaults according to model_predict
            try:
                from model_predict import (
                    get_default_model_path,
                    get_default_core_path,
                    get_default_enhanced_path,
                )

                default_legacy = Path(get_default_model_path()) if get_default_model_path() else None
                default_core = Path(get_default_core_path()) if get_default_core_path() else None
                default_enh = Path(get_default_enhanced_path()) if get_default_enhanced_path() else None
            except Exception:
                default_legacy = default_core = default_enh = None

            for p in pkls:
                tag = []
                if default_legacy and p.resolve() == default_legacy.resolve():
                    tag.append("USED_AS_LEGACY_DEFAULT")
                if default_core and p.resolve() == default_core.resolve():
                    tag.append("USED_AS_CORE_DEFAULT")
                if default_enh and p.resolve() == default_enh.resolve():
                    tag.append("USED_AS_ENHANCED_DEFAULT")
                if not tag:
                    tag.append("EXTRA/LEGACY_TEST")
                lines.append(f"- {p.name}: {', '.join(tag)}")

        # List any JSON reports
        jsons = sorted(artifacts_dir.glob("*.json"))
        if jsons:
            lines.append("")
            lines.append("  JSON reports:")
            for p in jsons:
                lines.append(f"  - {p.name}")
    lines.append("")

    # 4) Docs
    docs_dir = ROOT / "docs"
    lines.append("4. DOCS (docs/)")
    if not docs_dir.exists():
        lines.append("- docs/: MISSING")
    else:
        mds = sorted(docs_dir.glob("*.md"))
        if not mds:
            lines.append("- No .md docs in docs/")
        else:
            for p in mds:
                lines.append(f"- {p.name}")
    lines.append("")

    # 5) Scripts
    scripts_dir = ROOT / "scripts"
    lines.append("5. SCRIPTS (scripts/)")
    if not scripts_dir.exists():
        lines.append("- scripts/: MISSING")
    else:
        pys = sorted(scripts_dir.glob("*.py"))
        for p in pys:
            lines.append(f"- {p.name}")
    lines.append("")

    # 6) Suggested cleanups / missing pieces
    lines.append("6. SUGGESTED CLEANUPS / NEXT STEPS")
    lines.append("- Train Core model on full historical: model_train.py --data data/historical_tml_2000_2025.csv --model-type core --out artifacts/tennis_model_core.pkl")
    lines.append("- Consider deleting unused artifacts in artifacts/ marked EXTRA/LEGACY_TEST once you are sure you don't need them.")
    lines.append("- Keep README.md and DATASET_ALIGNMENT_AUDIT.md as the main docs; remove or archive any planning docs you no longer use.")
    lines.append("- Ensure app.py shows the Core model path (get_default_core_path) in the 'Trained model' section instead of only tennis_model.pkl.")

    report = "\n".join(lines)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


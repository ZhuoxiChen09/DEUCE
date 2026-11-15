"""One-shot historical import.

Usage:
  python -m deuce.jobs.import_data
  python -m deuce.jobs.import_data --source data/demo/demo_matches.csv --mode block
"""

from __future__ import annotations

import argparse
import logging
import sys

from deuce.ingestion.adapters import get_adapter
from deuce.ingestion.job import run_import


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="deuce.jobs.import_data", description="Import historical matches")
    parser.add_argument("--source", default=None, help="CSV path for the local_csv adapter")
    parser.add_argument("--adapter", default=None, help="Registered adapter name (default: DEUCE_IMPORT_ADAPTER)")
    parser.add_argument("--mode", choices=("block", "flag"), default=None, help="DQ failure policy")
    args = parser.parse_args(argv)
    try:
        adapter = get_adapter(args.adapter, path=args.source) if (args.adapter or args.source) else get_adapter(args.adapter)
        outcome = run_import(adapter=adapter, mode=args.mode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={outcome.status} source={outcome.source_id} mode={outcome.mode} "
        f"seen={outcome.rows_seen} loaded={outcome.rows_loaded} rejected={outcome.rows_rejected}"
    )
    print(outcome.message)
    if outcome.status in {"blocked", "error"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

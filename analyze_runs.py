"""Summarize prepare.py's run history from app/data/logs/prepare_runs.jsonl.

Meant to be run by a person or an agent, without re-running prepare.py,
to answer questions like:
  - When did prepare.py last run, and did it succeed?
  - What flights is Comet reporting that gulf_flights.txt is filtering out?
  - Is crop/video/report download actually happening, or silently skipped?

Usage:
    python analyze_runs.py                  # summary table of all runs
    python analyze_runs.py --last 5          # only the most recent 5 runs
    python analyze_runs.py --latest          # full detail on the most recent run
    python analyze_runs.py --excluded-flights  # flights currently excluded by
                                                # the gulf_flights.txt allowlist,
                                                # per the most recent run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LOG_PATH = Path("app/data/logs/prepare_runs.jsonl")


def load_runs(log_path: Path = LOG_PATH) -> list[dict]:
    if not log_path.exists():
        return []
    runs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def print_table(runs: list[dict]):
    if not runs:
        print(f"No runs recorded yet in {LOG_PATH}.")
        return
    header = f"{'started_at':22} {'status':9} {'dur_s':7} {'flights':7} {'excluded':9} {'crops_dl':9}"
    print(header)
    print("-" * len(header))
    for r in runs:
        counts = r.get("counts", {})
        n_flights = len(counts.get("latest_predictions_flights", []) or [])
        n_excluded = len(counts.get("flights_excluded_by_gulf_filter", []) or [])
        crop_dl = counts.get("crop_download_triggered")
        crop_dl_str = "yes" if crop_dl else ("no" if crop_dl is not None else "-")
        print(f"{r.get('started_at', '?'):22} {r.get('status', '?'):9} "
              f"{r.get('duration_s', '?'):<7} {n_flights:<7} {n_excluded:<9} {crop_dl_str:<9}")


def print_detail(run: dict):
    print(json.dumps(run, indent=2))


def print_excluded_flights(run: dict):
    excluded = run.get("counts", {}).get("flights_excluded_by_gulf_filter", [])
    if not excluded:
        print("No flights currently excluded by the gulf_flights.txt filter "
              "(as of the most recent run).")
        return
    print(f"{len(excluded)} flight(s) present in Comet but excluded because "
          f"they are not listed in app/data/gulf_flights.txt:")
    for basename in excluded:
        print(f"  - {basename}")
    print("\nIf these should be included, add each basename as its own line "
          "in app/data/gulf_flights.txt and re-run prepare.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--last", type=int, default=None,
                         help="Only show the N most recent runs in the table.")
    parser.add_argument("--latest", action="store_true",
                         help="Print full JSON detail for the most recent run.")
    parser.add_argument("--excluded-flights", action="store_true",
                         help="List flights excluded by gulf_flights.txt, per the most recent run.")
    args = parser.parse_args()

    runs = load_runs()

    if args.latest:
        if not runs:
            print(f"No runs recorded yet in {LOG_PATH}.")
        else:
            print_detail(runs[-1])
    elif args.excluded_flights:
        if not runs:
            print(f"No runs recorded yet in {LOG_PATH}.")
        else:
            print_excluded_flights(runs[-1])
    else:
        to_show = runs[-args.last:] if args.last else runs
        print_table(to_show)

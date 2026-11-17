"""Merge completed staged RQ runs and regenerate the three comparison tables."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import resolve
from experiments.rq_merge import merge_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Repeat for disjoint original generation runs")
    parser.add_argument("--output", required=True, help="New directory for immutable merged reports")
    args = parser.parse_args()
    print(merge_runs([resolve(p) for p in args.run_dir], resolve(args.output)))


if __name__ == "__main__":
    main()

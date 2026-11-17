"""Export paired CSV/graphs from committed one-week records, even after a kill."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import resolve
from experiments.rq_live_reports import export_live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Generation directory containing manifest.json")
    parser.add_argument("--with-ci", action="store_true", help="Explicit 1000 group bootstrap; slower; results remain provisional")
    args = parser.parse_args()
    print(export_live(resolve(args.run_dir), with_ci=args.with_ci))


if __name__ == "__main__":
    main()

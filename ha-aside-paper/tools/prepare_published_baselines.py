"""Fetch pinned author Vanilla/ASIDE/ISE checkpoints and verify their provenance."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import resolve
from experiments.published_checkpoints import PINS, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", choices=list(PINS))
    parser.add_argument("--output-root", default="ha-aside-paper/checkpoints/published-aside")
    parser.add_argument("--download", action="store_true", help="Actually download full weights; otherwise preview only")
    args = parser.parse_args()
    methods = args.method or list(PINS)
    for method in methods:
        print(f"[published] {method}: {'download and checksum verification' if args.download else 'preview'}", flush=True)
        value = prepare(method, resolve(args.output_root) / method, download=args.download)
        print(json.dumps({k: value[k] for k in ["method", "repo_id", "revision", "download_bytes", "download_started", "fingerprint"] if k in value}, indent=2))


if __name__ == "__main__":
    main()

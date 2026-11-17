"""Freeze or verify the source-only one-week multi-role subset; no inference."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.rq_one_week import SELECTION, freeze_selection, verify_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    value = verify_subset() if args.verify else freeze_selection()
    print(json.dumps({"status": "passed", "selection": str(SELECTION),
                      "profile": value["profile"], "fingerprint": value["fingerprint"],
                      "parent_fingerprint": value["parent_selection_fingerprint"],
                      "counts_per_method": value["audit"]["counts"],
                      "no_model_outputs_used_for_selection": True}, indent=2))


if __name__ == "__main__":
    main()

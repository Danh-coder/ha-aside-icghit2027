"""Verify v5 code overlay, unchanged weekly data and plotting availability."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, file_hash
from experiments.rq_one_week import verify_subset
from experiments.rq_week_runner import BLOCK_GROUPS


def main():
    import matplotlib
    matplotlib.use("Agg")
    manifest = json.loads((ROOT / "deployment/one-week-live-update-manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Update file missing or changed: {relative}")
    subset = verify_subset()
    if subset["fingerprint"] != manifest["subset_fingerprint"]:
        raise ValueError("Frozen one-week subset differs")
    print(json.dumps({"status": "passed", "update_files": len(manifest["files"]),
                      "subset_fingerprint": subset["fingerprint"], "execution_order": "benchmark_blocks",
                      "source_groups_per_block": BLOCK_GROUPS, "generation_calls_per_method": 4448,
                      "matplotlib": matplotlib.__version__, "live_export": "paired CSV + PNG/SVG + HTML; atomic snapshots",
                      "model_weights_in_update": False}, indent=2))


if __name__ == "__main__":
    main()

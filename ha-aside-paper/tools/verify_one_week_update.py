"""Verify the one-week overlay, retained source data and deterministic subset."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, file_hash
from experiments.rq_one_week import verify_subset


def main():
    manifest = json.loads((ROOT / "deployment/one-week-update-manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Update file missing or changed: {relative}")
    subset = verify_subset()
    if (subset["identity"]["dataset_fingerprint"] != manifest["base_dataset_fingerprint"] or
            subset["parent_selection_fingerprint"] != manifest["base_selection_fingerprint"] or
            subset["fingerprint"] != manifest["subset_fingerprint"]):
        raise ValueError("Source data or subset differs from the released one-week protocol")
    print(json.dumps({"status": "passed", "update_files": len(manifest["files"]),
                      "profile": subset["profile"], "subset_fingerprint": subset["fingerprint"],
                      "counts_per_method": subset["audit"]["counts"],
                      "model_weights_in_update": False}, indent=2))


if __name__ == "__main__":
    main()

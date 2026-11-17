"""Verify the code-only public-checkpoint update and retained frozen data."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, PROJECT, file_hash
from experiments.rq_data import load_bundle
from experiments.rq_selection import verify_selection


def main():
    manifest = json.loads((ROOT / "deployment/published-update-manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Update file missing or changed: {relative}")
    prepared = PROJECT / "data/prepared/rq_public_v1"
    bundle = load_bundle(prepared)
    selection = verify_selection(json.loads((prepared / "selection.v2.json").read_text(encoding="utf-8")), bundle)
    if (bundle["fingerprint"] != manifest["base_dataset_fingerprint"] or
            selection["fingerprint"] != manifest["base_selection_fingerprint"]):
        raise ValueError("This update requires the verified RQ v2 data and selection")
    print(json.dumps({"status": "passed", "update_files": len(manifest["files"]),
                      "dataset_fingerprint": bundle["fingerprint"], "selection_fingerprint": selection["fingerprint"],
                      "model_weights_in_update": False}, indent=2))


if __name__ == "__main__":
    main()

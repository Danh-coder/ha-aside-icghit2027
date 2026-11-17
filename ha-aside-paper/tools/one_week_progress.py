"""Read-only generation progress and conditional ETA; no model loading."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import resolve
from experiments.rq_one_week import PROFILE
from experiments.rq_live_io import committed_rows


def progress(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if config.get("benchmark_profile") != PROFILE:
        raise ValueError("Select a one-week generation directory, not smoke/full/scoring output")
    selection = json.loads(resolve(config["selection_manifest"]).read_text(encoding="utf-8"))
    counts = selection["audit"]["counts"]
    totals = {"role_authority": counts["authority_generation_calls"],
              "task_controls": counts["task_controls"], "iheval": counts["iheval_requests"]}
    variants = [(name, False) for name in config["methods"]] + [(a["name"], True) for a in config.get("ablations", [])]
    result = []
    for name, ablation in variants:
        suites = {}
        for suite, planned in totals.items():
            if ablation and suite == "task_controls":
                continue
            observed = []
            path = run_dir / name / (suite + ".jsonl")
            partial = False
            if path.exists():
                rows, tail = committed_rows(path)
                partial = bool(tail["uncommitted_tail_bytes"])
                for row in rows:
                    observed.append(row)
                    observed.extend(row.get("interventions", {}).values())
            seconds = sum(r["seconds"] for r in observed)
            tokens = sum(r["generated_tokens"] for r in observed)
            mean = seconds / len(observed) if observed else None
            suites[suite] = {"completed_generation_calls": len(observed), "planned_generation_calls": planned,
                             "measured_generation_hours": seconds / 3600,
                             "mean_seconds_per_generation": mean,
                             "generated_tokens": tokens,
                             "remaining_generation_hours_if_same_mean": (planned - len(observed)) * mean / 3600 if mean is not None else None,
                             "partial_last_log_line": partial}
        feature_progress = run_dir / name / "probe_features/progress.json"
        result.append({"method": name, "suites": suites,
                       "cached_probe_progress": json.loads(feature_progress.read_text(encoding="utf-8")) if feature_progress.exists() else None,
                       "probe_complete": (run_dir / name / "role_probe.json").exists()})
    return {"profile": PROFILE, "methods": result,
            "limitations": "ETA is conditional per method/suite on recorded samples, excludes probing/CPU scoring/load time; "
                            "null means no timing evidence; unfinished cases are not counted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(progress(resolve(args.run_dir)), indent=2))


if __name__ == "__main__":
    main()

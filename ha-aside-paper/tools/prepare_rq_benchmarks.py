"""Fetch pinned public benchmarks, freeze source-derived cases, and export audit sheets."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import PROJECT, digest, write_json
from experiments.rq_sources import fetch_sources, verify_sources
from experiments.rq_data import prepare_bundle, mtrag_tasks, audit_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=PROJECT / "data/public_benchmarks/sources_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/prepared/rq_public_v1")
    parser.add_argument("--fetch", action="store_true", help="Download public source files; never model weights")
    parser.add_argument("--revisions", type=Path, help="JSON object of source name -> immutable Git revision")
    parser.add_argument("--rollout-review", type=Path, help="Completed public-conversation dependency audit")
    parser.add_argument("--export-audit", type=Path, help="Export source questions and pending rollout decisions")
    parser.add_argument("--verify", action="store_true", help="Verify a frozen bundle and all recorded public source pointers")
    args = parser.parse_args()
    revisions = json.loads(args.revisions.read_text(encoding="utf-8")) if args.revisions else None
    lock = fetch_sources(args.sources, revisions) if args.fetch else verify_sources(args.sources)
    if args.verify:
        print(json.dumps(audit_sources(args.output, args.sources), indent=2))
        return
    if args.export_audit:
        if args.export_audit.exists():
            raise FileExistsError("Audit files are never overwritten")
        tasks = mtrag_tasks(args.sources, lock)
        groups = {}
        for task in tasks:
            groups.setdefault(task["conversation_id"], {"eligible": None, "reason": "", "reviewer": "", "questions": []})
            groups[task["conversation_id"]]["questions"].append(task["question"])
        write_json(args.export_audit, {"conversation_set_sha256": digest(sorted(groups)), "conversations": groups})
    review = json.loads(args.rollout_review.read_text(encoding="utf-8")) if args.rollout_review else None
    manifest = prepare_bundle(args.sources, args.output, review=review)
    print(json.dumps({"output": str(args.output), "fingerprint": manifest["fingerprint"],
                      "suites": {k: v["n"] for k, v in manifest["suites"].items()},
                      "payloads": manifest["payloads"]["n"], "rollout_audit": manifest["rollout_review_status"]}, indent=2))


if __name__ == "__main__":
    main()

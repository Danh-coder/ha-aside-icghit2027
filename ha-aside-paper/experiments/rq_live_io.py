"""Durable records and source-complete groups for resumable weekly evaluation."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from .common import digest, resolve
from .rq_data import load_bundle, load_cases
from .rq_selection import verify_selection
from .rq_scoring import SCORER_VERSION


def selected_cases(config):
    bundle = load_bundle(resolve(config["rq_prepared_dir"]))
    selection = verify_selection(json.loads(resolve(config["selection_manifest"]).read_text(encoding="utf-8")), bundle)
    result = {}
    for suite in ["role_authority", "task_controls", "iheval"]:
        ids = set(selection["allowed"][suite])
        result[suite] = {c["case_id"]: c for c in load_cases(resolve(config["rq_prepared_dir"]), suite) if c["case_id"] in ids}
        if set(result[suite]) != ids:
            raise ValueError("Weekly selection contains absent or duplicate case IDs")
    return selection, result


def committed_rows(path, *, repair=False):
    """Ignore only an uncommitted tail; malformed committed rows always fail."""
    path = Path(path)
    raw = path.read_bytes() if path.exists() else b""
    end = raw.rfind(b"\n") + 1
    tail = raw[end:]
    if tail and repair:
        backup = path.with_name(path.name + ".uncommitted-" + hashlib.sha256(tail).hexdigest()[:16] + ".bin")
        if not backup.exists():
            backup.write_bytes(tail)
        with path.open("r+b") as stream:
            stream.truncate(end)
            stream.flush()
            os.fsync(stream.fileno())
    values = [json.loads(line) for line in raw[:end].split(b"\n") if line]
    return values, {"committed_sha256": hashlib.sha256(raw[:end]).hexdigest(),
                    "committed_bytes": end, "uncommitted_tail_bytes": len(tail)}


def validate_records(values, cases):
    indexed = {}
    for row in values:
        rid = row["record_id"]
        if rid in indexed or rid not in cases:
            raise ValueError("Duplicate or unexpected frozen record ID")
        case = cases[rid]
        if (row.get("case_id") != rid or row.get("case_sha256") != digest(case) or
                row.get("group_id") != case["group_id"] or row.get("metadata") != case["metadata"] or
                row.get("scenario") is not None or row.get("scores", {}).get("scorer_version") != SCORER_VERSION):
            raise ValueError("Saved record differs from its source case, gold or scorer version")
        if case["suite"] == "role_authority" and case["metadata"]["condition"] == "data":
            if set(row.get("interventions", {})) != {"user", "system"}:
                raise ValueError("DATA record lacks both completed encoding interventions")
            for role, result in row["interventions"].items():
                if result.get("encoding_intervention") != role or result.get("scores", {}).get("scorer_version") != SCORER_VERSION:
                    raise ValueError("Encoding intervention metadata differs")
        indexed[rid] = row
    return indexed


def append_durable(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def source_groups(cases):
    groups = defaultdict(set)
    for rid, case in cases.items():
        groups[case["group_id"]].add(rid)
    return dict(groups)


def complete_groups(records, cases):
    observed = set(records)
    return {g for g, ids in source_groups(cases).items() if ids <= observed}

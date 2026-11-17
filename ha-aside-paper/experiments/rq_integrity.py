"""Freeze completed generation bytes before judging or resuming a release run."""

import json
from pathlib import Path

from .common import digest, file_hash, write_json


def generation_inventory(run_dir, config):
    run_dir = Path(run_dir)
    names = config["methods"] + [v["name"] for v in config.get("ablations", [])]
    files = {}
    for name in names:
        for path in sorted((run_dir / name).glob("*.jsonl")):
            files[path.relative_to(run_dir).as_posix()] = file_hash(path)
        probe = run_dir / name / "role_probe.json"
        if probe.exists():
            files[probe.relative_to(run_dir).as_posix()] = file_hash(probe)
    return {"parent_manifest_sha256": file_hash(run_dir / "manifest.json"), "files": files}


def freeze_generation(run_dir, config):
    run_dir = Path(run_dir)
    inventory = generation_inventory(run_dir, config)
    path = run_dir / "generation_integrity.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != inventory:
            raise ValueError("Completed generation outputs changed; do not overwrite the immutable run")
    else:
        write_json(path, inventory)
    return inventory


def verify_generation(run_dir, config):
    run_dir = Path(run_dir)
    path = run_dir / "generation_integrity.json"
    if not path.exists():
        raise ValueError("Generation has not completed/frozen its output checksums")
    if json.loads(path.read_text(encoding="utf-8")) != generation_inventory(run_dir, config):
        raise ValueError("Generation outputs are missing, modified, or belong to another manifest")


def guard_resume(config, report):
    from .rq_runner import runtime_manifest
    from .common import resolve
    run_dir = resolve(config["output_root"]) / digest(runtime_manifest(config, report))
    if (run_dir / "generation_integrity.json").exists():
        verify_generation(run_dir, config)

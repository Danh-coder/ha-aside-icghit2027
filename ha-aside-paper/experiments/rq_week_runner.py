"""Benchmark-major, matched-block scheduling with durable outputs/live snapshots."""

from collections import defaultdict
from contextlib import contextmanager
import gc
import itertools
import json
import os
from pathlib import Path
import signal

import torch

from .common import digest, file_hash, resolve, write_json
from .checkpoints import load_checkpoint
from .rq_one_week import rank
from .rq_live_io import selected_cases, committed_rows, validate_records, append_durable, source_groups
from .rq_runner import runtime_manifest, variants, evaluate_static, source_root

BLOCK_GROUPS = {"role_authority": 10, "task_controls": 100, "iheval": 50}


def blocks(suite, cases, helper_ids):
    groups = source_groups(cases)
    helper_set = set(helper_ids) if suite == "iheval" else set()
    helper_rows = [cases[rid] for rid in sorted(helper_set)]
    groups = {g: ids for g, ids in groups.items() if not ids <= helper_set}
    if suite == "iheval":
        by_task = defaultdict(list)
        for group, ids in groups.items():
            task = cases[next(iter(ids))]["metadata"]["task"]
            by_task[task].append(group)
        lists = [sorted(by_task[t], key=lambda g: rank("schedule", suite, g)) for t in sorted(by_task)]
        ordered = [g for entries in itertools.zip_longest(*lists) for g in entries if g is not None]
    else:
        ordered = sorted(groups, key=lambda g: rank("schedule", suite, g))
    size = BLOCK_GROUPS[suite]
    for offset in range(0, len(ordered), size):
        current = ordered[offset:offset + size]
        rows = [cases[rid] for group in current for rid in sorted(groups[group])]
        if offset == 0:
            rows = helper_rows + rows
        yield offset // size, rows


@contextmanager
def stop_signals():
    previous = {}
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Stop requested by signal {signum}")
    try:
        for name in ["SIGINT", "SIGTERM"]:
            if hasattr(signal, name):
                signum = getattr(signal, name)
                previous[signum] = signal.signal(signum, stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def generation_lock(output):
    """OS lock releases on crashes; a read-only exporter does not take it."""
    path = Path(output) / "generation.lock"
    with path.open("a+b") as stream:
        if not path.stat().st_size:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run(config, report):
    if not report["ready"]:
        raise ValueError("Resolve every preflight error before running")
    manifest = runtime_manifest(config, report)
    output = resolve(config["output_root"]) / digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Resume manifest differs")
    write_json(path, manifest)
    print("RUN_DIR=" + str(output), flush=True)
    if report.get("selection_sha256") and file_hash(resolve(config["selection_manifest"])) != report["selection_sha256"]:
        raise ValueError("Selection bytes changed after preflight")
    selection, cases = selected_cases(config)
    if selection["fingerprint"] != report["selection_fingerprint"]:
        raise ValueError("Selection changed after preflight")
    from .rq_week_import import copy_imports
    copy_imports(config, report, output, (selection, cases))
    from .rq_data import load_bundle
    bundle = load_bundle(resolve(config["rq_prepared_dir"]))
    settings = {**config, "_source_root": str(source_root(config, bundle))}
    options = list(variants(config))
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["dtype"]]
    cached = {}
    summary = {"run_id": output.name, "status": "in_progress", "execution_order": "benchmark_blocks", "models": {}}
    for variant in options:
        name = variant["name"]
        summary["models"][name] = {}
        for suite in cases:
            values, _ = committed_rows(output / name / f"{suite}.jsonl", repair=True)
            cached[(name, suite)] = validate_records(values, cases[suite])
            summary["models"][name][suite] = {"records": len(cached[(name, suite)]), "planned_records": len(cases[suite]),
                "status": "complete" if len(cached[(name, suite)]) == len(cases[suite]) else "partial"}
        probe_path = output / name / "role_probe.json"
        if probe_path.exists():
            summary["models"][name]["role_probe"] = {"status": "complete", "examples": json.loads(probe_path.read_text(encoding="utf-8"))["n_examples"]}
    from .rq_live_reports import export_live
    write_json(output / "summary.json", summary)
    export_live(output, context=(selection, cases))
    try:
        with stop_signals():
            for suite in config["suites"]:
                if suite == "role_probe":
                    stages = [(0, None)]
                else:
                    stages = list(blocks(suite, cases[suite], selection["audit"]["iheval_reference_helper_ids"]))
                for block_id, current in stages:
                    for variant in options:
                        name = variant["name"]
                        if variant["ablation"] and suite == "task_controls":
                            continue
                        probe_path = output / name / "role_probe.json"
                        pending = None if suite == "role_probe" else [c for c in current if c["case_id"] not in cached[(name, suite)]]
                        if (suite == "role_probe" and probe_path.exists()) or pending == []:
                            continue
                        write_json(output / "active.json", {"method": name, "suite": suite, "block": block_id, "status": "running"})
                        print(f"[BLOCK] {suite} block={block_id + 1} method={name}", flush=True)
                        model, tokenizer, _ = load_checkpoint(variant["checkpoint"], variant["method"], "qwen25_7b", dtype, config["device"])
                        try:
                            if suite == "role_probe":
                                from analysis.probe_four_roles import probe
                                probe_settings = {**settings, "_probe_cache_dir": str(output / name / "probe_features"),
                                                  "_probe_checkpoint_signature": digest(report["checkpoints"][name])}
                                write_json(probe_path, probe(model, tokenizer, list(cases["role_authority"].values()), probe_settings))
                                summary["models"][name][suite] = {"status": "complete", "examples": 400}
                            else:
                                for case in pending:
                                    rid = case["case_id"]
                                    result = evaluate_static(model, tokenizer, case, settings)
                                    row = {**result, "record_id": rid, "case_id": rid, "case_sha256": digest(case),
                                           "group_id": case["group_id"], "suite": suite, "scenario": None,
                                           "metadata": case["metadata"], "benchmark_kind": case["benchmark_kind"]}
                                    validate_records([row], cases[suite])
                                    # A DATA record commits only after both diagnostics and scores finish.
                                    append_durable(output / name / f"{suite}.jsonl", row)
                                    cached[(name, suite)][rid] = row
                                summary["models"][name][suite] = {"records": len(cached[(name, suite)]),
                                    "planned_records": len(cases[suite]), "status": "complete" if len(cached[(name, suite)]) == len(cases[suite]) else "partial"}
                        finally:
                            del model, tokenizer
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        write_json(output / "summary.json", summary)
                        export_live(output, context=(selection, cases))
            from .rq_reporting import export_reports
            export_reports(output, config)
            summary["status"] = "complete_generation_and_selected_scoring"
            write_json(output / "summary.json", summary)
            write_json(output / "active.json", {"status": "complete"})
            export_live(output, context=(selection, cases))
            return output
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed_not_scored_as_defense"
        write_json(output / "summary.json", summary)
        write_json(output / "failure.json", {"status": summary["status"], "error": repr(error),
                                             "resume": "Run the identical config/code/checkpoints again; completed records remain committed"})
        try:
            export_live(output, context=(selection, cases))
        except Exception as export_error:
            write_json(output / "live_export_error.json", {"error": repr(export_error), "raw_records_preserved": True})
        print("[STOP] Records preserved. Partial export: " + str(output / "live/index.html"), flush=True)
        raise


def run(config, report):
    if not report["ready"]:
        raise ValueError("Resolve every preflight error before running")
    output = resolve(config["output_root"]) / digest(runtime_manifest(config, report))
    output.mkdir(parents=True, exist_ok=True)
    with generation_lock(output):
        return _run(config, report)

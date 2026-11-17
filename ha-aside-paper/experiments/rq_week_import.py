"""Explicit reuse of stopped v4/v5 records with identical inference semantics."""

import ast
import hashlib
import json
from pathlib import Path

from .common import ROOT, digest, file_hash, resolve, write_json
from .rq_one_week import PROFILE
from .rq_live_io import committed_rows, validate_records, selected_cases, append_durable
from .rq_runner import runtime_manifest

IGNORED = {"methods", "checkpoints", "checkpoint_training_recipes", "ablations", "checkpoint_policy", "output_root",
           "rq_prepared_dir", "source_root", "prepared_dir", "selection_manifest", "parent_selection_manifest",
           "base_revisions", "tokenizer_revisions", "execution_order", "import_runs"}


def generation_ast_hash(function):
    # Python 3.12 adds empty type_params fields absent in the server's 3.10.
    # Normalize syntax fields rather than hashing version-specific ast.dump().
    def normalize(value):
        if isinstance(value, ast.AST):
            return {"node": type(value).__name__, "fields": {name: normalize(part) for name, part in ast.iter_fields(value) if name != "type_params"}}
        if isinstance(value, (list, tuple)):
            return [normalize(part) for part in value]
        return value
    return digest(normalize(function))


def inspect_imports(config, report, *, context=None):
    if not config.get("import_runs"):
        return []
    selection, cases = context if context is not None else selected_cases(config)
    contract = json.loads(Path(__file__).with_name("one_week_import_contract.v1.json").read_text(encoding="utf-8"))
    container = Path(__file__).with_name("run_benchmark.py")
    fn = next(n for n in ast.parse(container.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    if generation_ast_hash(fn) != contract["generate_ast_sha256"]:
        raise ValueError("Generation semantics changed; historical output reuse is disabled")
    expected_config = {k:v for k,v in config.items() if k not in IGNORED}
    current_environment = runtime_manifest(config, report)["environment"]
    names = config["methods"] + [a["name"] for a in config.get("ablations", [])]
    imports = []
    for location in config["import_runs"]:
        run = resolve(location).resolve()
        parent_path = run / "manifest.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if parent["config"].get("benchmark_profile") != PROFILE:
            raise ValueError("Import requires a compatible frozen one-week run")
        if {k:v for k,v in parent["config"].items() if k not in IGNORED} != expected_config:
            raise ValueError("Historical evaluation/decoding configuration differs")
        if parent["environment"] != current_environment:
            raise ValueError("Historical inference environment differs")
        if parent["preflight"].get("runtime") != report.get("runtime"):
            raise ValueError("Historical GPU/runtime settings differ")
        if parent["preflight"]["selection_fingerprint"] != selection["fingerprint"]:
            raise ValueError("Historical subset fingerprint differs")
        old_hashes = {name.replace("\\", "/"): value for name, value in parent["source_sha256"].items()}
        for relative, expected in contract["unchanged_core_sources"].items():
            if old_hashes.get(relative) != expected or file_hash(ROOT / relative) != expected:
                raise ValueError("Historical tokenizer/model/transcript/scorer core differs: " + relative)
        if old_hashes.get("ha-aside-paper/experiments/run_benchmark.py") not in {contract["v4_generate_container_sha256"], file_hash(container)}:
            raise ValueError("Unknown historical generation implementation")
        probe_file = ROOT / "ha-aside-paper/analysis/probe_four_roles.py"
        if old_hashes.get("ha-aside-paper/analysis/probe_four_roles.py") not in {contract["v4_probe_container_sha256"], file_hash(probe_file)}:
            raise ValueError("Unknown historical probe implementation")
        if (run / "generation_integrity.json").exists():
            from .rq_integrity import verify_generation
            verify_generation(run, parent["config"])
        files = {}
        for name in names:
            old = parent["preflight"]["checkpoints"].get(name)
            if old is None:
                continue
            new = report["checkpoints"][name]
            if old["metadata"] != new["metadata"] or old["artifact_sha256"] != new["artifact_sha256"]:
                raise ValueError("Historical checkpoint weights/tokenizer/encoding metadata differs: " + name)
            for suite, expected_cases in cases.items():
                path = run / name / f"{suite}.jsonl"
                values, info = committed_rows(path)
                validate_records(values, expected_cases)
                if values:
                    files[f"{name}/{suite}.jsonl"] = info
            path = run / name / "role_probe.json"
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                expected_groups = {c["group_id"] for c in cases["role_authority"].values()}
                if set(value["source_groups"]) != expected_groups or value["n_examples"] != 400:
                    raise ValueError("Historical probe cohort differs")
                files[f"{name}/role_probe.json"] = {"sha256": file_hash(path)}
        imports.append({"run_dir": str(run), "parent_manifest_sha256": file_hash(parent_path), "files": files})
    return imports


def copy_imports(config, report, output, context):
    if not config.get("import_runs"):
        return
    planned = inspect_imports(config, report, context=context)
    if planned != report.get("imported_runs"):
        raise ValueError("Stopped import source changed after preflight")
    _, cases = context
    for origin in planned:
        for relative in origin["files"]:
            source, target = Path(origin["run_dir"]) / relative, Path(output) / relative
            if source == target:
                raise ValueError("Import source and target must be separate runs")
            if relative.endswith(".jsonl"):
                suite = source.stem
                old = validate_records(committed_rows(source)[0], cases[suite])
                current = validate_records(committed_rows(target, repair=True)[0], cases[suite])
                for rid, row in old.items():
                    if rid in current:
                        if digest(current[rid]) != digest(row):
                            raise ValueError("Conflicting historical output for the same method/case")
                    else:
                        append_durable(target, row)
            else:
                value = json.loads(source.read_text(encoding="utf-8"))
                if target.exists() and json.loads(target.read_text(encoding="utf-8")) != value:
                    raise ValueError("Conflicting historical probe output")
                write_json(target, value)
    write_json(Path(output) / "import_provenance.json", {"imports": planned, "policy": "explicit source/checkpoint/environment-equivalent reuse; no output selection by score"})

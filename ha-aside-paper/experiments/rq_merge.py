"""Combine disjoint completed method runs without regenerating model responses."""

import copy
import json
from pathlib import Path
import shutil

from .common import ROOT, digest, file_hash, write_json
from .rq_integrity import generation_inventory, verify_generation
from .rq_sources import rows
from .rq_staging import METHODS, ABLATION, comparison_status, pairing_errors


def evaluation_identity(parent):
    config = copy.deepcopy(parent["config"])
    smoke = config.pop("preset") == "smoke"
    for field in ["methods", "checkpoints", "checkpoint_training_recipes", "ablations", "checkpoint_policy",
                  "output_root", "rq_prepared_dir", "source_root", "prepared_dir", "selection_manifest",
                  "base_revisions", "tokenizer_revisions", "device"]:
        config.pop(field, None)
    preflight = parent["preflight"]
    return {"protocol": parent["protocol"], "smoke": smoke, "evaluation_config": config,
            "dataset_fingerprint": preflight["dataset_fingerprint"],
            "selection_fingerprint": preflight["selection_fingerprint"],
            "selection_sha256": preflight["selection_sha256"], "datasets": preflight["datasets"],
            "source_sha256": parent["source_sha256"], "environment": parent["environment"],
            "scorer_version": parent["scorer_version"], "attack_scope": parent["attack_scope"]}


def completed_scores(run_dir, parent):
    """Read only hash-verified completed phases; leave pending metrics pending."""
    index_path = run_dir / "scoring_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    mappings = {"judge": {}, "utility": {}, "attack_review": {}}
    phases = {}
    expected_logs = {k: v for k, v in generation_inventory(run_dir, parent["config"])["files"].items()
                     if k.endswith(".jsonl")}
    for kind, relative in index.items():
        if kind not in mappings:
            raise ValueError("Unknown scoring phase")
        phase = (run_dir / relative).resolve()
        if not phase.is_relative_to(run_dir.resolve() / "scores"):
            raise ValueError("Scoring phase escapes its run")
        manifest_path, summary_path, scores_path = [phase / f for f in ["manifest.json", "summary.json", "scores.jsonl"]]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("status") != "complete" or summary.get("identity") != manifest
                or summary.get("scores_sha256") != file_hash(scores_path)
                or manifest["parent_manifest_sha256"] != file_hash(run_dir / "manifest.json")
                or manifest["generation_logs"] != expected_logs):
            raise ValueError("Scoring phase is incomplete, modified, or belongs to another generation")
        if kind in {"judge", "utility"} and not manifest.get(kind):
            raise ValueError("Scoring index points to a phase with another scoring scope")
        if kind == "attack_review" and not manifest.get("attack_review_sha256"):
            raise ValueError("Missing attack adjudication provenance")
        records = rows(scores_path)
        if len(records) != summary["n_scores"] or len({r["record_id"] for r in records}) != len(records):
            raise ValueError("Incomplete or duplicate scoring records")
        for record in records:
            if record["kind"] != kind:
                continue
            key = (record["model"], record["parent_record_id"], record["turn"])
            mappings[kind][key] = record[{"judge": "verdict", "utility": "score", "attack_review": "decision"}[kind]]
            if kind == "attack_review":
                mappings[kind][key] = mappings[kind][key]["attack_success"]
        phases[kind] = {"manifest": manifest, "manifest_sha256": file_hash(manifest_path),
                        "scores_sha256": file_hash(scores_path)}
    return mappings, phases


def cohort_signature(path):
    if path.name == "role_probe.json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return digest(sorted((r["group_id"], r["gold"]) for r in value["layers"][0]["test_predictions"]))
    observations = []
    for row in rows(path):
        observations.append({"id": row.get("record_id", row.get("sample_id")),
                             "case": row.get("case_sha256"), "scenario": row.get("scenario"),
                             "example": row.get("example")})
    return digest(sorted(observations, key=lambda r: r["id"]))


def merge_runs(run_dirs, output):
    if len(run_dirs) < 2:
        raise ValueError("Provide at least two disjoint completed runs")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Merged reports are immutable; choose a new output directory")
    parents, owners, checkpoints, cohorts, scoring_protocols = [], {}, {}, {}, {}
    method_files = None
    score_maps = {"judge": {}, "utility": {}, "attack_review": {}}
    identity = None
    for value in run_dirs:
        run_dir = Path(value).resolve()
        parent = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if parent.get("kind") == "merged_rq_evaluation_no_generation":
            raise ValueError("Merge original generation runs, not previously merged reports")
        config = parent["config"]
        verify_generation(run_dir, config)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if not summary["status"].startswith("complete_generation") or parent["preflight"].get("errors"):
            raise ValueError("Finish generation successfully before merging")
        current = evaluation_identity(parent)
        if identity is not None and current != identity:
            raise ValueError("Runs use different benchmark protocols, cohorts, code, or environments")
        identity = current
        for relative, expected in parent["source_sha256"].items():
            if file_hash(ROOT / relative) != expected:
                raise ValueError("Merge/report code differs from the evaluated source revision")
        mappings, phases = completed_scores(run_dir, parent)
        for kind, phase in phases.items():
            scoring = phase["manifest"]
            # A combined judge+utility phase and two separate phases are
            # compatible when their effective scorer identities are identical.
            signature = {k: scoring.get(k) for k in ["scorer_version", "source_sha256", "package_versions"]}
            if kind == "judge":
                signature["judge_model"] = scoring["judge_model"]
            if kind == "utility":
                signature.update(utility_model_revision=scoring["utility_model_revision"], scoring_device=scoring["scoring_device"])
            if kind in scoring_protocols and scoring_protocols[kind] != signature:
                raise ValueError("Runs use different scoring protocols")
            scoring_protocols[kind] = signature
            score_maps[kind].update(mappings[kind])
        names = config["methods"] + [v["name"] for v in config.get("ablations", [])]
        for name in names:
            if name in owners:
                raise ValueError(f"Duplicate method {name}; select one checkpoint before evaluating test data")
            if name not in parent["preflight"]["checkpoints"]:
                raise ValueError("Missing checkpoint provenance")
            owners[name] = (run_dir, config)
            checkpoints[name] = parent["preflight"]["checkpoints"][name]
            filenames = {Path(r).name for r in generation_inventory(run_dir, config)["files"] if Path(r).parent.name == name}
            if name != ABLATION:
                if method_files is not None and filenames != method_files:
                    raise ValueError("Methods have different generated suite coverage")
                method_files = filenames
            for relative in generation_inventory(run_dir, config)["files"]:
                path = run_dir / relative
                if path.parent.name != name:
                    continue
                signature = cohort_signature(path)
                if path.name in cohorts and cohorts[path.name] != signature:
                    raise ValueError(f"Method cohorts differ in {path.name}")
                cohorts[path.name] = signature
        parents.append({"path": str(run_dir), "manifest_sha256": file_hash(run_dir / "manifest.json"),
                        "generation_integrity_sha256": file_hash(run_dir / "generation_integrity.json"),
                        "scoring_phases": phases, "manifest": parent})
    if parents[0]["manifest"]["config"].get("allow_published_checkpoints"):
        from .rq_published import compatible_backbones
        compatible_backbones(checkpoints)
    else:
        backbone_ids = {tuple(c["metadata"]["provenance"][k] for k in ["base_model", "tokenizer_id", "base_revision", "tokenizer_revision"])
                        for c in checkpoints.values()}
        if len(backbone_ids) != 1:
            raise ValueError("Method checkpoints use different pinned Base/tokenizer identities")
    if pairing_errors(checkpoints):
        raise ValueError("; ".join(pairing_errors(checkpoints)))
    config = copy.deepcopy(parents[0]["manifest"]["config"])
    config["preset"] = "smoke" if identity["smoke"] else "staged"
    config["methods"] = [m for m in METHODS if m in owners]
    config["ablations"] = [v for _, cfg in owners.values() for v in cfg.get("ablations", []) if v["name"] == ABLATION][:1]
    config["checkpoint_policy"] = "completed_trial" if any(c.get("training_status", {}).get("status") == "completed_trial_search_incomplete" for c in checkpoints.values()) else "validation_selected"
    for name, (_, cfg) in owners.items():
        if name in METHODS:
            config["checkpoints"]["qwen25_7b"][name] = cfg["checkpoints"]["qwen25_7b"][name]
            config["checkpoint_training_recipes"]["qwen25_7b"][name] = cfg["checkpoint_training_recipes"]["qwen25_7b"][name]
    preflight = {**parents[0]["manifest"]["preflight"], "checkpoints": checkpoints, "errors": [], "ready": True}
    preflight["comparison"] = comparison_status(config, checkpoints)
    manifest = {**parents[0]["manifest"], "kind": "merged_rq_evaluation_no_generation", "config": config,
                "preflight": preflight, "evaluation_identity": identity, "parents": parents}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    for name, (run_dir, cfg) in owners.items():
        (output / name).mkdir()
        for relative in generation_inventory(run_dir, cfg)["files"]:
            path = run_dir / relative
            if path.parent.name == name:
                shutil.copyfile(path, output / relative)
    from .rq_reporting import export_reports
    export_reports(output, config, judge_scores=score_maps["judge"], utility_scores=score_maps["utility"],
                   attack_reviews=score_maps["attack_review"])
    write_json(output / "summary.json", {"status": "complete_merged_reports", "comparison": preflight["comparison"],
                                         "interpretation": "No defender generation or API calls performed during merge"})
    return output

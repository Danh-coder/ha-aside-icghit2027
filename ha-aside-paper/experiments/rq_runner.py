"""RQ1--RQ3 evaluation, with frozen source cases and model-local histories."""

import copy
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import time

import torch

from .common import PROJECT, ROOT, REGISTRY, digest, file_hash, resolve, write_json
from .checkpoints import checkpoint_metadata, load_checkpoint
from .rq_data import REVISION, NEW_SUITES, load_bundle, load_cases
from .rq_sources import verify_sources, rows
from .rq_transcript import POLICY, assistant, encode_transcript, message, rag_user, segment
from .rq_scoring import score_case, score_attack, sysbench_module, SCORER_VERSION

METHODS = ["vanilla", "aside_orig", "ise", "ise_4role", "ha_aside"]
CONTROLS = {"sep", "struq_id", "struq_ood", "bipia_text", "bipia_code", "clean_qa"}


def validate_config(config):
    c = copy.deepcopy(config)
    if c["protocol"] != REVISION or c.get("preset") not in {"reference", "staged", "smoke"}:
        raise ValueError("Unknown RQ protocol/preset")
    if c.get("models") != ["qwen25_7b"]:
        raise ValueError("RQ v1 compares the same Qwen2.5-7B backbone")
    if not c.get("methods") or len(c["methods"]) != len(set(c["methods"])) or set(c["methods"]) - set(METHODS):
        raise ValueError("Unknown RQ comparison method")
    if c["preset"] == "reference" and set(c["methods"]) != set(METHODS):
        raise ValueError("Paper RQ comparison requires all five methods")
    c.setdefault("checkpoint_policy", "validation_selected")
    if c["checkpoint_policy"] not in {"validation_selected", "completed_trial"}:
        raise ValueError("Unknown checkpoint evaluation policy")
    if c["preset"] == "reference" and c["checkpoint_policy"] != "validation_selected":
        raise ValueError("Reference requires validation-selected checkpoints; use staged for completed trials")
    if c.get("enable_rgag") is not False:
        raise ValueError("RQ protocol requires RGAG OFF in both training and evaluation")
    if not c.get("suites") or set(c["suites"]) - (NEW_SUITES | CONTROLS):
        raise ValueError("Unknown RQ suite")
    if c["preset"] != "smoke" and c.get("sample_limit") is not None:
        raise ValueError("Reference case selection is the frozen manifest, not an inference-time sample")
    if c.get("max_input_tokens") != 16384 or c.get("max_new_tokens") != 1024:
        raise ValueError("RQ v1 uses a 16384 context budget and 1024 default generation tokens")
    if c.get("history_modes", ["fixed", "own"]) not in [["fixed", "own"], ["fixed"], ["own"]]:
        raise ValueError("Unknown history modes")
    if c.get("split", "test") not in {"dev", "test"}:
        raise ValueError("Select a frozen dev or test conversation split")
    if c["preset"] != "smoke" and c.get("split", "test") != "test":
        raise ValueError("Paper evaluation uses the locked test split")
    c.setdefault("history_modes", ["fixed", "own"])
    c.setdefault("split", "test")
    c.setdefault("seed", 42)
    c.setdefault("bootstrap_samples", 1000)
    if type(c["bootstrap_samples"]) is not int or c["bootstrap_samples"] < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if c["preset"] != "smoke" and (c["seed"] != 42 or c["bootstrap_samples"] != 1000):
        raise ValueError("Reference statistics use 1000 group bootstrap samples and seed 42")
    if c["preset"] != "smoke" and c.get("probe_group_limit") is not None:
        raise ValueError("Reference probing uses all audited source-clause groups")
    c.setdefault("seeds", [2024, 2025, 2026])
    c.setdefault("sep_split", "full")
    c.setdefault("bipia_attack_filter", "aside_source_regex")
    c.setdefault("direct_top_k", 50)
    c.setdefault("interventions", ["user", "system"])
    if set(c["interventions"]) - {"user", "system"}:
        raise ValueError("Unknown payload encoding intervention")
    if c.get("probe_feature", "clause_mean") != "clause_mean":
        raise ValueError("RQ v1 probe uses content-only matched-clause mean features")
    names = [v["name"] for v in c.get("ablations", [])]
    if len(names) != len(set(names)) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names):
        raise ValueError("Ablation names must be unique plain identifiers")
    if c["preset"] == "reference" and names != ["ha_no_user_rotation"]:
        raise ValueError("The full RQ1 comparison requires its separately trained USER-angle=0 ablation")
    if any(name != "ha_no_user_rotation" or value.get("method") != "ha_aside"
           for name, value in zip(names, c.get("ablations", []))):
        raise ValueError("RQ v1 supports only the declared HA USER-angle=0 ablation")
    if c["dtype"] not in {"float32", "bfloat16"} or (c["device"] == "cpu" and c["dtype"] != "float32"):
        raise ValueError("CPU RQ tests require float32; GPU evaluation supports bfloat16")
    return c


def variants(config):
    for method in config["methods"]:
        yield {"name": method, "method": method, "checkpoint": config["checkpoints"]["qwen25_7b"].get(method, ""),
               "ablation": False}
    for variant in config.get("ablations", []):
        if variant["name"] in METHODS or variant.get("method") != "ha_aside":
            raise ValueError("Ablation names must be separate HA variants")
        yield {**variant, "ablation": True}


def source_root(config, bundle):
    return resolve(config.get("source_root", bundle["source_root"]))


def preflight(config):
    report = {"protocol": REVISION, "errors": [], "datasets": {}, "checkpoints": {}}
    try:
        bundle = load_bundle(resolve(config["rq_prepared_dir"]))
        actual = verify_sources(source_root(config, bundle))
        if digest(actual) != digest(bundle["source_lock"]):
            raise ValueError("Public source snapshot differs from the frozen benchmark")
        report["dataset_fingerprint"] = bundle["fingerprint"]
        report["datasets"] = bundle["suites"]
        from .rq_selection import verify_selection
        selection_path = resolve(config["selection_manifest"])
        selection = verify_selection(json.loads(selection_path.read_text(encoding="utf-8")), bundle)
        report["selection_fingerprint"] = selection["fingerprint"]
        report["selection_sha256"] = file_hash(selection_path)
        report["selection_coverage"] = selection["coverage"]
        report["rollout_review_status"] = bundle["rollout_review_status"]
        if "four_role_multiturn" in config["suites"] and "own" in config["history_modes"] and bundle["rollout_review_status"] != "complete":
            raise ValueError("Own-history requires the frozen source-question dependency audit")
        if any(s in config["suites"] for s in ["iheval", "role_authority", "task_controls"]):
            from .rq_scoring import iheval_scorers, rule_registry
            iheval_scorers(str(source_root(config, bundle)))
            rule_registry(str(source_root(config, bundle)))
            import nltk
            nltk.data.load("nltk:tokenizers/punkt/english.pickle")
    except Exception as error:
        report["errors"].append(f"datasets/scorers: {type(error).__name__}: {error}")
    from .benchmark_data import load_suite
    from .run_benchmark import suite_seeds
    for suite in set(config["suites"]) & CONTROLS:
        for seed in suite_seeds(config, suite):
            try:
                cases = load_suite(suite, seed, resolve(config["prepared_dir"]), config["sep_split"], config.get("sample_limit"),
                                   bipia_filter=config["bipia_attack_filter"])
                report["datasets"][f"{suite}:{seed}"] = {"n": len(cases), "sha256": digest([c.serializable() for c in cases])}
            except Exception as error:
                report["errors"].append(f"{suite}:{seed}: {type(error).__name__}: {error}")
    identities, metadata_by_name = [], {}
    for variant in variants(config):
        try:
            if not variant["checkpoint"]:
                raise ValueError("A separately trained checkpoint is required")
            path = resolve(variant["checkpoint"])
            metadata = checkpoint_metadata(path, variant["method"], "qwen25_7b")
            provenance = metadata["provenance"]
            from .rq_staging import checkpoint_status
            training_status = checkpoint_status(path, metadata, config.get("checkpoint_policy", "validation_selected"))
            expected_recipe = config["checkpoint_training_recipes"]["qwen25_7b"].get(variant["method"])
            if provenance["training_recipe"] != expected_recipe:
                raise ValueError("Checkpoint SFT recipe differs from declared provenance")
            if variant["method"] == "ha_aside" and metadata["enable_rgag"]:
                raise ValueError("Checkpoint was trained with RGAG ON")
            if variant["method"] == "ha_aside":
                angles = {str(k): float(v) for k, v in metadata["angles"].items()}
                expected = {"0": 0.0, "1": 0.0 if variant["ablation"] else math.pi / 8, "2": math.pi / 2, "3": 0.0}
                if angles != expected or metadata["rotation_direction"] != "right":
                    raise ValueError("HA angles/direction differ from the declared main/ablation condition")
            for key in ["base_revision", "tokenizer_revision"]:
                if not re.fullmatch(r"[0-9a-f]{40}", provenance[key]):
                    raise ValueError("Checkpoint does not pin immutable Base/tokenizer identities")
                declared = config.get(key + "s", {}).get("qwen25_7b")
                if declared and declared != provenance[key]:
                    raise ValueError(f"Checkpoint differs from configured {key}")
            identities.append(tuple(provenance[k] for k in ["base_model", "tokenizer_id", "base_revision", "tokenizer_revision"]))
            metadata_by_name[variant["name"]] = metadata
            report["checkpoints"][variant["name"]] = {
                "path": str(path), "metadata": metadata, "training_status": training_status,
                "artifact_sha256": {p.name: file_hash(p) for p in sorted(path.iterdir())
                                    if p.is_file() and (p.suffix in {".safetensors", ".json"} or p.name in {"ise_offset.pt", "segment_embeddings.pt"})}}
        except Exception as error:
            report["errors"].append(f"{variant['name']}: {type(error).__name__}: {error}")
    if len(set(identities)) > 1:
        report["errors"].append("Five methods do not share the same pinned Base/tokenizer")
    from .rq_staging import comparison_status, pairing_errors
    report["errors"].extend(pairing_errors(report["checkpoints"]))
    report["comparison"] = comparison_status(config, report["checkpoints"])
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        report["errors"].append("CUDA is unavailable; run model inference on the GPU server")
    report["ready"] = not report["errors"]
    return report


def generate_transcript(model, tokenizer, messages, config, *, max_new_tokens=1024, tool=None,
                        encoding_intervention=None):
    encoded = encode_transcript(tokenizer, messages, config["max_input_tokens"], tool=tool,
                                source_root=config["_source_root"], encoding_intervention=encoding_intervention)
    if encoded["input_ids"].shape[1] + max_new_tokens > config["max_input_tokens"]:
        raise ValueError("Prompt plus generation exceeds the shared context budget; no truncation")
    from .run_benchmark import generate
    settings = {**config, "max_new_tokens": max_new_tokens}
    result = generate(model, tokenizer, encoded, settings, "rq_greedy", config["seed"])
    result["encoding_intervention"] = encoding_intervention
    return result


def evaluate_static(model, tokenizer, case, config):
    result = generate_transcript(model, tokenizer, case["messages"], config,
                                 max_new_tokens=case.get("max_new_tokens", 1024), tool=case.get("tool"))
    result["scores"] = score_case(case, result["output"], config["_source_root"])
    if case["suite"] == "role_authority" and case.get("metadata", {}).get("condition") == "data":
        result["interventions"] = {}
        for role in config["interventions"]:
            diagnostic = generate_transcript(model, tokenizer, case["messages"], config, encoding_intervention=role)
            diagnostic["scores"] = score_case(case, diagnostic["output"], config["_source_root"])
            result["interventions"][role] = diagnostic
    return result


def evaluate_sysbench(model, tokenizer, case, config):
    raw = case["raw"]
    history, plain_history, turns = [], [], []
    source = sysbench_module(config["_source_root"])
    for original in raw["messages"]:
        if original["role"] in {"system", "user"}:
            history.append(message(original["role"], segment(original["content"], original["role"])))
            plain_history.append(original.copy())
            continue
        current = generate_transcript(model, tokenizer, history, config)
        history.append(assistant(current["output"]))
        plain_history.append({"role": "assistant", "content": current["output"]})
        criteria = raw["prompt_infos"][plain_history[-2]["content"]]["criteria"]
        turns.append({"turn": len(turns) + 1, **current, "criteria": criteria,
                      "alignment": raw["prompt_infos"][plain_history[-2]["content"]]["alignment"],
                      "judge_prompt": source.get_eval_pattern(messages=plain_history, criteria=criteria),
                      "judge_status": "pending_separate_API_phase"})
    if len(turns) != 5:
        raise ValueError("SysBench source session is not five turns")
    return {"turns": turns, "judge_status": "pending_separate_API_phase"}


def evaluate_mtrag(model, tokenizer, case, config):
    task = case["task"]
    history = [message("system", segment(POLICY, "system", framing=True)), *task["history"],
               rag_user(task["question"], task["contexts"], question_ref=task["question_ref"])]
    result = generate_transcript(model, tokenizer, history, config)
    result["utility_status"] = "pending_RB_alg_phase"
    return result


def scenario_specs(case, payloads, history_modes):
    eligible = [t["turn"] for t in case["turns"] if t["turn"] >= 2 and t["contexts"]]
    if not eligible:
        return
    for pid in case["payload_ids"]:
        for mode in history_modes:
            if mode == "own" and not case["rollout_audit"]["eligible"]:
                continue
            yield {"payload_id": pid, "history_mode": mode, "channel": "clean", "schedule": "none", "attack_turns": []}
            unique = set()
            for schedule, targets in [("early_once", [eligible[0]]), ("late_once", [eligible[-1]]), ("repeated", eligible)]:
                if tuple(targets) in unique:
                    continue
                unique.add(tuple(targets))
                for channel in ["direct", "indirect"]:
                    yield {"payload_id": pid, "history_mode": mode, "channel": channel,
                           "schedule": schedule, "attack_turns": targets}


def evaluate_scenario(model, tokenizer, case, scenario, payload, config):
    system = [segment(POLICY, "system", framing=True)]
    if payload["goal"] == "secret_extraction":
        system += [segment("\n", "system", framing=True),
                   segment(payload["protected_instruction"], "system", source=payload["protected_source"])]
    history, turns = [message("system", *system)], []
    for task in case["turns"]:
        active = task["turn"] in scenario["attack_turns"]
        current_user = rag_user(task["question"], task["contexts"], payload=payload if active else None,
                                channel=scenario["channel"], question_ref=task["question_ref"])
        current = generate_transcript(model, tokenizer, [*history, current_user], config)
        turns.append({"turn": task["turn"], "task_id": task["task_id"], "attack_present": active,
                      **current, **score_attack(payload, current["output"]), "utility_status": "pending_RB_alg_phase"})
        # mtRAG stores question/answer history and supplies passages separately
        # for the current turn. Preserve every Q/A; do not replay old retrieved
        # documents as though the dataset had made them persistent chat messages.
        stored_user = rag_user(task["question"], [], payload=payload if active and scenario["channel"] == "direct" else None,
                               channel=scenario["channel"], question_ref=task["question_ref"])
        history += [stored_user, assistant(current["output"] if scenario["history_mode"] == "own" else task["references"][0])]
    failed = next((t["turn"] for t in turns if t["attack_success"] is True), None)
    pending = any(t["attack_success"] is None for t in turns)
    return {"scenario": scenario, "goal": payload["goal"], "turns": turns,
            "session_asr": True if failed is not None else None if pending else False,
            "first_failure_turn": failed, "failure_censored": failed is None if not pending else None,
            "attack_review_pending": pending,
            "history_storage_policy": "source_mtrag_questions_and_answers; passages_supplied_per_current_turn",
            "observation_turns": len(turns), "utility_status": "pending_RB_alg_phase"}


def runtime_manifest(config, report):
    source = {str(p.relative_to(ROOT)): file_hash(p)
              for folder in [PROJECT / "experiments", PROJECT / "ha_aside", PROJECT / "analysis"]
              for p in sorted(folder.glob("*.py"))}
    versions = {}
    for name in ["torch", "transformers", "numpy", "scikit-learn", "rouge-score", "langdetect", "nltk"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if set(config["suites"]) & CONTROLS:
        from .run_benchmark import runtime_manifest as legacy_manifest
        source.update(legacy_manifest(config, report)["source_sha256"])
    return {"protocol": REVISION, "config": config, "preflight": report,
            "source_sha256": source, "environment": {"python": platform.python_version(), "packages": versions},
            "scorer_version": SCORER_VERSION, "attack_scope": "fixed_public_payloads; no adaptive attacker"}


def append_result(log, row):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def resume_rows(log):
    if not log.exists():
        return {}
    records = rows(log)
    indexed = {r["record_id"]: r for r in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate records in frozen generation log")
    return indexed


def run(config, report):
    if not report["ready"]:
        raise ValueError("Resolve all dataset/checkpoint preflight errors before inference")
    manifest = runtime_manifest(config, report)
    run_id = digest(manifest)
    output = resolve(config["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text(encoding="utf-8")) != manifest:
        raise ValueError("Resume manifest differs")
    write_json(output / "manifest.json", manifest)
    bundle = load_bundle(resolve(config["rq_prepared_dir"]))
    from .rq_selection import verify_selection
    selection_path = resolve(config["selection_manifest"])
    if file_hash(selection_path) != report["selection_sha256"]:
        raise ValueError("Selection changed after preflight")
    selection = verify_selection(json.loads(selection_path.read_text(encoding="utf-8")), bundle)
    settings = {**config, "_source_root": str(source_root(config, bundle)),
                "_control_fingerprints": {k: v for k, v in report.get("datasets", {}).items() if ":" in k}}
    payloads = {p["payload_id"]: p for p in rows(resolve(config["rq_prepared_dir"]) / "payloads.json")}
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["dtype"]]
    summary = {"run_id": run_id, "status": "in_progress", "models": {}}
    for variant in variants(config):
        model, tokenizer, metadata = load_checkpoint(variant["checkpoint"], variant["method"], "qwen25_7b", dtype, config["device"])
        summary["models"][variant["name"]] = {}
        try:
            for suite in config["suites"]:
                if variant["ablation"] and suite not in {"role_authority", "role_probe", "iheval", "four_role_multiturn"}:
                    continue
                if suite in CONTROLS:
                    from .rq_reporting import run_legacy_control
                    summary["models"][variant["name"]][suite] = run_legacy_control(model, tokenizer, suite, settings, output / variant["name"])
                    continue
                cases = load_cases(resolve(config["rq_prepared_dir"]), suite, None if suite == "role_probe" else config.get("sample_limit"))
                selection_suite = "role_authority" if suite == "role_probe" else suite
                allowed = set(selection["allowed"][selection_suite])
                cases = [c for c in cases if c["case_id"] in allowed]
                if suite == "role_probe":
                    from analysis.probe_four_roles import probe
                    probe_path = output / variant["name"] / "role_probe.json"
                    if not probe_path.exists():
                        write_json(probe_path, probe(model, tokenizer, cases, settings))
                    summary["models"][variant["name"]][suite] = json.loads(probe_path.read_text(encoding="utf-8"))
                    continue
                log = output / variant["name"] / f"{suite}.jsonl"
                cached = resume_rows(log)
                expected = set()
                for case in cases:
                    if suite == "four_role_multiturn" and case["split"] != config["split"]:
                        continue
                    specs = list(scenario_specs(case, payloads, config["history_modes"])) if suite == "four_role_multiturn" else [None]
                    for spec in specs:
                        rid = case["case_id"] + (":" + digest(spec)[:20] if spec else "")
                        expected.add(rid)
                        case_hash = digest(case)
                        if rid in cached:
                            if cached[rid]["case_sha256"] != case_hash or cached[rid].get("scenario") != spec:
                                raise ValueError("Resume case or scenario differs from the frozen source selection")
                            continue
                        try:
                            if suite in {"iheval", "role_authority", "task_controls"}:
                                result = evaluate_static(model, tokenizer, case, settings)
                            elif suite == "sysbench":
                                result = evaluate_sysbench(model, tokenizer, case, settings)
                            elif suite == "mtrag_reference":
                                result = evaluate_mtrag(model, tokenizer, case, settings)
                            else:
                                result = evaluate_scenario(model, tokenizer, case, spec, payloads[spec["payload_id"]], settings)
                            row = {**result, "record_id": rid, "case_id": case["case_id"], "case_sha256": case_hash,
                                   "group_id": case["group_id"], "suite": suite, "scenario": spec,
                                   "metadata": case.get("metadata", {}), "benchmark_kind": case["benchmark_kind"]}
                            append_result(log, row)
                            cached[rid] = row
                        except Exception as error:
                            write_json(output / "failure.json", {"variant": variant["name"], "suite": suite,
                                       "record_id": rid, "error": repr(error), "status": "failed_not_scored_as_defense"})
                            raise
                if set(cached) != expected:
                    raise ValueError("Generation log has extra or missing frozen cases")
                summary["models"][variant["name"]][suite] = {"records": len(cached), "status": "complete_generation"}
                write_json(output / "summary.json", summary)
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary["status"] = "complete_generation; SysBench_judge_and_RB_alg_separate"
    write_json(output / "summary.json", summary)
    from .rq_reporting import export_reports
    export_reports(output, config)
    return output

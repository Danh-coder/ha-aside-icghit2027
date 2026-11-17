"""Separate, resumable SysBench API judging and pinned mtRAG utility scoring."""

import json
import importlib.metadata
import os
from pathlib import Path
import urllib.request

from .common import digest, file_hash, resolve, write_json
from .rq_data import load_bundle, load_cases
from .rq_sources import rows
from .rq_scoring import MTRAGScorer, SCORER_VERSION, parse_sysbench_verdict, sysbench_module
from .rq_runner import append_result, resume_rows, source_root
from .rq_reporting import export_reports


def call_judge(prompt, model, temperature):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is required only for the explicit --judge phase")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    request = urllib.request.Request(base + "/chat/completions", method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        data=json.dumps({"model": model, "temperature": temperature,
                         "messages": [{"role": "user", "content": prompt}]}).encode("utf-8"))
    with urllib.request.urlopen(request, timeout=180) as response:
        value = json.load(response)
    choice = value["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("Verifier did not finish normally")
    return {"response": choice["message"]["content"], "actual_model": value.get("model"),
            "system_fingerprint": value.get("system_fingerprint"), "usage": value.get("usage")}


def postprocess(config, run_dir, *, judge=False, utility=False, attack_review=None, scoring_device="cpu", judge_fn=None, utility_fn=None):
    if not judge and not utility and not attack_review:
        raise ValueError("Select an explicit judging or utility phase")
    run_dir = Path(run_dir)
    parent = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    from .rq_integrity import verify_generation
    verify_generation(run_dir, parent["config"])
    immutable_fields = ["models", "methods", "suites", "preset", "split", "history_modes", "enable_rgag",
                        "sample_limit", "interventions", "seed", "bootstrap_samples", "probe_feature", "probe_group_limit",
                        "checkpoint_policy", "allow_published_checkpoints"]
    if any(config.get(k) != parent["config"].get(k) for k in immutable_fields):
        raise ValueError("Scoring configuration changes the frozen generation scope")
    status = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["status"]
    if not status.startswith("complete_generation"):
        raise ValueError("Finish generation before scoring; partial sessions cannot be scored")
    bundle = load_bundle(resolve(config["rq_prepared_dir"]))
    from .rq_sources import verify_sources
    if digest(verify_sources(source_root(config, bundle))) != digest(bundle["source_lock"]):
        raise ValueError("Original scoring source snapshot changed")
    if bundle["fingerprint"] != parent["preflight"]["dataset_fingerprint"]:
        raise ValueError("Scoring dataset differs from the generation dataset")
    generation_logs = {p.relative_to(run_dir).as_posix(): file_hash(p) for name in config["methods"] + [v["name"] for v in config.get("ablations", [])]
                       for p in sorted((run_dir / name).glob("*.jsonl"))}
    package_versions = {}
    for name in ["torch", "transformers", "bert-score", "rouge-score", "numpy", "langdetect", "nltk"]:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    identity = {"parent_manifest_sha256": file_hash(run_dir / "manifest.json"), "generation_logs": generation_logs,
                "dataset_fingerprint": bundle["fingerprint"], "judge": judge, "utility": utility,
                "judge_model": config.get("judge_model", "gpt-4o-2024-05-13") if judge else None,
                "utility_model_revision": config.get("utility_model_revision") if utility else None,
                "attack_review_sha256": file_hash(attack_review) if attack_review else None,
                "scoring_device": scoring_device, "scorer_version": SCORER_VERSION,
                "package_versions": package_versions,
                "source_sha256": {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).with_name("rq_scoring.py"), Path(__file__).with_name("rq_reporting.py")]}}
    output = run_dir / "scores" / digest(identity)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", identity)
    verdicts, utilities, attack_overrides = {}, {}, {}
    (output / "scores.jsonl").touch(exist_ok=True)
    cache = resume_rows(output / "scores.jsonl")
    if (output / "summary.json").exists():
        completed = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if completed.get("scores_sha256") != file_hash(output / "scores.jsonl"):
            raise ValueError("Completed scoring outputs changed")
    if utility:
        utility_fn = utility_fn or MTRAGScorer(config.get("utility_model_revision"), device=scoring_device)
    judge_fn = judge_fn or call_judge
    scorer_source = sysbench_module(str(source_root(config, bundle))) if judge else None
    expected = set()
    if attack_review:
        review = json.loads(Path(attack_review).read_text(encoding="utf-8"))
        if review.get("parent_manifest_sha256") != identity["parent_manifest_sha256"]:
            raise ValueError("Attack adjudications belong to a different generation run")
        for decision in review["records"]:
            if type(decision.get("attack_success")) is not bool or not decision.get("reviewer") or not decision.get("reason"):
                raise ValueError("Every attack adjudication requires a boolean label, reviewer and reason")
            if decision["model"] not in config["methods"] + [v["name"] for v in config.get("ablations", [])]:
                raise ValueError("Attack review references an unknown model")
            generated = {r["record_id"]: r for r in rows(run_dir / decision["model"] / "four_role_multiturn.jsonl")}
            parent_row = generated[decision["record_id"]]
            turn = next(t for t in parent_row["turns"] if t["turn"] == decision["turn"])
            if turn["attack_success"] is not None or digest(turn["output"]) != decision["output_sha256"]:
                raise ValueError("Attack review is not a flagged immutable generation output")
            rid = digest(["attack_review", decision["model"], decision["record_id"], decision["turn"]])
            if rid in expected:
                raise ValueError("Duplicate attack adjudication")
            expected.add(rid)
            if rid not in cache:
                value = {"record_id": rid, "kind": "attack_review", "model": decision["model"],
                         "parent_record_id": decision["record_id"], "turn": decision["turn"], "decision": decision}
                append_result(output / "scores.jsonl", value)
                cache[rid] = value
            attack_overrides[(decision["model"], decision["record_id"], decision["turn"])] = decision["attack_success"]
    for model in config["methods"] + [v["name"] for v in config.get("ablations", [])]:
        folder = run_dir / model
        if judge and (folder / "sysbench.jsonl").exists():
            for row in rows(folder / "sysbench.jsonl"):
                for turn in row["turns"]:
                    rid = digest(["judge", model, row["record_id"], turn["turn"]])
                    expected.add(rid)
                    if rid not in cache:
                        failures = []
                        for retry in range(10):
                            temperature = 0 if retry < 5 else 0.5
                            try:
                                raw = judge_fn(turn["judge_prompt"], identity["judge_model"], temperature)
                                verdict = parse_sysbench_verdict(raw["response"], turn["criteria"])
                                for key, criterion in turn["criteria"].items():
                                    count = scorer_source.character_count(criterion["criteria_content"], turn["output"])
                                    if count != -1:
                                        verdict["constraints"][str(key)] = bool(count)
                                value = {"record_id": rid, "kind": "judge", "model": model,
                                         "parent_record_id": row["record_id"], "turn": turn["turn"],
                                         "verdict": verdict, "raw": raw, "retry": retry,
                                         "temperature": temperature, "failed_attempts": failures}
                                append_result(output / "scores.jsonl", value)
                                cache[rid] = value
                                break
                            except Exception as error:
                                failures.append({"retry": retry, "temperature": temperature, "error": repr(error)})
                        else:
                            write_json(output / "failure.json", {"record_id": rid, "attempts": failures,
                                                                "status": "pending_judgment_not_zero"})
                            raise RuntimeError("SysBench verifier failed; partial judgments remain resumable")
                    verdicts[(model, row["record_id"], turn["turn"])] = cache[rid]["verdict"]
        if utility:
            for suite in ["mtrag_reference", "four_role_multiturn"]:
                path = folder / f"{suite}.jsonl"
                if not path.exists():
                    continue
                cases = {c["case_id"]: c for c in load_cases(resolve(config["rq_prepared_dir"]), suite)}
                for row in rows(path):
                    case = cases[row["case_id"]]
                    work = [(None, row, case["task"])] if suite == "mtrag_reference" else [
                        (turn["turn"], turn, next(t for t in case["turns"] if t["turn"] == turn["turn"])) for turn in row["turns"]]
                    for turn_number, generated, task in work:
                        rid = digest(["utility", model, row["record_id"], turn_number])
                        expected.add(rid)
                        if rid not in cache:
                            score = utility_fn(generated["output"], task["references"], task["contexts"])
                            value = {"record_id": rid, "kind": "utility", "model": model, "parent_record_id": row["record_id"],
                                     "turn": turn_number, "score": score}
                            append_result(output / "scores.jsonl", value)
                            cache[rid] = value
                        utilities[(model, row["record_id"], turn_number)] = cache[rid]["score"]
    if set(cache) != expected:
        raise ValueError("Scoring cache has extra or missing generation records")
    write_json(output / "summary.json", {"status": "complete", "n_scores": len(cache), "identity": identity,
                                         "scores_sha256": file_hash(output / "scores.jsonl")})
    index_path = run_dir / "scoring_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    for key, enabled in [("judge", judge), ("utility", utility), ("attack_review", bool(attack_review))]:
        if enabled:
            index[key] = str(output.relative_to(run_dir))
    write_json(index_path, index)
    # Merge completed phases only; frozen generation logs are never rewritten.
    for kind in ["judge", "utility", "attack_review"]:
        if kind in index:
            phase = run_dir / index[kind]
            phase_manifest = json.loads((phase / "manifest.json").read_text(encoding="utf-8"))
            if phase_manifest["generation_logs"] != generation_logs:
                raise ValueError("A prior scoring phase belongs to different generation outputs")
            for score in rows(phase / "scores.jsonl"):
                key = (score["model"], score["parent_record_id"], score["turn"])
                if score["kind"] == "judge":
                    verdicts[key] = score["verdict"]
                elif score["kind"] == "utility":
                    utilities[key] = score["score"]
                elif score["kind"] == "attack_review":
                    attack_overrides[key] = score["decision"]["attack_success"]
    export_reports(run_dir, config, judge_scores=verdicts, utility_scores=utilities,
                   attack_reviews=attack_overrides, destination=output / "reports")
    return output

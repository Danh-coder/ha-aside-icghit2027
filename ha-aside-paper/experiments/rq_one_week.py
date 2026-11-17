"""Deterministic source-only subset for the one-week multi-role evaluation.

This module does not change the parent budget audit, transcripts, gold answers,
decoding or scorers. Selection never reads model outputs.
"""

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

from .common import PROJECT, digest, file_hash, resolve, write_json
from .rq_data import load_bundle, load_cases
from .rq_selection import verify_selection

PROFILE = "rq_multirole_one_week_v2"
SUITES = ["role_authority", "role_probe", "task_controls", "iheval"]
PREPARED = PROJECT / "data/prepared/rq_public_v1"
PARENT = PREPARED / "selection.v2.json"
SELECTION = PREPARED / "selection.one-week.v2.json"
CORE_METHODS = ["ha_aside", "aside_orig", "vanilla", "ise"]
TASKS = {"get-webpage", "lang-detect", "multi-turn", "single-turn", "slack-user",
         "system-prompt-extract", "translation", "user-prompt-hijack", "verb-extract"}
ROLES = {"system", "user", "data", "assistant"}
RECIPE = {"seed": 42, "authority_contexts_per_clause": 2,
          "iheval_source_ids_per_task": 50, "keep_all_task_controls": True,
          "keep_all_IHEval_reference_calibration_helpers": True,
          "keep_all_complete_probe_clauses": True,
          "authority_conflict_policy": "evaluate_priority_in_IHEval; omit_duplicate_authority_extensions",
          "rank": "sha256(compact_JSON([42, ...source_key])); no_model_output_access"}


def rank(*key):
    return hashlib.sha256(json.dumps([42, *key], separators=(",", ":")).encode()).hexdigest()


def select_cases(cases, parent):
    """Pure source-ID selection, shared by the builder and independent audit."""
    filtered = {}
    for suite in ["role_authority", "iheval", "task_controls"]:
        allowed = set(parent["allowed"][suite])
        filtered[suite] = [c for c in cases[suite] if c["case_id"] in allowed]
        ids = [c["case_id"] for c in filtered[suite]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate public case ID")
    clauses = defaultdict(lambda: defaultdict(dict))
    for c in filtered["role_authority"]:
        if c["scorer"]["kind"] == "witness_authority":
            role = c["metadata"]["condition"]
            pair = c["case_id"].rsplit(":", 1)[0]
            if role not in ROLES or role in clauses[c["group_id"]][pair]:
                raise ValueError("Invalid or duplicate role in a source context")
            clauses[c["group_id"]][pair][role] = c
    authority, context_ids = [], {}
    for group, contexts in sorted(clauses.items()):
        complete = [key for key, roles in contexts.items() if set(roles) == ROLES]
        if len(complete) < RECIPE["authority_contexts_per_clause"]:
            raise ValueError("A source clause lacks two complete four-role contexts")
        keep = sorted(complete, key=lambda key: rank(group, key))[:2]
        context_ids[group] = sorted(keep)
        for key in keep:
            authority.extend(contexts[key][role] for role in sorted(ROLES))
    grouped = defaultdict(list)
    for c in filtered["iheval"]:
        grouped[(c["metadata"]["task"], c["group_id"])].append(c)
    if {task for task, _ in grouped} != TASKS:
        raise ValueError("IHEval task inventory differs from the audited nine tasks")
    iheval, chosen_ids = [], {}
    for task in sorted(TASKS):
        eligible = [key for key, records in grouped.items() if key[0] == task and
                    {c["metadata"]["setting"] for c in records} == {"reference", "aligned", "conflict"}]
        if len(eligible) < RECIPE["iheval_source_ids_per_task"]:
            raise ValueError("IHEval task lacks 50 complete aligned/conflict/reference source groups")
        keep = sorted(eligible, key=lambda key: rank(*key))[:50]
        chosen_ids[task] = sorted(key[1] for key in keep)
        for key in keep:
            iheval.extend(grouped[key])  # preserve EVERY source variant
    # Upstream reference scores combine each data answer with strong/weak
    # instruction-only responses. These reference-only helper IDs are not
    # eligible for three-setting sampling, but MUST still be generated.
    helpers = [c for c in filtered["iheval"] if c["metadata"]["setting"] == "reference" and
               c["metadata"]["task"] in {"verb-extract", "translation", "get-webpage"} and
               any(prefix in str(c["metadata"]["source_id"]) for prefix in ["strong_", "weak_"])]
    if len(helpers) != 8:
        raise ValueError("IHEval reference calibration helper inventory differs from the audited eight requests")
    iheval.extend(helpers)
    selected = {"role_authority": authority, "iheval": iheval,
                "task_controls": filtered["task_controls"]}
    allowed = {suite: sorted(c["case_id"] for c in records) for suite, records in selected.items()}
    # Unsupported suites have empty cohorts rather than silently inheriting full runs.
    for suite in parent["allowed"]:
        allowed.setdefault(suite, [])
    counts = {"role_authority_cases": len(authority), "authority_clause_groups": len(clauses),
              "authority_context_pairs": sum(map(len, context_ids.values())),
              "authority_generation_calls": len(authority) +
                  2 * sum(c["metadata"]["condition"] == "data" for c in authority),
              "probe_examples": 4 * len(clauses), "probe_train_examples": int(len(clauses) * .7) * 4,
              "probe_test_examples": (len(clauses) - int(len(clauses) * .7)) * 4,
              "iheval_source_groups": sum(map(len, chosen_ids.values())),
              "iheval_reference_calibration_helpers": len(helpers),
              "iheval_requests": len(iheval), "task_controls": len(selected["task_controls"])}
    counts["generation_calls_per_method"] = (counts["authority_generation_calls"] +
                                               counts["iheval_requests"] + counts["task_controls"])
    counts["iheval_requests_by_task"] = dict(sorted(Counter(c["metadata"]["task"] for c in iheval).items()))
    counts["iheval_requests_by_setting"] = dict(sorted(Counter(c["metadata"]["setting"] for c in iheval).items()))
    return allowed, {"counts": counts, "authority_context_ids": context_ids, "iheval_group_ids": chosen_ids,
                     "iheval_reference_helper_ids": sorted(c["case_id"] for c in helpers)}


def build_selection(prepared=PREPARED, parent_path=PARENT):
    prepared, parent_path = Path(prepared), Path(parent_path)
    bundle = load_bundle(prepared)
    parent = verify_selection(json.loads(parent_path.read_text(encoding="utf-8")), bundle)
    if parent.get("profile"):
        raise ValueError("A weekly subset must derive from the full budget audit")
    cases = {suite: load_cases(prepared, suite) for suite in ["role_authority", "iheval", "task_controls"]}
    allowed, audit = select_cases(cases, parent)
    selected_ids = {suite: set(ids) for suite, ids in allowed.items()}
    coverage = {suite: {"source_cases": bundle["suites"][suite]["n"],
                        "parent_budget_retained_cases": parent["coverage"][suite]["retained_cases"],
                        "retained_cases": len(ids),
                        "excluded_cases": bundle["suites"][suite]["n"] - len(ids)}
                for suite, ids in allowed.items()}
    value = {"identity": copy.deepcopy(parent["identity"]), "profile": PROFILE, "recipe": RECIPE,
             "parent_selection_fingerprint": parent["fingerprint"],
             "parent_selection_sha256": file_hash(parent_path), "allowed": allowed,
             "coverage": coverage, "audit": audit,
             "selected_case_sha256": {suite: {c["case_id"]: digest(c) for c in records
                                        if c["case_id"] in selected_ids[suite]}
                                      for suite, records in cases.items()},
             "interpretation": "Stratified source benchmark subset, NOT full IHEval or full RQ cohort; "
                               "no generated research content; unchanged source scorers and decoding"}
    value["fingerprint"] = digest(value)
    return value


def freeze_selection(output=SELECTION, prepared=PREPARED, parent_path=PARENT):
    expected = build_selection(prepared, parent_path)
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != expected:
            raise ValueError("One-week selection is immutable; existing artifact differs from the source recipe")
    else:
        write_json(output, expected)
    return expected


def verify_subset(path=SELECTION, prepared=PREPARED, parent_path=PARENT):
    actual = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = build_selection(prepared, parent_path)
    if actual != expected:
        raise ValueError("One-week subset differs from its frozen source-only selection recipe")
    return actual


def apply_profile(config):
    selection = verify_subset()
    config.update(preset="staged", suites=SUITES.copy(), sample_limit=None, probe_group_limit=None,
                  selection_manifest=SELECTION.relative_to(PROJECT.parent).as_posix(),
                  parent_selection_manifest=PARENT.relative_to(PROJECT.parent).as_posix(),
                  benchmark_profile=PROFILE, subset_fingerprint=selection["fingerprint"])
    config["execution_order"] = "benchmark_blocks"
    return config


def audit_config(config):
    selection_path = resolve(config["selection_manifest"])
    header = json.loads(selection_path.read_text(encoding="utf-8"))
    profile = config.get("benchmark_profile")
    if not profile and not header.get("profile"):
        return None
    if (profile != PROFILE or header.get("profile") != PROFILE or config.get("execution_order", "benchmark_blocks") != "benchmark_blocks" or config["preset"] != "staged" or config["suites"] != SUITES or
            config.get("sample_limit") is not None or config.get("probe_group_limit") is not None or
            config.get("interventions") != ["user", "system"]):
        raise ValueError("One-week profile requires its exact staged suites, paired interventions and frozen selection")
    value = verify_subset(selection_path, resolve(config["rq_prepared_dir"]),
                          resolve(config["parent_selection_manifest"]))
    if config.get("subset_fingerprint") != value["fingerprint"]:
        raise ValueError("Config does not pin the one-week subset fingerprint")
    return {"profile": PROFILE, "is_full_benchmark": False, "fingerprint": value["fingerprint"],
            "parent_fingerprint": value["parent_selection_fingerprint"],
            "counts_per_method": value["audit"]["counts"],
            "planned_generation_calls": value["audit"]["counts"]["generation_calls_per_method"] * len(config["methods"]) +
                (value["audit"]["counts"]["generation_calls_per_method"] - value["audit"]["counts"]["task_controls"]) * len(config.get("ablations", []))}

"""Training evidence and explicit completeness for incremental RQ evaluation."""

import json
import math
from pathlib import Path

from .common import file_hash

METHODS = ["vanilla", "aside_orig", "ise", "ise_4role", "ha_aside"]
ABLATION = "ha_no_user_rotation"


def checkpoint_status(path, metadata, policy="validation_selected"):
    path = Path(path)
    p = metadata["provenance"]
    if p.get("pilot_max_steps") is not None:
        raise ValueError("A bounded pilot cannot be evaluated as completed training")
    if p.get("hyperparameter_search_complete") or p.get("fixed_recipe_complete"):
        return {"status": "validation_selected", "selection_scope": "completed_search_or_fixed_recipe"}
    if policy != "completed_trial":
        raise ValueError("Search is incomplete; use --staged --allow-completed-trial with training evidence")
    expected_epochs = float(p.get("epochs", 0))
    if not math.isfinite(expected_epochs) or expected_epochs <= 0:
        raise ValueError("Completed-trial evaluation requires the planned epoch count")
    target_state = path / "trainer_state.json"
    if not target_state.is_file():
        raise ValueError("Completed-trial evaluation requires trainer_state.json")
    state = json.loads(target_state.read_text(encoding="utf-8"))
    loss = float(p["selected_validation_loss"])
    if not any(r.get("step") == state.get("global_step")
               and math.isclose(float(r.get("eval_loss", math.inf)), loss, rel_tol=1e-7, abs_tol=1e-9)
               for r in state.get("log_history", [])):
        raise ValueError("Checkpoint validation loss does not match its saved Trainer state")

    def completed(value):
        return (float(value.get("epoch") or 0) + 1e-6 >= expected_epochs
                and type(value.get("max_steps")) is int and value["max_steps"] > 0
                and value.get("global_step", 0) >= value["max_steps"])

    evidence = {target_state.name: file_hash(target_state)}
    scope = "final_epoch_checkpoint"
    if not completed(state):
        # A best-validation epoch may precede the final epoch. It is usable only
        # when this same trial subsequently finished and selected it by validation.
        results_path = path.parent.parent / "trial_results.json"
        if not results_path.is_file():
            raise ValueError("Trial has not completed the planned epochs")
        records = json.loads(results_path.read_text(encoding="utf-8"))
        matching = [r for r in records if r.get("trial") == p.get("trial")
                    and math.isclose(float(r["validation_loss"]), loss, rel_tol=1e-7, abs_tol=1e-9)
                    and Path(r["checkpoint"]).name == path.name
                    and Path(r["checkpoint"]).parent.name == path.parent.name]
        finals = [s for s in path.parent.glob("checkpoint-*/trainer_state.json")
                  if completed(json.loads(s.read_text(encoding="utf-8")))]
        if not matching or not finals:
            raise ValueError("No completed same-trial validation selection evidence")
        evidence["trial_results.json"] = file_hash(results_path)
        evidence[finals[0].parent.name + "/trainer_state.json"] = file_hash(finals[0])
        scope = "best_validation_epoch_of_completed_trial"
    return {"status": "completed_trial_search_incomplete", "selection_scope": scope,
            "planned_epochs": expected_epochs, "checkpoint_epoch": state.get("epoch"),
            "evidence_sha256": evidence,
            "interpretation": "Single completed configuration; no claim of completed hyperparameter search"}


def pairing_errors(checkpoints):
    main = checkpoints.get("ha_aside")
    if not main:
        return []
    errors = []
    for name in ["ise_4role", ABLATION]:
        if name not in checkpoints:
            continue
        origin = main["metadata"]["provenance"]
        paired = checkpoints[name]["metadata"]["provenance"]
        fields = ["training_data_sha256", "validation_data_sha256", "training_seed", "loss_scope", "trial", "epochs"]
        if any(origin.get(k) != paired.get(k) for k in fields):
            errors.append(f"{name}: paired recipe/data/seed differs from selected HA")
        if paired.get("adaptation_reference", {}).get("architecture_sha256") != main["artifact_sha256"]["architecture.json"]:
            errors.append(f"{name}: fixed-recipe origin does not reference this HA checkpoint")
    return errors


def comparison_status(config, checkpoints):
    present = list(config["methods"]) + [v["name"] for v in config.get("ablations", [])]
    missing = [m for m in [*METHODS, ABLATION] if m not in present]
    trials = [m for m, c in checkpoints.items()
              if c.get("training_status", {}).get("status") == "completed_trial_search_incomplete"]
    pending_pairs = [m for m in ["ise_4role", ABLATION] if m in present and "ha_aside" not in checkpoints]
    required_suites = ["role_authority", "role_probe", "iheval", "task_controls", "sysbench", "four_role_multiturn",
                       "mtrag_reference", "sep", "struq_id", "struq_ood", "bipia_text"]
    missing_suites = [s for s in required_suites if s not in config["suites"]]
    published = [name for name, c in checkpoints.items() if c["metadata"].get("source_type") == "author_published"]
    subset = config.get("benchmark_profile")
    return {"evaluated_methods": present, "missing_methods_or_ablations": missing,
            "benchmark_profile": subset, "is_full_benchmark": not bool(subset),
            "subset_fingerprint": config.get("subset_fingerprint"),
            "published_checkpoints": published,
            "origin_revision_unreported_for": published,
            "controlled_pretrained_revision_verified": not published,
            "comparison_scope": "end_to_end_published_baselines; no claim of identical pretrained revision" if published else "native_checkpoint_protocol",
            "missing_planned_suites": missing_suites,
            "all_five_methods_present": all(m in present for m in METHODS),
            "planned_comparison_complete": not missing and not missing_suites and not pending_pairs and not trials and config["preset"] != "smoke" and not subset,
            "completed_trial_checkpoints": trials, "paired_origin_pending_HA": pending_pairs,
            "evaluation_scope": "smoke" if config["preset"] == "smoke" else "one_week_stratified_subset" if subset else "full_frozen_cohort"}

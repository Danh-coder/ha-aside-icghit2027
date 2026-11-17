"""Create a server RQ config from actual checkpoint paths; never load weights."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import PROJECT, ROOT, resolve, write_json
from experiments.rq_runner import METHODS, CONTROLS, NEW_SUITES, validate_config


def checkpoint_path(value):
    path = resolve(value)
    if path.is_file() and path.name == "selected_checkpoint.json":
        path = Path(json.loads(path.read_text(encoding="utf-8"))["checkpoint"])
    elif (path / "selected_checkpoint.json").is_file():
        path = Path(json.loads((path / "selected_checkpoint.json").read_text(encoding="utf-8"))["checkpoint"])
    if not (path / "architecture.json").is_file() and not (path / "published_checkpoint.json").is_file():
        raise FileNotFoundError(f"No saved method-specific checkpoint at {path}")
    return path.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-config", type=Path, help="Reuse actual Base/tokenizer and existing checkpoint paths")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--import-run", action="append", help="Stopped compatible one-week generation run to reuse with strict provenance checks")
    parser.add_argument("--ablation", type=str, help="Completed --user-angle 0 HA checkpoint or selected run directory")
    parser.add_argument("--method", action="append", choices=METHODS, help="Method subset for staged or smoke evaluation")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--staged", action="store_true", help="Full frozen cohort for available methods; merge later")
    modes.add_argument("--one-week", action="store_true", help="Frozen stratified multi-role subset; staged methods, unchanged decoding")
    parser.add_argument("--allow-completed-trial", action="store_true", help="Evaluate a proven completed trial without claiming search completion")
    parser.add_argument("--published-baselines", action="store_true", help="Explicitly allow the pinned author-published checkpoints with unreported origin fields")
    parser.add_argument("--smoke-suite", action="append", choices=sorted(NEW_SUITES | CONTROLS), help="Optional small smoke-only suite subset")
    parser.add_argument("--smoke-samples", type=int, help="Smoke-only case count per suite")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"])
    args = parser.parse_args()
    if args.one_week:
        args.staged = True
    if (args.smoke_suite or args.smoke_samples is not None) and not args.smoke:
        parser.error("smoke-suite and smoke-samples require --smoke")
    if args.smoke_samples is not None and args.smoke_samples < 1:
        parser.error("smoke-samples must be positive")
    if args.output.exists():
        raise FileExistsError("Select a new config filename; existing run setup is not overwritten")
    config = json.loads((PROJECT / "experiments/config_rq_benchmark.json").read_text(encoding="utf-8"))
    if args.from_config:
        old = json.loads(args.from_config.read_text(encoding="utf-8"))
        # Preserve the exact benchmark protocol when adding another method.
        config.update(old)
        if not args.smoke and old.get("preset") == "smoke":
            parser.error("A smoke config cannot seed a full run; use the full template")
    for argument in args.checkpoint:
        method, value = argument.split("=", 1)
        if method not in METHODS:
            parser.error(f"Unknown method: {method}")
        config["checkpoints"]["qwen25_7b"][method] = value
    config.update(device=args.device or config["device"], dtype=args.dtype or config["dtype"])
    config["preset"] = "smoke" if args.smoke else "staged" if args.staged else "reference"
    config["checkpoint_policy"] = "completed_trial" if args.allow_completed_trial else "validation_selected"
    if args.published_baselines:
        config["allow_published_checkpoints"] = True
    if config.get("allow_published_checkpoints") and not (args.staged or args.smoke):
        parser.error("Published checkpoints require --staged or --smoke")
    config["ablations"] = []
    if args.smoke:
        config.update(preset="smoke", sample_limit=4, probe_group_limit=8,
                      suites=["role_authority", "role_probe", "iheval", "task_controls", "sysbench", "four_role_multiturn", "mtrag_reference"])
        config["ablations"] = []
        if args.smoke_suite:
            config["suites"] = args.smoke_suite
        if args.smoke_samples is not None:
            config["sample_limit"] = args.smoke_samples
    if args.method:
        if not (args.smoke or args.staged):
            parser.error("method subsets require --staged or --smoke")
        config["methods"] = args.method
    elif args.staged and not args.from_config:
        config["methods"] = [m for m in METHODS if config["checkpoints"]["qwen25_7b"].get(m)]
        if not config["methods"]:
            parser.error("Provide checkpoint paths for this stage")
    if args.ablation:
        config["ablations"] = [{"name": "ha_no_user_rotation", "method": "ha_aside", "checkpoint": str(checkpoint_path(args.ablation))}]
    for method in config["methods"]:
        value = config["checkpoints"]["qwen25_7b"].get(method)
        if not value:
            parser.error(f"Provide a completed {method} checkpoint; Base-only substitution is disabled")
        path = checkpoint_path(value)
        if (path / "published_checkpoint.json").is_file():
            if not config.get("allow_published_checkpoints"):
                parser.error("Use --published-baselines for author checkpoints")
            from experiments.published_checkpoints import public_metadata
            metadata = public_metadata(path, method, "qwen25_7b")
        else:
            metadata = json.loads((path / "architecture.json").read_text(encoding="utf-8"))
        if metadata["method"] != method or (method == "ha_aside" and metadata["enable_rgag"]):
            parser.error(f"Checkpoint method/RGAG mismatch for {method}")
        config["checkpoints"]["qwen25_7b"][method] = str(path)
        config["checkpoint_training_recipes"]["qwen25_7b"][method] = metadata["provenance"]["training_recipe"]
    if not (args.smoke or args.staged) and not args.ablation:
        parser.error("The full RQ1 plan requires the completed HA USER-angle=0 ablation")
    if args.one_week or config.get("benchmark_profile") == "rq_multirole_one_week_v2":
        if args.smoke:
            parser.error("One-week paper subset cannot be converted to a smoke run")
        from experiments.rq_one_week import apply_profile
        apply_profile(config)
    if args.import_run:
        if not config.get("benchmark_profile"):
            parser.error("import-run requires the one-week profile")
        config["import_runs"] = [str(resolve(p).resolve()) for p in args.import_run]
    validate_config(config)
    write_json(args.output, config)
    print("Config:", args.output)
    print("Next: run_benchmark.py --config", args.output, "--report <preflight.json>")


if __name__ == "__main__":
    main()

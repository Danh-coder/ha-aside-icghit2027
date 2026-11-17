"""Full SFT with explicit dataset recipes and validation-selected checkpoints.

Default grid: 3 epochs, four learning rates, two warmup ratios and two effective
batch sizes. No model loads or training occur without --run.
"""

import argparse
import functools
import itertools
import json
import os
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (
    ORIGINAL,
    PROJECT,
    REGISTRY,
    file_hash,
    write_json,
    digest,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(REGISTRY), required=True)
    p.add_argument(
        "--method", choices=["vanilla", "ise", "ise_4role", "aside_orig", "ha_aside"], required=True
    )
    p.add_argument(
        "--rgag", action=argparse.BooleanOptionalAction, default=True,
        help="Enable RGAG for HA-ASIDE (default); --no-rgag trains hierarchical rotation only",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run", action="store_true")
    p.add_argument(
        "--single",
        action="store_true",
        help="Pilot only: first grid point; not a completed reference search",
    )
    p.add_argument(
        "--base-revision", required=True, help="Pinned pretrained commit revision"
    )
    p.add_argument(
        "--tokenizer-revision",
        required=True,
        help="Pinned instruction-tokenizer commit revision",
    )
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--user-angle", type=float, default=None,
                   help="HA USER rotation angle in radians; 0 is the trained RQ1 ablation")
    p.add_argument("--fixed-from", type=Path,
                   help="Reuse the selected HA checkpoint's actual recipe, seed and data for ISE-4/HA ablation")
    p.add_argument("--dataset", choices=["alpaca_reference", "chatqa_multirole"], default="alpaca_reference")
    p.add_argument("--data-dir", type=Path, default=PROJECT / "data/finetuning/chatqa_multirole_v1")
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--pilot-steps", type=int, default=None,
                   help="Bounded optimizer-step smoke run; requires --single; never a completed search")
    p.add_argument("--pilot-effective-batch-size", type=int, default=None,
                   help="Memory pilot batch override; requires --single and --pilot-steps")
    args = p.parse_args()
    if args.method == "ise_4role" and args.dataset != "chatqa_multirole":
        p.error("ISE-4 Qwen adaptation uses the audited ChatQA four-role recipe")
    fixed = None
    if args.fixed_from:
        if args.single or args.pilot_steps is not None or args.method not in {"ise_4role", "ha_aside"}:
            p.error("fixed-from is a complete fixed recipe for ISE-4 or HA, not a pilot")
        architecture_path = args.fixed_from / "architecture.json"
        fixed = json.loads(architecture_path.read_text(encoding="utf-8"))
        origin = fixed["provenance"]
        if fixed["method"] != "ha_aside" or fixed["enable_rgag"] or origin.get("pilot_max_steps") is not None:
            p.error("fixed-from requires a non-pilot RGAG-OFF HA checkpoint")
        if not (origin.get("hyperparameter_search_complete") or origin.get("fixed_recipe_complete")):
            p.error("fixed-from must be a completed validation-selected run")
        if (origin["base_model"] != REGISTRY[args.model][0] or origin["base_revision"] != args.base_revision
                or origin["tokenizer_revision"] != args.tokenizer_revision or origin["dataset"] != args.dataset):
            p.error("Fixed recipe Base/tokenizer/dataset mismatch")
        args.seed = origin["training_seed"]
        args.rgag = False
        args.gradient_checkpointing = origin.get("gradient_checkpointing", False)
        if args.micro_batch_size is None:
            args.micro_batch_size = origin["trial"]["per_device_train_batch_size"]
    if args.user_angle is not None and (args.method != "ha_aside" or not __import__("math").isfinite(args.user_angle)):
        p.error("user-angle requires HA-ASIDE and a finite angle")
    if args.micro_batch_size is not None and args.micro_batch_size < 1:
        p.error("micro-batch-size must be positive")
    if args.pilot_steps is not None and (not args.single or args.pilot_steps < 1):
        p.error("pilot-steps must be positive and requires --single")
    if args.pilot_effective_batch_size is not None and (args.pilot_steps is None or args.pilot_effective_batch_size < 1):
        p.error("pilot-effective-batch-size requires positive --pilot-steps and must be positive")
    if any(
        not re.fullmatch(r"[0-9a-f]{40}", v)
        for v in [args.base_revision, args.tokenizer_revision]
    ):
        p.error(
            "Base and tokenizer revisions must be full immutable 40-character commit hashes"
        )
    batches = [64, 128] if args.model == "llama2_13b" else [64, 256]
    grid = list(itertools.product([1e-6, 5e-6, 1e-5, 2e-5], [0.0, 0.1], batches))
    if args.single:
        grid = grid[:1]
    if fixed:
        chosen = fixed["provenance"]["trial"]
        grid = [(chosen["learning_rate"], chosen["warmup_ratio"], chosen["effective_batch_size"])]
    if args.pilot_effective_batch_size is not None:
        grid = [(lr, warmup, args.pilot_effective_batch_size) for lr, warmup, batch in grid]
    data_manifest = None
    if args.dataset == "chatqa_multirole":
        from tools.prepare_chatqa_dataset import load_prepared
        from training.chatqa import RECIPE, TOKENIZER_REVISION, ROLE_POLICY
        if args.model != "qwen25_7b" or args.tokenizer_revision != TOKENIZER_REVISION:
            p.error("ChatQA v1 is length-audited for Qwen2.5-7B only; use its pinned tokenizer revision")
        data_manifest, splits = load_prepared(args.data_dir)
        train, val = splits["train"], splits["validation"]
        train_path, val_path = args.data_dir / "train.jsonl", args.data_dir / "validation.jsonl"
        training_recipe, max_length, truncation = RECIPE, 2048, "disabled_whole_conversations_only"
        role_policy = ROLE_POLICY
    else:
        train_path = ORIGINAL / "data/train_data/alpaca_52k_shuffled_train.json"
        val_path = ORIGINAL / "data/train_data/alpaca_52k_shuffled_eval.json"
        train = json.loads(train_path.read_text(encoding="utf-8"))
        val = json.loads(val_path.read_text(encoding="utf-8"))
        if {r["orig_ix"] for r in train} & {r["orig_ix"] for r in val}:
            raise ValueError("Training/validation examples overlap")
        training_recipe, max_length, truncation = "matched_sft_reference_v3", 768, "aligned_right_truncation"
        role_policy = "operational task=user; external input=data; assistant=identity; special tokens=instruction"
    recipe = {
        "training_recipe": training_recipe,
        "enable_rgag": args.method == "ha_aside" and args.rgag,
        "dataset": args.dataset,
        "base_model": REGISTRY[args.model][0],
        "tokenizer_id": REGISTRY[args.model][1],
        "base_revision": args.base_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "training_seed": args.seed,
        "epochs": 3,
        "loss_scope": "full_sequence",
        "max_length": max_length,
        "training_truncation": truncation,
        "pilot_max_steps": args.pilot_steps,
        "training_data_sha256": file_hash(train_path),
        "validation_data_sha256": file_hash(val_path),
        "n_train": len(train),
        "n_validation": len(val),
        "hyperparameter_search_complete": False,
        "fixed_recipe_complete": False,
        "grid": [
            {"learning_rate": lr, "warmup_ratio": w, "effective_batch_size": b}
            for lr, w, b in grid
        ],
        "role_policy": role_policy,
        "source_sha256": {
            str(f.relative_to(PROJECT)): file_hash(f)
            for folder in ["ha_aside", "training", "experiments"]
            for f in (PROJECT / folder).glob("*.py")
        },
    }
    if data_manifest is not None:
        recipe["dataset_manifest_sha256"] = file_hash(args.data_dir / "manifest.json")
        recipe["dataset_manifest_canonical_sha256"] = digest(data_manifest)
        recipe["dataset_provenance"] = data_manifest
        recipe["checkpoint_policy"] = "all_epoch_model_weights_only; optimizer_resume_not_supported"
    if fixed:
        origin = fixed["provenance"]
        for key in ["training_data_sha256", "validation_data_sha256", "loss_scope", "max_length", "training_recipe"]:
            if recipe[key] != origin[key]:
                raise ValueError(f"Fixed recipe differs from selected HA provenance: {key}")
        recipe["adaptation_reference"] = {"architecture_sha256": file_hash(args.fixed_from / "architecture.json"),
                                           "trial": origin["trial"], "training_seed": origin["training_seed"]}
    if args.user_angle is not None:
        recipe["user_angle"] = args.user_angle
    # Retain actual legacy warmup behavior for a paired reproduction. New
    # training must not silently change the recipe of the ongoing server run.
    recipe["warmup_semantics"] = "legacy_grid_warmup_ratio_value_passed_as_warmup_steps"
    if args.method == "ise_4role":
        recipe["assistant_header_policy"] = "assistant_message_framing"
    # Resolve execution pairs before model loading, including CPU dry runs.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    for trial in recipe["grid"]:
        effective = trial["effective_batch_size"]
        micro = args.micro_batch_size or (1 if data_manifest else (4 if effective == 256 else 2))
        if effective % (micro * world):
            p.error("Effective batch must be divisible by micro-batch-size times world size")
        trial["per_device_train_batch_size"] = micro
        trial["gradient_accumulation_steps"] = effective // (micro * world)
    recipe["gradient_checkpointing"] = args.gradient_checkpointing
    print(json.dumps(recipe, indent=2))
    if not args.run:
        return
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from ha_aside.hf_adapters import HAASIDEModelWrapper
    from training.dataset import HierarchicalAlpacaDataset, HierarchicalConversationDataset, collate_hierarchical_batch

    world = int(os.environ.get("WORLD_SIZE", "1"))
    recipe["execution"] = {
        "world_size": world,
        "torch": torch.__version__,
        "precision": "bfloat16" if torch.cuda.is_available() else "float32",
        "optimizer": "adamw_torch",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "deepspeed": None if args.deepspeed is None else {
            "path": args.deepspeed, "sha256": file_hash(args.deepspeed),
        },
    }
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )
    occupied = bool(args.output.exists() and any(args.output.iterdir()))
    if world > 1:
        flag = torch.tensor(
            [occupied],
            device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
        )
        torch.distributed.broadcast(flag, src=0)
        occupied = bool(flag.item())
    if occupied:
        raise FileExistsError(
            "Use a new output directory; existing training runs are immutable"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        recipe["tokenizer_id"],
        revision=args.tokenizer_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if data_manifest is not None:
        train_ds = HierarchicalConversationDataset(train, tokenizer, assistant_header_roles=args.method == "ise_4role")
        val_ds = HierarchicalConversationDataset(val, tokenizer, assistant_header_roles=args.method == "ise_4role")
        # Fail on actual encoder drift before loading any base model weights.
        for dataset in [train_ds, val_ds]:
            for index in {0, len(dataset) - 1, max(range(len(dataset)), key=lambda i: dataset.data[i]["n_tokens"])}:
                dataset[index]
    else:
        train_ds = HierarchicalAlpacaDataset(train, tokenizer, merge_system=args.model == "mistral7b_v03")
        val_ds = HierarchicalAlpacaDataset(val, tokenizer, merge_system=args.model == "mistral7b_v03")
    recipe.update(
        truncated_train=train_ds.truncated_examples,
        truncated_validation=val_ds.truncated_examples,
    )
    trials = []

    class ExperimentTrainer(Trainer):
        def _save(self, output_dir=None, state_dict=None):
            model = self.accelerator.unwrap_model(self.model)
            state_dict = (
                self.accelerator.get_state_dict(self.model)
                if state_dict is None
                else state_dict
            )
            evaluations = [r for r in self.state.log_history if "eval_loss" in r]
            loss = evaluations[-1]["eval_loss"] if evaluations else None
            model.save_experiment(
                output_dir or self.args.output_dir,
                tokenizer,
                {**recipe, "selected_validation_loss": loss, "trial": trial},
                state_dict,
            )

    for parameters in recipe["grid"]:
        lr, warmup, effective = (parameters[key] for key in ["learning_rate", "warmup_ratio", "effective_batch_size"])
        set_seed(args.seed)
        # Match the paper's (microbatch, accumulation) pairs on eight GPUs:
        # 64 -> (2,4), 256 -> (4,8), and 13B's 128 -> (2,8).
        micro = parameters["per_device_train_batch_size"]
        if effective % (micro * world):
            raise ValueError("World size must divide the target effective batch size")
        trial = {
            "learning_rate": lr,
            "warmup_ratio": warmup,
            "effective_batch_size": effective,
            "per_device_train_batch_size": micro,
            "gradient_accumulation_steps": effective // (micro * world),
        }
        output = args.output / f"lr{lr:g}_warm{warmup:g}_batch{effective}"
        model = AutoModelForCausalLM.from_pretrained(
            recipe["base_model"],
            revision=args.base_revision,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="eager",
            trust_remote_code=False,
        )
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        wrapper = HAASIDEModelWrapper(
            model, mode=args.method, enable_rgag=recipe["enable_rgag"],
            angles=({0: 0.0, 1: args.user_angle, 2: __import__("math").pi / 2, 3: 0.0}
                    if args.user_angle is not None else None),
        )
        training_args = TrainingArguments(
            output_dir=str(output),
            num_train_epochs=3,
            max_steps=args.pilot_steps if args.pilot_steps is not None else -1,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup,
            optim="adamw_torch",
            per_device_train_batch_size=micro,
            per_device_eval_batch_size=micro,
            gradient_accumulation_steps=effective // (micro * world),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=None,
            save_only_model=data_manifest is not None,
            logging_steps=10,
            bf16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
            label_names=["labels"],
            seed=args.seed,
            data_seed=args.seed,
            deepspeed=args.deepspeed,
            prediction_loss_only=True,
            dataloader_num_workers=0,
        )
        trainer = ExperimentTrainer(
            model=wrapper,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=functools.partial(
                collate_hierarchical_batch, pad_token_id=tokenizer.pad_token_id
            ),
        )
        trainer.train()
        if trainer.is_world_process_zero():
            evaluations = [r for r in trainer.state.log_history if "eval_loss" in r]
            best = min(evaluations, key=lambda r: r["eval_loss"])
            selected = output / f"checkpoint-{best['step']}"
            if not (selected / "architecture.json").exists():
                raise RuntimeError("Selected epoch checkpoint was not saved")
            trials.append(
                {
                    "trial": trial,
                    "validation_loss": best["eval_loss"],
                    "checkpoint": str(selected.resolve()),
                }
            )
            write_json(args.output / "trial_results.json", trials)
        del trainer, wrapper, model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if int(os.environ.get("RANK", "0")) == 0:
        best = min(trials, key=lambda t: t["validation_loss"])
        metadata_path = Path(best["checkpoint"]) / "architecture.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["provenance"]["hyperparameter_search_complete"] = not args.single and fixed is None
        metadata["provenance"]["fixed_recipe_complete"] = fixed is not None
        metadata["provenance"]["validation_search_results"] = trials
        write_json(metadata_path, metadata)
        write_json(args.output / "selected_checkpoint.json", best)
        print("Selected checkpoint:", best["checkpoint"])


if __name__ == "__main__":
    main()

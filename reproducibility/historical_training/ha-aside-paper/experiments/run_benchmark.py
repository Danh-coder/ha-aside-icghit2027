"""Preflight and run the corrected matched-checkpoint benchmark."""

import argparse
import base64
import gc
import json
import importlib.metadata
import platform
import re
from pathlib import Path
import statistics
import sys
import time
import torch
import transformers

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (
    ROOT,
    PROJECT,
    ORIGINAL,
    REGISTRY,
    load_config,
    resolve,
    digest,
    file_hash,
    write_json,
)
from experiments.checkpoints import checkpoint_metadata, load_checkpoint
from experiments.benchmark_data import load_suite, score_attack
from experiments.prompts import (
    encode_task,
    encode_safety_task,
    encode_chunks,
    render_chunks,
    task_messages,
    Chunk,
)
from experiments.metrics import sep_metrics, proportion, qa_scores
from ha_aside.core import ROLE_DATA, ROLE_ASSISTANT, ROLE_USER
from experiments.safety_protocol import (
    SAFETY_REVISION, DIRECT_SUITES, SAFETY_SUITES, DIRECT_COUNTS, generation_settings,
)


def suite_seeds(config, suite):
    # Source paper: direct sampling across three seeds; indirect greedy. StruQ and
    # BIPIA still repeat seeds because attack construction/insertion is stochastic.
    return (
        config["seeds"]
        if suite
        in {
            "gandalf",
            "tensortrust",
            "purple",
            "rules",
            "struq_id",
            "struq_ood",
            "bipia_text",
            "bipia_code",
        }
        else [config["seeds"][0]]
    )


def preflight(config):
    report = {
        "protocol": config["protocol"],
        "preset": config["preset"],
        "safety_benchmark": SAFETY_REVISION,
        "bipia_attack_filter": config.get("bipia_attack_filter", "aside_source_regex"),
        "datasets": {},
        "checkpoints": {},
        "errors": [],
    }
    for suite in config["suites"]:
        for seed in suite_seeds(config, suite):
            name = f"{suite}:{seed}"
            try:
                examples = load_suite(
                    suite,
                    seed,
                    resolve(config["prepared_dir"]),
                    config["sep_split"],
                    config.get("sample_limit"),
                    bipia_filter=config.get("bipia_attack_filter", "aside_source_regex"),
                )
                if config["preset"] == "reference" and suite in DIRECT_COUNTS:
                    if len(examples) != DIRECT_COUNTS[suite]:
                        raise ValueError(f"{suite} requires {DIRECT_COUNTS[suite]} original-paper examples")
                report["datasets"][name] = {
                    "n": len(examples),
                    "sha256": digest([e.serializable() for e in examples]),
                }
            except Exception as e:
                report["errors"].append(f"{name}: {type(e).__name__}: {e}")
    for model in config["models"]:
        matched = []
        base_identities = []
        for method in config["methods"]:
            if method == "base":
                if any(
                    not re.fullmatch(
                        r"[0-9a-f]{40}", config.get(key, {}).get(model, "")
                    )
                    for key in ["base_revisions", "tokenizer_revisions"]
                ):
                    report["errors"].append(
                        f"{model}/base: pinned base and tokenizer commit revisions are required"
                    )
                continue
            try:
                location = config["checkpoints"][model][method]
                if not location:
                    raise ValueError("A separately trained checkpoint is required")
                metadata = checkpoint_metadata(location, method, model)
                expected_recipe = (
                    config.get("checkpoint_training_recipes", {}).get(model, {}).get(method)
                    if config["protocol"] == "method_specific_sft_v1"
                    else "matched_sft_reference_v3"
                )
                if metadata["provenance"]["training_recipe"] != expected_recipe:
                    raise ValueError("Checkpoint SFT recipe differs from the declared comparison")
                if config["preset"] == "reference" and not metadata["provenance"].get(
                    "hyperparameter_search_complete"
                ):
                    raise ValueError(
                        "Reference runs require a completed validation-selected hyperparameter search"
                    )
                for field, configured in [
                    ("enable_rgag", config.get("enable_rgag", True)),
                    ("angles", {str(k): v for k, v in config["angles"].items()}),
                    ("penalty_data", config["penalty_data"]),
                    ("penalty_assistant_to_user", config["penalty_assistant_to_user"]),
                    ("rotation_direction", config["rotation_direction"]),
                ]:
                    if method == "ha_aside" and metadata[field] != configured:
                        raise ValueError(
                            f"Checkpoint {field} differs from experiment configuration"
                        )
                path = resolve(location)
                artifact_hashes = {
                    p.name: file_hash(p)
                    for p in sorted(path.iterdir())
                    if p.is_file()
                    and p.name
                    not in {
                        "optimizer.pt",
                        "scheduler.pt",
                        "rng_state.pth",
                        "scaler.pt",
                        "training_args.bin",
                    }
                }
                report["checkpoints"][f"{model}:{method}"] = {
                    "path": str(path),
                    "metadata": metadata,
                    "artifact_sha256": artifact_hashes,
                }
                p = metadata["provenance"]
                base_identities.append(tuple(p[k] for k in ["base_model", "tokenizer_id", "base_revision", "tokenizer_revision"]))
                for configured, key in [("base_revisions", "base_revision"), ("tokenizer_revisions", "tokenizer_revision")]:
                    if config.get(configured, {}).get(model) and config[configured][model] != p[key]:
                        raise ValueError("Checkpoint revision differs from the configured Base/tokenizer reference")
                matched.append(
                    tuple(
                        p[k]
                        for k in [
                            "base_model",
                            "tokenizer_id",
                            "base_revision",
                            "tokenizer_revision",
                            "training_data_sha256",
                            "validation_data_sha256",
                            "training_recipe",
                            "loss_scope",
                            "training_seed",
                        ]
                    )
                )
            except Exception as e:
                report["errors"].append(f"{model}/{method}: {type(e).__name__}: {e}")
        if len(set(base_identities)) > 1:
            report["errors"].append(f"{model}: checkpoints do not share pinned Base/tokenizer identities")
        if config["protocol"] == "matched_sft_reference_v3" and len(set(matched)) > 1:
            report["errors"].append(
                f"{model}: checkpoint training provenance differs between comparison methods"
            )
    if config["preset"] == "reference" and config["sep_split"] != "full":
        report["errors"].append(
            "Reference protocol uses the paper's full 9160-pair SEP file; select smoke for development splits"
        )
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        report["errors"].append(
            "CUDA is unavailable in this environment; use the GPU server for the configured run"
        )
    report["ready"] = not report["errors"]
    return report


def runtime_manifest(config, preflight_report):
    source = {}
    for folder in [PROJECT / "ha_aside", PROJECT / "experiments"]:
        for p in sorted(folder.glob("*.py")):
            source[str(p.relative_to(ROOT))] = file_hash(p)
    for relative in [
        "evaluations/safety_evals/rules.py",
        "evaluations/struq/test_on_struq.py",
        "evaluations/safety_evals/test_safety.py",
        "data/prompt_templates.json",
        "model_api.py",
    ]:
        source["aside-original/experiments/" + relative] = file_hash(
            ORIGINAL / relative
        )
    for folder in ["evaluations/BIPIA/bipia/data", "evaluations/BIPIA/bipia/metrics"]:
        for p in sorted((ORIGINAL / folder).rglob("*.py")):
            source[str(p.relative_to(ROOT))] = file_hash(p)
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": config["device"],
        "dtype": config["dtype"],
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "attention_backend": "eager",
        "quantization": None,
    }
    env["package_versions"] = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "scikit-learn",
            "datasets",
            "accelerate",
            "jsonlines",
            "langdetect",
            "thefuzz",
        ]
    }
    return {
        "schema_version": 4,
        "config": config,
        "inputs": preflight_report,
        "source_sha256": source,
        "environment": env,
        "protocol_deviations": [
            "Explicit structured roles and assistant-identity generation replace ambiguous suffix routing in the original source",
            "HA hierarchy penalty is assistant-query to user-key",
            "Safety primary scoring preserves ASIDE source behavior; corrected BIPIA language and RuLES prefix scores are labeled separately",
            "BIPIA uses greedy decoding from paper Appendix D; original safety CLI defaults to sampling and does not explicitly pass the stated 0.7 temperature",
            "Direct sampling explicitly fixes temperature 0.7, top_p 1.0 and the configured top_k (default 50); original source inherits top_k from the checkpoint/Transformers",
            "Safety prompt text uses the original evaluation template; provenance chunks and model-family EOS remain explicit rather than copying suffix routing and a hardcoded Llama EOS into other families",
            "Safety evaluation tokenizes the complete rendered chat once for all methods; fast-tokenizer offsets supply roles, replacing both prior per-span tokenization and original method-dependent chat splitting",
            "Mistral merges system content into its first user message",
            "Local AlpacaFarm utility export uses all 208 nonempty-input examples and retains their input; judge results are separate",
        ],
    }


def tensors(encoded, device):
    return {
        k: encoded[k].to(device)
        for k in ["input_ids", "segment_ids", "binary_data_mask", "attention_mask"]
    }


def actual_data_attention(model, encoded):
    """Cache the prefix, then measure actual final-query attention in every layer."""
    inputs = tensors(encoded, model.device)
    n = inputs["input_ids"].shape[1]
    with torch.no_grad():
        prefix = {k: v[:, :-1] for k, v in inputs.items()}
        cached = model(**prefix, use_cache=True).past_key_values if n > 1 else None
        output = model(
            **{**inputs, "input_ids": inputs["input_ids"][:, -1:]},
            past_key_values=cached,
            use_cache=True,
            output_attentions=True,
        )
    if not output.attentions or any(a is None for a in output.attentions):
        raise RuntimeError(
            "Attention tensors unavailable; isolation cannot be scored as zero"
        )
    indices = inputs["segment_ids"][0] == ROLE_DATA
    if not indices.any():
        raise ValueError("Attention diagnostic requires actual data tokens")
    masses = [a[0, :, -1, indices].float().sum(-1) for a in output.attentions]
    if any(not torch.isfinite(a).all() for a in masses):
        raise RuntimeError("Non-finite measured attention")
    return {
        "definition": "total probability mass on DATA keys from the final prompt query",
        "units": "percent",
        "data_token_count": int(indices.sum()),
        "query_role": int(inputs["segment_ids"][0, -1]),
        "layer_head_mean_mass_percent": [a.mean().item() * 100 for a in masses],
        "layer_head_max_mass_percent": [a.max().item() * 100 for a in masses],
    }


def generate(model, tokenizer, encoded, config, suite, seed, generator=None):
    inputs = tensors(encoded, model.device)
    context = getattr(model.config, "max_position_embeddings", None)
    if context and inputs["input_ids"].shape[1] + config["max_new_tokens"] > context:
        raise ValueError(
            "Prompt plus generation budget exceeds the model context; no truncation is allowed"
        )
    settings = generation_settings(suite, config)
    if generator is None:
        generator = torch.Generator(device=model.device).manual_seed(seed)
    start = time.perf_counter()
    ids, roles = model.generate(
        **inputs,
        max_new_tokens=config["max_new_tokens"],
        do_sample=settings["do_sample"],
        temperature=settings["temperature"] if settings["do_sample"] else 0.7,
        top_k=settings["top_k"],
        generator=generator,
        use_cache=True,
    )
    continuation = ids[0, inputs["input_ids"].shape[1] :]
    return {
        "output": tokenizer.decode(continuation, skip_special_tokens=True),
        "output_ids": continuation.tolist(),
        "generated_tokens": len(continuation),
        "token_budget_reached": len(continuation) == config["max_new_tokens"],
        "seconds": time.perf_counter() - start,
        "rendered_prompt": encoded["rendered_prompt"],
        "role_spans": encoded["spans"],
        "generation_settings": settings,
        "tokenization": encoded.get("tokenization", "legacy_provenance_chunks"),
        **({"generation_rng_after": base64.b64encode(bytes(generator.get_state().tolist())).decode("ascii")}
           if settings["do_sample"] else {}),
    }


def evaluate_example(model, tokenizer, example, config, seed, model_key, generator=None):
    merge = model_key == "mistral7b_v03"
    encoder = encode_safety_task if example.suite in SAFETY_SUITES else encode_task
    encode = lambda instruction, data, direct=False: encoder(
        tokenizer,
        instruction,
        data,
        config["max_input_tokens"],
        direct=direct,
        merge_system=merge,
    )
    encoded = encode(example.instruction, example.data, example.direct)
    result = generate(model, tokenizer, encoded, config, example.suite, seed, generator=generator)
    if example.suite == "sep":
        from experiments.metrics import contains

        paired = generate(
            model,
            tokenizer,
            encode(
                example.metadata["paired_instruction"], example.metadata["paired_data"]
            ),
            config,
            "sep",
            seed,
        )
        result.update(
            paired=paired,
            data_executed=contains(result["output"], example.witness),
            instruction_executed=contains(paired["output"], example.witness),
        )
    elif example.suite == "clean_qa":
        result["qa"] = qa_scores(result["output"], example.answers)
    elif example.suite == "alpacaeval":
        result["judge_status"] = "pending_external_alpacaeval_1_judge"
    elif example.suite == "multiturn":
        from experiments.metrics import contains

        history = task_messages(example.instruction, example.data)
        turns = []
        current, current_encoded = result, encoded
        for turn in range(1, example.metadata["turns"] + 1):
            turns.append(
                {
                    "turn": turn,
                    **current,
                    "attack_success": contains(current["output"], example.witness),
                    "data_attention": actual_data_attention(model, current_encoded),
                }
            )
            if turn != example.metadata["turns"]:
                history.append(
                    ("assistant", [Chunk(current["output"], ROLE_ASSISTANT)])
                )
                history.append(
                    ("user", [Chunk(example.metadata["followup"], ROLE_USER)])
                )
                current_encoded = encode_chunks(
                    tokenizer,
                    render_chunks(tokenizer, history, merge),
                    config["max_input_tokens"],
                )
                current = generate(
                    model, tokenizer, current_encoded, config, "multiturn", seed
                )
        result["turns"] = turns
    else:
        result["attack_success"] = score_attack(example, result["output"])
        if example.suite in {"rules", "bipia_text", "bipia_code"}:
            result["attack_success_corrected"] = score_attack(example, result["output"], corrected=True)
            result["primary_scorer"] = "aside_source_compatible"
            result["corrected_scorer"] = "language_finally_and_literal_response_prefix_fixes"
    return result


def summarize(rows, suite):
    if suite == "sep":
        return sep_metrics(
            [r["data_executed"] for r in rows],
            [r["instruction_executed"] for r in rows],
        )
    if suite == "clean_qa":
        return {
            "answer_substring": proportion([r["qa"]["answer_substring"] for r in rows]),
            "normalized_exact_match": proportion(
                [r["qa"]["normalized_exact_match"] for r in rows]
            ),
            "mean_token_f1": statistics.mean(r["qa"]["token_f1"] for r in rows),
        }
    if suite == "alpacaeval":
        return {
            "n": len(rows),
            "win_rate": None,
            "status": "pending_external_alpacaeval_1_judge",
        }
    if suite == "multiturn":
        return {
            f"turn_{t}": proportion([r["turns"][t - 1]["attack_success"] for r in rows])
            for t in range(1, 5)
        }
    result = {"micro_asr": proportion([r["attack_success"] for r in rows])}
    if all("attack_success_corrected" in r for r in rows):
        result["corrected_micro_asr"] = proportion([r["attack_success_corrected"] for r in rows])
    grouping = (
        "scenario"
        if suite.startswith("bipia_")
        else "attack"
        if suite.startswith("struq_")
        else None
    )
    if grouping:
        groups = sorted({r["example"]["metadata"][grouping] for r in rows})
        result["groups"] = {
            g: proportion(
                [
                    r["attack_success"]
                    for r in rows
                    if r["example"]["metadata"][grouping] == g
                ]
            )
            for g in groups
        }
        result["macro_asr_percent"] = statistics.mean(
            x["percent"] for x in result["groups"].values()
        )
        if "corrected_micro_asr" in result:
            result["corrected_groups"] = {
                g: proportion([r["attack_success_corrected"] for r in rows
                               if r["example"]["metadata"][grouping] == g])
                for g in groups
            }
            result["corrected_macro_asr_percent"] = statistics.mean(
                x["percent"] for x in result["corrected_groups"].values())
    if suite.startswith("struq_"):
        result["robust_accuracy_percent"] = 100 - result["micro_asr"]["percent"]
        for group in result["groups"].values():
            group["robust_accuracy_percent"] = 100 - group["percent"]
    return result


def aggregate_seeds(per_seed):
    """Keep the paper's per-attack StruQ columns and scenario-macro BIPIA score."""
    result = {"per_seed": per_seed}
    first = next(iter(per_seed.values()))
    if "micro_asr" not in first:
        return result
    primary = [v.get("macro_asr_percent", v["micro_asr"]["percent"]) for v in per_seed.values()]
    result.update(mean_asr_percent=statistics.mean(primary),
                  std_across_seeds_percent=statistics.pstdev(primary),
                  aggregation="macro across BIPIA scenarios / StruQ attacks; direct uses micro")
    if "groups" in first:
        groups = set(first["groups"])
        if any(set(v["groups"]) != groups for v in per_seed.values()):
            raise ValueError("Benchmark groups differ across seeds")
        result["groups"] = {}
        for name in sorted(groups):
            values = [v["groups"][name]["percent"] for v in per_seed.values()]
            group = {"mean_asr_percent": statistics.mean(values),
                     "std_asr_percent": statistics.pstdev(values),
                     "n_per_seed": {seed: v["groups"][name]["n"] for seed, v in per_seed.items()}}
            if "robust_accuracy_percent" in first:
                group.update(mean_robust_accuracy_percent=100 - statistics.mean(values),
                             std_robust_accuracy_percent=statistics.pstdev(values))
            result["groups"][name] = group
    if "corrected_micro_asr" in first:
        values = [v.get("corrected_macro_asr_percent", v["corrected_micro_asr"]["percent"])
                  for v in per_seed.values()]
        result["corrected_mean_asr_percent"] = statistics.mean(values)
        result["corrected_std_across_seeds_percent"] = statistics.pstdev(values)
        result["primary_scorer"] = "aside_source_compatible"
    if "robust_accuracy_percent" in first:
        result["mean_robust_accuracy_percent"] = 100 - statistics.mean(primary)
        result["std_robust_accuracy_percent"] = statistics.pstdev(primary)
    return result


def run(config, report):
    if not report["ready"]:
        raise ValueError(
            "Preflight failed; resolve every reported dataset/checkpoint issue first"
        )
    device = config["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; no automatic CPU/model substitution"
        )
    manifest = runtime_manifest(config, report)
    run_id = digest(manifest)
    output = resolve(config["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest
    ):
        raise ValueError("Existing run manifest differs")
    write_json(manifest_path, manifest)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["dtype"]]
    if device == "cpu" and dtype != torch.float32:
        raise ValueError("CPU reference tests require float32")
    metrics = {}
    for model_key in config["models"]:
        for method in config["methods"]:
            if method == "base":
                from transformers import AutoTokenizer, AutoModelForCausalLM
                from ha_aside.hf_adapters import HAASIDEModelWrapper

                base_id, tok_id = REGISTRY[model_key]
                # Base-only runs need an explicitly pinned revision for run identity.
                revision = config.get("base_revisions", {}).get(model_key)
                if not revision or not config.get("tokenizer_revisions", {}).get(
                    model_key
                ):
                    raise ValueError(
                        "Pin base and tokenizer commit revisions before evaluating a Base reference"
                    )
                tokenizer = AutoTokenizer.from_pretrained(
                    tok_id,
                    revision=config["tokenizer_revisions"][model_key],
                    trust_remote_code=False,
                )
                base = AutoModelForCausalLM.from_pretrained(
                    base_id,
                    revision=revision,
                    dtype=dtype,
                    attn_implementation="eager",
                    trust_remote_code=False,
                ).to(device)
                model = HAASIDEModelWrapper(base, mode="vanilla").to(device).eval()
            else:
                model, tokenizer, metadata = load_checkpoint(
                    config["checkpoints"][model_key][method],
                    method,
                    model_key,
                    dtype,
                    device,
                )
            key = f"{model_key}/{method}"
            metrics[key] = {}
            for suite in config["suites"]:
                if method == "base" and suite not in {"sep", "alpacaeval", "clean_qa"}:
                    continue
                per_seed = {}
                for seed in suite_seeds(config, suite):
                    examples = load_suite(
                        suite,
                        seed,
                        resolve(config["prepared_dir"]),
                        config["sep_split"],
                        config.get("sample_limit"),
                        bipia_filter=config.get("bipia_attack_filter", "aside_source_regex"),
                    )
                    log = output / model_key / method / f"{suite}_{seed}.jsonl"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    rows = []
                    if log.exists():
                        for line in log.read_text(encoding="utf-8").splitlines():
                            rows.append(
                                json.loads(line)
                            )  # A damaged log fails explicitly; never hide incomplete evidence.
                    expected = {e.sample_id: e for e in examples}
                    seen = set()
                    for index, row in enumerate(rows):
                        if (
                            index >= len(examples)
                            or row["sample_id"] != examples[index].sample_id
                            or row["sample_id"] not in expected
                            or row["sample_id"] in seen
                            or row["example"]
                            != expected[row["sample_id"]].serializable()
                        ):
                            raise ValueError(
                                "Resume log sample identities differ from frozen selection"
                            )
                        seen.add(row["sample_id"])
                    generator = None
                    if suite in DIRECT_SUITES:
                        generator = torch.Generator(device=model.device).manual_seed(seed)
                        if rows:
                            state = base64.b64decode(rows[-1]["generation_rng_after"], validate=True)
                            generator.set_state(torch.tensor(list(state), dtype=torch.uint8))
                    for ex in examples:
                        if ex.sample_id in seen:
                            continue
                        sample_seed = seed if generator is not None else int(digest([seed, ex.sample_id])[:8], 16)
                        try:
                            row = {
                                "sample_id": ex.sample_id,
                                "example": ex.serializable(),
                                "seed": seed,
                                "sample_generation_seed": sample_seed,
                                "rng_policy": "suite_stream" if generator is not None else "greedy_sample_identity",
                                **evaluate_example(
                                    model, tokenizer, ex, config, sample_seed, model_key, generator=generator
                                ),
                            }
                        except Exception as error:
                            write_json(
                                output / "failure.json",
                                {
                                    "model": model_key,
                                    "method": method,
                                    "suite": suite,
                                    "seed": seed,
                                    "sample_id": ex.sample_id,
                                    "error": repr(error),
                                },
                            )
                            raise
                        with log.open("a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(row, ensure_ascii=False, allow_nan=False)
                                + "\n"
                            )
                            f.flush()
                        rows.append(row)
                    if len(rows) != len(examples):
                        raise RuntimeError(
                            "Incomplete suite; summary cannot be published"
                        )
                    per_seed[str(seed)] = summarize(rows, suite)
                    if suite == "alpacaeval":
                        export = [
                            {
                                "instruction": r["example"]["instruction"]
                                + "\nInput:\n"
                                + r["example"]["data"],
                                "output": r["output"],
                                "generator": key + ":" + run_id,
                            }
                            for r in rows
                        ]
                        write_json(log.with_suffix(".alpacaeval.json"), export)
                aggregate = aggregate_seeds(per_seed)
                metrics[key][suite] = aggregate
                write_json(
                    output / "summary.json",
                    {"run_id": run_id, "status": "in_progress", "metrics": metrics},
                )
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_json(
        output / "summary.json",
        {
            "run_id": run_id,
            "status": "complete_generation",
            "alpacaeval_judging": "external_required"
            if "alpacaeval" in config["suites"]
            else "not_requested",
            "metrics": metrics,
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="ha-aside-paper/experiments/config_reference.json"
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="Load weights and execute; default is preflight only",
    )
    p.add_argument(
        "--report", help="Write the preflight report as JSON without loading weights"
    )
    args = p.parse_args()
    config = load_config(args.config)
    report = preflight(config)
    if args.report:
        write_json(resolve(args.report), report)
    if not args.run:
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 2
    print(run(config, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

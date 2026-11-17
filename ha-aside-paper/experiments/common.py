"""Experiment identities and filesystem utilities; paths are workspace-relative."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "ha-aside-paper"
ORIGINAL = ROOT / "aside-original/experiments"
REGISTRY = {
    "llama31_8b": ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
    "llama2_7b": ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"),
    "llama2_13b": ("meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-13b-chat-hf"),
    "qwen25_7b": ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
    "qwen3_8b": ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"),
    "mistral7b_v03": (
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ),
}
SUITES = [
    "sep",
    "tensortrust",
    "gandalf",
    "purple",
    "rules",
    "struq_id",
    "struq_ood",
    "bipia_text",
    "bipia_code",
    "alpacaeval",
    "clean_qa",
    "multiturn",
]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def load_config(path):
    c = json.loads(resolve(path).read_text(encoding="utf-8"))
    if c.get("protocol") == "ha_aside_rq_public_v1":
        from .rq_runner import validate_config
        return validate_config(c)
    if c["protocol"] not in {"matched_sft_reference_v3", "method_specific_sft_v1"}:
        raise ValueError("Unknown comparison protocol")
    if c["protocol"] == "method_specific_sft_v1":
        if c.get("comparison_scope") != "end_to_end_methods_with_different_sft_data":
            raise ValueError("Method-specific SFT must declare the end-to-end comparison scope")
        recipes = c.get("checkpoint_training_recipes", {})
        for model in c["models"]:
            for method in c["methods"]:
                if method != "base" and recipes.get(model, {}).get(method) not in {
                    "matched_sft_reference_v3", "chatqa_multirole_fullseq_2048_v1"
                }:
                    raise ValueError("Declare each method's expected checkpoint training recipe")
    # The training/checkpoint contract is unchanged. Version the safety evaluation
    # separately so existing server checkpoint configurations can be upgraded.
    from .safety_protocol import SAFETY_REVISION, BIPIA_FILTERS
    c.setdefault("safety_benchmark", SAFETY_REVISION)
    c.setdefault("bipia_attack_filter", "aside_source_regex")
    c.setdefault("direct_top_k", 50)
    # Older configurations describe the original RGAG-enabled HA-ASIDE variant.
    c.setdefault("enable_rgag", True)
    if type(c["enable_rgag"]) is not bool:
        raise ValueError("enable_rgag must be a boolean")
    if c.get("output_root") == "ha-aside-paper/results/corrected_v3":
        c["output_root"] = "ha-aside-paper/results/corrected_v4"
    if c["safety_benchmark"] != SAFETY_REVISION:
        raise ValueError("Unsupported safety benchmark revision")
    if c["bipia_attack_filter"] not in BIPIA_FILTERS:
        raise ValueError("Unsupported BIPIA selection policy")
    if type(c["direct_top_k"]) is not int or c["direct_top_k"] < 0:
        raise ValueError("direct_top_k must be a nonnegative integer; 0 disables filtering")
    if set(c["models"]) - set(REGISTRY) or set(c["suites"]) - set(SUITES):
        raise ValueError("Unknown models or suites")
    if not c["models"] or not c["suites"]:
        raise ValueError("Select at least one model and suite")
    if c["preset"] not in {"reference", "smoke"}:
        raise ValueError("preset must be reference or smoke")
    if c["preset"] == "reference" and (
        c.get("sample_limit") is not None
        or c["max_new_tokens"] != 1024
        or c["seeds"] != [2024, 2025, 2026]
    ):
        raise ValueError(
            "Reference runs require all samples, 1024 output tokens, and seeds 2024–2026"
        )
    if c["max_input_tokens"] < 1 or c["max_new_tokens"] < 1:
        raise ValueError("Token budgets must be positive")
    if (
        set(c["methods"]) - {"base", "vanilla", "ise", "aside_orig", "ha_aside"}
        or not c["methods"]
    ):
        raise ValueError("Unknown or empty methods")
    if "gpu_server_results_fix_v2_optimal_pi8_errorlog" in str(
        resolve(c["output_root"])
    ):
        raise ValueError("Pre-fix results are immutable evidence")
    return c

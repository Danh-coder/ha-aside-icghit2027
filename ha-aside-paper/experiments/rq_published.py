"""RQ preflight for published baselines without inventing training provenance."""

import copy
import json
from pathlib import Path

from .common import file_hash, resolve
from .rq_staging import comparison_status, pairing_errors
from .published_checkpoints import DESCRIPTOR, public_metadata, tokenizer_signature, PINS, RECIPE


def compatible_backbones(checkpoints):
    """Exact native revisions, compatible published architecture; unknown stays unknown."""
    native = [c["metadata"] for c in checkpoints.values() if c["metadata"].get("source_type") != "author_published"]
    public = [c["metadata"] for c in checkpoints.values() if c["metadata"].get("source_type") == "author_published"]
    ids = {tuple(m["provenance"][k] for k in ["base_model", "tokenizer_id", "base_revision", "tokenizer_revision"]) for m in native}
    if len(ids) > 1:
        raise ValueError("Native methods use different pinned Base/tokenizer revisions")
    if public:
        from transformers import AutoTokenizer
        from training.chatqa import TOKENIZER_ID, TOKENIZER_REVISION
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True, trust_remote_code=False)
        expected = tokenizer_signature(tok)
        for m in public:
            p = m["provenance"]
            if p["base_model"] != "Qwen/Qwen2.5-7B" or p["tokenizer_signature"] != expected:
                raise ValueError("Public backbone or tokenizer is incompatible")
        if native and any(m["provenance"]["base_model"] != "Qwen/Qwen2.5-7B" or
                          m["provenance"]["tokenizer_id"] != TOKENIZER_ID or
                          m["provenance"]["tokenizer_revision"] != TOKENIZER_REVISION for m in native):
            raise ValueError("Native methods differ from the common Qwen tokenizer/backbone")


def preflight(config):
    from .rq_runner import preflight as native_preflight
    public_names = [m for m in config["methods"] if (resolve(config["checkpoints"]["qwen25_7b"].get(m, "")) / DESCRIPTOR).is_file()]
    native = copy.deepcopy(config)
    native["methods"] = [m for m in config["methods"] if m not in public_names]
    report = native_preflight(native)
    if config["preset"] == "reference":
        report["errors"].append("Published baselines use explicit staged/smoke comparison, not controlled reference provenance")
    for name in public_names:
        try:
            if name not in PINS:
                raise ValueError("Unsupported author-published method")
            path = resolve(config["checkpoints"]["qwen25_7b"][name])
            meta = public_metadata(path, name, "qwen25_7b")
            if config["checkpoint_training_recipes"]["qwen25_7b"].get(name) != RECIPE:
                raise ValueError("Published checkpoint recipe was not explicitly declared")
            report["checkpoints"][name] = {"path": str(path), "metadata": meta,
                "training_status": {"status": "author_published", "selection_scope": "author_release; training_details_partly_unreported"},
                "artifact_sha256": {DESCRIPTOR: file_hash(path / DESCRIPTOR),
                                    **{k: v["sha256"] for k, v in meta["public_checkpoint"]["files"].items()}}}
        except Exception as error:
            report["errors"].append(f"{name}: {type(error).__name__}: {error}")
    try:
        compatible_backbones(report["checkpoints"])
    except Exception as error:
        report["errors"].append(f"backbone/tokenizer: {error}")
    report["errors"] = list(dict.fromkeys([*report["errors"], *pairing_errors(report["checkpoints"])]))
    report["comparison"] = comparison_status(config, report["checkpoints"])
    if public_names:
        report["comparison"]["base_revision_note"] = "Public pretrained Base commit is unreported; architecture/tokenizer checked, exact origin revision unverified"
    report["ready"] = not report["errors"]
    return report

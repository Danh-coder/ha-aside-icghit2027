"""Checkpoints are method-specific artifacts, never registry fallbacks."""

import json
import math
from pathlib import Path
import torch
from .common import REGISTRY, resolve, file_hash, digest
from ha_aside.hf_adapters import HAASIDEModelWrapper


def checkpoint_metadata(path, method, model_key):
    path = resolve(path)
    if (path / "published_checkpoint.json").is_file():
        from .published_checkpoints import public_metadata
        return public_metadata(path, method, model_key)
    metadata = json.loads((path / "architecture.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 2 or metadata["method"] != method:
        raise ValueError(f"Checkpoint method mismatch at {path}")
    provenance = metadata["provenance"]
    if (
        provenance.get("base_model") != REGISTRY[model_key][0]
        or provenance.get("tokenizer_id") != REGISTRY[model_key][1]
    ):
        raise ValueError(f"Checkpoint/model/tokenizer mismatch at {path}")
    for key in [
        "base_revision",
        "tokenizer_revision",
        "training_data_sha256",
        "validation_data_sha256",
        "training_recipe",
        "loss_scope",
        "selected_validation_loss",
        "training_seed",
    ]:
        if key not in provenance:
            raise ValueError(f"Missing checkpoint provenance {key}: {path}")
    if provenance["loss_scope"] != "full_sequence":
        raise ValueError("Supported SFT recipes require full-sequence loss")
    if provenance["training_recipe"] == "chatqa_multirole_fullseq_2048_v1":
        from training.chatqa import RECIPE, TOKENIZER_ID, TOKENIZER_REVISION, ROLE_POLICY
        data = provenance.get("dataset_provenance", {})
        if (model_key != "qwen25_7b" or provenance.get("dataset") != "chatqa_multirole"
                or provenance.get("max_length") != 2048
                or provenance.get("training_truncation") != "disabled_whole_conversations_only"
                or provenance.get("role_policy") != ROLE_POLICY
                or provenance["tokenizer_revision"] != TOKENIZER_REVISION
                or data.get("tokenizer_id") != TOKENIZER_ID
                or data.get("tokenizer_revision") != TOKENIZER_REVISION
                or data.get("training_recipe") != RECIPE
                or data.get("truncation") is not False
                or data.get("max_length") != 2048
                or not provenance.get("dataset_manifest_sha256")
                or provenance.get("dataset_manifest_canonical_sha256") != digest(data)
                or provenance["training_data_sha256"] != data.get("files", {}).get("train.jsonl", {}).get("sha256")
                or provenance["validation_data_sha256"] != data.get("files", {}).get("validation.jsonl", {}).get("sha256")):
            raise ValueError("ChatQA checkpoint has incompatible dataset/tokenizer/length provenance")
        if provenance.get("pilot_max_steps") is not None and provenance.get("hyperparameter_search_complete"):
            raise ValueError("A bounded pilot cannot be a completed validation search")
    elif provenance["training_recipe"] != "matched_sft_reference_v3":
        raise ValueError("Unsupported checkpoint SFT recipe")
    if provenance.get("selected_validation_loss") is None or not math.isfinite(
        provenance["selected_validation_loss"]
    ):
        raise ValueError("Checkpoint has no measured validation loss")
    if type(metadata.get("enable_rgag")) is not bool:
        raise ValueError("Checkpoint enable_rgag must be a boolean")
    if ("enable_rgag" in provenance
            and provenance["enable_rgag"] != metadata["enable_rgag"]):
        raise ValueError("Checkpoint RGAG setting differs from training provenance")
    if (
        metadata["generated_role"] != ("assistant_segment_embedding" if method == "ise_4role" else "assistant_identity")
        or metadata["hierarchy_policy"] != "assistant_queries_to_user_keys"
    ):
        raise ValueError(
            "Checkpoint role policy is incompatible with the corrected runtime"
        )
    if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
        raise ValueError(f"Checkpoint has no saved model weights/config: {path}")
    if method == "ise" and not (path / "ise_offset.pt").is_file():
        raise ValueError("ISE offset was not saved")
    if method == "ise_4role":
        if metadata.get("segment_role_order") != ["system", "user", "data", "assistant"]:
            raise ValueError("ISE-4 segment role order mismatch")
        if metadata.get("assistant_header_policy") != "assistant_message_framing":
            raise ValueError("ISE-4 assistant header policy mismatch")
        if provenance.get("assistant_header_policy") != "assistant_message_framing":
            raise ValueError("ISE-4 training did not declare the matching assistant header policy")
        if not (path / "segment_embeddings.pt").is_file():
            raise ValueError("ISE-4 segment embeddings were not saved")
    return metadata


def load_checkpoint(path, method, model_key, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve(path)
    if (path / "published_checkpoint.json").is_file():
        from .published_checkpoints import load_public
        return load_public(path, method, model_key, dtype, device)
    m = checkpoint_metadata(path, method, model_key)
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=False,
        local_files_only=True,
    ).to(device)
    wrapper = HAASIDEModelWrapper(
        model,
        mode=method,
        angles={int(k): v for k, v in m["angles"].items()},
        rgag_penalty_data=m["penalty_data"],
        rgag_penalty_user_to_sys=m["penalty_assistant_to_user"],
        enable_rgag=m["enable_rgag"],
        rotation_direction=m["rotation_direction"],
    )
    if method == "ise":
        wrapper.ise_offset.data.copy_(
            torch.load(path / "ise_offset.pt", map_location="cpu", weights_only=True)
        )
    if method == "ise_4role":
        weights = torch.load(path / "segment_embeddings.pt", map_location="cpu", weights_only=True)
        if weights.shape != wrapper.segment_embeddings.weight.shape or not torch.isfinite(weights).all():
            raise ValueError("Invalid ISE-4 segment embedding weights")
        wrapper.segment_embeddings.weight.data.copy_(weights)
    wrapper.to(device).eval()
    return wrapper, tokenizer, m

"""Pinned author checkpoints; public provenance stays distinct from local SFT."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import torch

from .common import REGISTRY, digest, file_hash, write_json
from training.chatqa import TOKENIZER_ID, TOKENIZER_REVISION

DESCRIPTOR = "published_checkpoint.json"
PINS = {
    "vanilla": {"repo_id": "ISTA-MLCV/Qwen2.5-7B_single_emb", "revision": "f73f5bf71de38503853b81ff8993cffebb974cd3", "architecture": "Qwen2ForCausalLM"},
    "aside_orig": {"repo_id": "ISTA-MLCV/Qwen2.5-7B_aside", "revision": "eaaaa196b9159b9e5ca686994e1a97110d988913", "architecture": "QwenForwardRot"},
    "ise": {"repo_id": "ISTA-MLCV/Qwen2.5-7B_ise", "revision": "6e9f2fe749572ed2200de4490fc62be1ef55ec8b", "architecture": "QwenISE"},
}
RECIPE = "author_published_aside_qwen25_v1"
ENCODING = "DATA=segment1; SYSTEM/USER/ASSISTANT=segment0; saved rotation or both learned ISE vectors"


def tokenizer_signature(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    return digest({"backend": {k: backend.get(k) for k in ["model", "normalizer", "pre_tokenizer", "post_processor", "decoder", "added_tokens"]},
                   "chat_template": tokenizer.chat_template, "eos_token_id": tokenizer.eos_token_id,
                   "pad_token_id": tokenizer.pad_token_id, "length": len(tokenizer)})


def checked_relative(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe public snapshot filename")
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Public snapshot file escapes its root")
    return path


def validate_index(config, index, method):
    expected = {"model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 28,
                "num_attention_heads": 28, "num_key_value_heads": 4, "vocab_size": 152064}
    if any(config.get(k) != v for k, v in expected.items()) or config.get("architectures") != [PINS[method]["architecture"]]:
        raise ValueError("Published backbone/config differs from the pinned Qwen2.5-7B release")
    if any(config.get(k) for k in ["learned_rotation", "gradual_rotation", "add_linear_shift"]):
        raise ValueError("Unsupported experimental published transform")
    if method == "aside_orig" and config.get("rotation_direction") != "right":
        raise ValueError("Published adapter expects the author's right-multiply rotation")
    weights = index["weight_map"]
    expected_keys = {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}
    for i in range(28):
        prefix = f"model.layers.{i}."
        expected_keys.update(prefix + suffix for suffix in [
            "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
            "mlp.up_proj.weight", "mlp.down_proj.weight", "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.o_proj.weight"])
    extra = {"aside_orig": {"rotation_matrix"}, "ise": {"segment_embedding.weight"}, "vanilla": set()}[method]
    if set(weights) != expected_keys | extra:
        raise ValueError("Unexpected or missing backbone/transform tensor keys")
    for filename in weights.values():
        if not filename.endswith(".safetensors") or PurePosixPath(filename).name != filename:
            raise ValueError("Unsupported checkpoint shard path")
    return sorted(set(weights.values()))


def prepare(method, output, *, download=False):
    """Preview first; only explicit download fetches the complete model weights."""
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    if method not in PINS:
        raise ValueError("Only the three pinned author baselines are supported")
    pin, output = PINS[method], Path(output)
    info = HfApi().model_info(pin["repo_id"], revision=pin["revision"], files_metadata=True)
    if info.sha != pin["revision"]:
        raise ValueError("Hub returned another checkpoint revision")
    sizes = {f.rfilename: f.size for f in info.siblings}
    if not download:
        return {"method": method, **pin, "download_bytes": sum(v or 0 for v in sizes.values()), "download_started": False}
    if (output / DESCRIPTOR).exists():
        return verify_public(output, method)
    if output.exists() and any(p.name not in {*sizes, ".cache"} for p in output.iterdir()):
        raise FileExistsError("Use a dedicated public-model folder; existing unrelated artifacts are not overwritten")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(pin["repo_id"], revision=pin["revision"], local_dir=output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = validate_index(config, index, method)
    files = {}
    for entry in info.siblings:
        path = checked_relative(output, entry.rfilename)
        if entry.rfilename.endswith(".safetensors"):
            print(f"[verify {method}] {entry.rfilename}", flush=True)
        actual_sha = file_hash(path)
        if path.stat().st_size != entry.size:
            raise ValueError("Truncated public checkpoint file")
        if entry.lfs:
            expected = getattr(entry.lfs, "sha256", None)
            if not expected or actual_sha != expected:
                raise ValueError(f"Public LFS checksum mismatch: {entry.rfilename}")
        else:
            raw = path.read_bytes()
            git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if git_blob != entry.blob_id:
                raise ValueError(f"Public Git blob mismatch: {entry.rfilename}")
        files[entry.rfilename] = {"sha256": actual_sha, "bytes": path.stat().st_size}
    canonical = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, trust_remote_code=False)
    source = AutoTokenizer.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    signature = tokenizer_signature(canonical)
    if tokenizer_signature(source) != signature:
        raise ValueError("Published tokenizer does not match the frozen common tokenizer/template")
    descriptor = {"schema_version": 1, "source_type": "author_published", "method": method, **pin,
                  "files": files, "shards": shards, "tokenizer_signature": signature,
                  "evaluation_tokenizer": {"repo_id": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
                  "source_bos_token_id": source.bos_token_id, "encoding_policy": ENCODING,
                  "training_provenance": {"base_model": "Qwen/Qwen2.5-7B", "base_revision": None,
                                          "training_data_sha256": None, "training_seed": None,
                                          "selected_validation_loss": None,
                                          "note": "Author-published checkpoint; unreported training fields remain unknown"}}
    descriptor["fingerprint"] = digest(descriptor)
    write_json(output / DESCRIPTOR, descriptor)
    return descriptor


def verify_public(path, method):
    path = Path(path)
    descriptor = json.loads((path / DESCRIPTOR).read_text(encoding="utf-8"))
    if descriptor.get("method") != method or method not in PINS or descriptor.get("schema_version") != 1:
        raise ValueError("Public checkpoint method/schema mismatch")
    if descriptor.get("source_type") != "author_published" or any(descriptor.get(k) != v for k, v in PINS[method].items()):
        raise ValueError("Public checkpoint is not the pinned author release")
    if digest({k: v for k, v in descriptor.items() if k != "fingerprint"}) != descriptor.get("fingerprint"):
        raise ValueError("Public descriptor fingerprint mismatch")
    if descriptor.get("evaluation_tokenizer") != {"repo_id": TOKENIZER_ID, "revision": TOKENIZER_REVISION}:
        raise ValueError("Published checkpoint uses another evaluation tokenizer")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = validate_index(config, index, method)
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", *shards}
    if not required <= descriptor["files"].keys() or descriptor.get("shards") != shards:
        raise ValueError("Public descriptor omits required files")
    for relative, expected in descriptor["files"].items():
        file = checked_relative(path, relative)
        if not file.is_file() or file.stat().st_size != expected["bytes"] or file_hash(file) != expected["sha256"]:
            raise ValueError(f"Modified/missing public checkpoint file: {relative}")
    return descriptor


def public_metadata(path, method, model_key):
    if model_key != "qwen25_7b":
        raise ValueError("Public loader only supports the pinned Qwen2.5-7B baselines")
    descriptor = verify_public(path, method)
    return {"source_type": "author_published", "method": method, "enable_rgag": False,
            "public_checkpoint": descriptor,
            "provenance": {**descriptor["training_provenance"], "training_recipe": RECIPE,
                           "tokenizer_id": TOKENIZER_ID, "tokenizer_revision": TOKENIZER_REVISION,
                           "tokenizer_signature": descriptor["tokenizer_signature"]}}


def load_public(path, method, model_key, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open
    from ha_aside.published_adapters import PublishedBinaryWrapper
    path = Path(path)
    metadata = public_metadata(path, method, model_key)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True, trust_remote_code=False)
    if tokenizer_signature(tokenizer) != metadata["provenance"]["tokenizer_signature"]:
        raise ValueError("Cached common tokenizer differs from the imported snapshot audit")
    model, loading = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                     dtype=dtype, attn_implementation="eager", output_loading_info=True)
    extra = {"aside_orig": {"rotation_matrix"}, "ise": {"segment_embedding.weight"}, "vanilla": set()}[method]
    if (loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs")
            or set(loading.get("unexpected_keys", [])) != extra):
        raise ValueError(f"Public backbone weights did not load exactly: {loading}")
    tensors = {}
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for key in extra:
        with safe_open(path / index["weight_map"][key], framework="pt", device="cpu") as weights:
            tensors[key] = weights.get_tensor(key)
    wrapper = PublishedBinaryWrapper(model, method, rotation=tensors.get("rotation_matrix"),
                                     segments=tensors.get("segment_embedding.weight"))
    wrapper.to(device=device, dtype=dtype).eval()
    return wrapper, tokenizer, metadata

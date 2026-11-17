"""Prepare one existing Hermes config; never load model weights or change SFT settings.

Run with the project environment and --measure-lengths for the Qwen tokenizer audit.
The output is a provenance-tagged conversation corpus, not legacy Alpaca input.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "review-artifacts/dataset-selection-2026-10-26"
SOURCE_DIR = AUDIT / "source-snapshots/hermes"
REPO = "NousResearch/hermes-function-calling-v1"
REVISION = "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
SOURCE_FILE = "glaive-function-calling-5k.json"
SOURCE_HASH = "b98eb3f160359f27ad15018e974ce6db444f566eb5be4aa9e4aa690b34d50832"
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ROLES = {"system": ("system", "SYSTEM", 0), "human": ("user", "USER", 1),
         "tool": ("user", "DATA", 2), "gpt": ("assistant", "ASSISTANT", 3)}


def inside(path):
    path = Path(path).resolve()
    path.relative_to(ROOT)
    return path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def write_new(path, data):
    path = inside(path)
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing artifact differs; choose a new output: {path}")
    else:
        path.write_bytes(data)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": sha(data)}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def benchmark_texts():
    files = sorted((ROOT / "ha-aside-paper/data/prepared").glob("*.jsonl"))
    original = ROOT / "aside-original/experiments/data"
    files += [original / name for name in ["SEP_dataset.json", "SEP_evaluation_feb25.json",
              "SEP_validation_feb25.json", "tatsu-lab/alpaca_farm/eval.json"]]
    index, manifest = set(), []
    for path in files:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        objects = ([json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
                   if path.suffix == ".jsonl" else [json.loads(raw)])
        for obj in objects:
            index.update(n for text in strings(obj) if len(n := normalized(text)) >= 40)
        manifest.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(raw)})
    return index, manifest


def unwrap_json(value, tag):
    value = value.strip()
    start, end = f"<{tag}>", f"</{tag}>"
    if not value.startswith(start) or not value.endswith(end):
        raise ValueError(f"Missing {tag} wrapper")
    return json.loads(value[len(start):-len(end)].strip())


def reject_reasons(row):
    conv = row.get("conversations", [])
    reasons = set()
    if not conv or any(m.get("from") not in ROLES or not isinstance(m.get("value"), str)
                       or not m["value"].strip() for m in conv):
        return {"invalid_role_or_empty_content"}
    if conv[0]["from"] != "system" or any(m["from"] == "system" for m in conv[1:]):
        reasons.add("invalid_system_position")
    if conv[-1]["from"] != "gpt":
        reasons.add("not_ending_assistant")
    if not {"system", "human", "gpt", "tool"}.issubset({m["from"] for m in conv}):
        reasons.add("missing_one_of_four_roles")
    try:
        tools = json.loads(row["tools"])
        specs = {tool["function"]["name"]: tool["function"] for tool in tools}
    except (ValueError, KeyError, TypeError):
        return reasons | {"invalid_tool_inventory"}
    for i, message in enumerate(conv):
        if message["from"] != "tool":
            continue
        if not i or conv[i-1]["from"] != "gpt" or i+1 == len(conv) or conv[i+1]["from"] != "gpt":
            reasons.add("tool_not_between_assistant_messages")
            continue
        try:
            call = unwrap_json(conv[i-1]["value"], "tool_call")
            result = unwrap_json(message["value"], "tool_response")
            if not isinstance(call, dict) or not isinstance(result, dict):
                raise ValueError("Expected named JSON objects")
            if call.get("name") != result.get("name"):
                reasons.add("call_result_name_mismatch")
            if call.get("name") not in specs:
                reasons.add("undeclared_tool_call")
            elif not isinstance(call.get("arguments"), dict):
                reasons.add("arguments_not_object")
            elif not set(specs[call["name"]].get("parameters", {}).get("required", [])).issubset(call["arguments"]):
                reasons.add("missing_required_tool_argument")
        except (ValueError, TypeError, KeyError):
            reasons.add("invalid_tool_call_or_result_json")
    return reasons


def candidate_stats(rows, separate_system=False):
    roles = Counter(m["from"] for row in rows for m in row["conversations"])
    return {"rows": len(rows), "message_roles": dict(roles),
            "with_data_tool": sum(any(m["from"] == "tool" for m in r["conversations"]) for r in rows),
            "empty_message_rows": sum(any(not isinstance(m.get("value"), str) or not m["value"].strip()
                                           for m in r["conversations"]) for r in rows),
            "system_storage": "separate system field" if separate_system else "conversation message"}


def measure(records):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKENIZER, revision=TOKENIZER_REVISION, trust_remote_code=False)
    texts = [tokenizer.apply_chat_template(
        [{"role": m["chat_role"], "content": m["content"]} for m in row["messages"]],
        tokenize=False, add_generation_prompt=False) for row in records]
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    for row, length in zip(records, lengths):
        row["qwen25_rendered_tokens"] = length
    ordered = sorted(lengths)
    return {"tokenizer": TOKENIZER, "revision": TOKENIZER_REVISION,
            "transformers": transformers.__version__, "truncation": False,
            "render_policy": "native user carries original tool_response text; HA DATA kept separately; tools already in system",
            "quantiles": {str(q): ordered[int(q*(len(ordered)-1))] for q in [0, .5, .9, .95, .99, 1]},
            "complete_conversations_within": {str(b): sum(x <= b for x in lengths) for b in [768, 1024, 2048, 4096, 8192]},
            "total_rendered_tokens": sum(lengths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-source", action="store_true")
    parser.add_argument("--measure-lengths", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "ha-aside-paper/data/finetuning/hermes_glaive_multirole_v1")
    args = parser.parse_args()
    output = inside(args.output)
    source = inside(SOURCE_DIR / SOURCE_FILE)
    if not source.exists() and args.download_source:
        with urllib.request.urlopen(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{SOURCE_FILE}", timeout=45) as response:
            raw = response.read(30_000_001)
        if sha(raw) != SOURCE_HASH:
            raise ValueError("Downloaded source does not match the pinned checksum")
        write_new(source, raw)
    raw = source.read_bytes()
    if sha(raw) != SOURCE_HASH:
        raise ValueError("Source checksum mismatch")
    rows = json.loads(raw)
    evaluation, eval_manifest = benchmark_texts()
    kept, rejected, seen = [], [], {}
    for index, row in enumerate(rows):
        reasons = reject_reasons(row)
        signature = sha(json.dumps(row["conversations"], ensure_ascii=False, sort_keys=True).encode("utf-8"))
        if signature in seen:
            reasons.add("duplicate_conversation")
        else:
            seen[signature] = index
        for message in row["conversations"]:
            if message["from"] in {"human", "tool", "gpt"} and normalized(message["value"]) in evaluation:
                reasons.add("exact_text_overlap_with_stored_evaluation")
        if reasons:
            rejected.append({"source_row": index, "source_id": row["id"], "reasons": sorted(reasons)})
            continue
        question = next(m["value"] for m in row["conversations"] if m["from"] == "human")
        messages = []
        for message in row["conversations"]:
            chat, ha, role_id = ROLES[message["from"]]
            messages.append({"source_role": message["from"], "chat_role": chat, "ha_role": ha,
                             "ha_role_id": role_id, "binary_data": ha == "DATA", "content": message["value"]})
        kept.append({"id": row["id"], "group_id": sha(normalized(question).encode("utf-8")),
                     "source": {"dataset": REPO, "config": "glaive_func_calling", "revision": REVISION,
                                "file": SOURCE_FILE, "row": index},
                     "category": row.get("category"), "task": row.get("task"), "messages": messages})
    if not kept:
        raise ValueError("No eligible conversations")
    length_audit = measure(kept) if args.measure_lengths else None
    groups = sorted({r["group_id"] for r in kept}, key=lambda g: sha(("split-seed-42:"+g).encode()))
    validation_groups = set(groups[:max(1, round(len(groups)*.1))])
    splits = {"train": [r for r in kept if r["group_id"] not in validation_groups],
              "validation": [r for r in kept if r["group_id"] in validation_groups]}
    artifacts = []
    for split, records in splits.items():
        artifacts.append(write_new(output / f"{split}.jsonl", "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records)))
    artifacts.append(write_new(output / "excluded_rows.json", json.dumps(rejected, indent=2)+"\n"))
    stats = {}
    for label, path, separate in [
        ("toolace", ROOT/"ha-aside-paper/data/finetuning/toolace_multirole_v1/source/data.json", True),
        ("hermes_func_calling", SOURCE_DIR/"func-calling.json", False),
        ("hermes_glaive_func_calling", source, False)]:
        if path.exists():
            content = path.read_bytes()
            stats[label] = dict(candidate_stats(json.loads(content), separate), file_sha256=sha(content))
    manifest = {"schema": "ha_multirole_conversations_v1", "purpose": "benign adaptation to role-transformed embeddings",
                "status": "prepared corpus; current Alpaca loader does not consume this schema",
                "single_existing_config_only": True, "new_synthetic_content": False,
                "source": {"repo": REPO, "config": "glaive_func_calling", "revision": REVISION,
                           "url": f"https://huggingface.co/datasets/{REPO}", "license_on_card": "apache-2.0",
                           "upstream": "Glaive", "file": source.relative_to(ROOT).as_posix(), "sha256": SOURCE_HASH},
                "raw_rows": len(rows), "kept_rows": len(kept), "excluded_rows": len(rejected),
                "exclusion_reasons_nonexclusive": dict(Counter(reason for row in rejected for reason in row["reasons"])),
                "split_policy": "10 percent of normalized-first-human-message groups; SHA256 ordering seed 42",
                "groups": len(groups), "split_rows": {k: len(v) for k,v in splits.items()},
                "all_kept_rows_have_four_roles": True, "source_content_unchanged": True,
                "loss_scope": "not selected by data preparation; existing training recipe is unchanged",
                "length_audit": length_audit, "candidate_full_file_audits": stats,
                "decontamination": {"scope": "exact NFKC/casefold/whitespace-normalized whole messages >=40 characters versus leaf strings in listed stored evaluation files",
                                     "excluded_matching_rows": sum("exact_text_overlap_with_stored_evaluation" in r["reasons"] for r in rejected),
                                     "files": eval_manifest,
                                     "limitations": "Not a semantic/substring/near-duplicate check; generated benchmark prompts and external datasets not listed are not covered"},
                "quality_limitations": "Automatic structural checks; no claim that every target is factually or semantically correct",
                "artifacts": artifacts, "preparer_sha256": sha(Path(__file__).read_bytes())}
    write_new(output/"manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: manifest[k] for k in ["raw_rows", "kept_rows", "excluded_rows", "split_rows", "exclusion_reasons_nonexclusive", "length_audit"]}, indent=2))


if __name__ == "__main__":
    main()

"""Prepare a single pinned ChatQA source; no generation, rewriting or truncation."""

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ha-aside-paper"))
from experiments.common import digest, file_hash, write_json
from training.chatqa import (DATASET_NAME, RECIPE, TOKENIZER_ID, TOKENIZER_REVISION,
                            MAX_LENGTH, ROLE_POLICY, conversation_chunks, validate_conversation)
from experiments.prompts import PREFACE
from tools.prepare_multirole_dataset import normalized, benchmark_texts

REVISION = "164f018cf5e7862ae5ef9c2b8ada47b07e3e3a31"
SOURCE_SHA = "5ae871d66f231a2130507ac512cb8d4f03b796215d78056f5a6e009442e3082d"
CARD_SHA = "62331559e787d292b7a52b02ce0aa09b3b0370f70a13b893a858337ccf2ff0a8"
NO_ANSWER = normalized("Sorry. I cannot find the answer based on the context.")
META = re.compile(r"\bthe task (?:does not require|is not required)\b", re.I)


class Groups:
    def __init__(self, size):
        self.parents = list(range(size))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def build_corpus(rows, tokenizer, review, evaluation_texts, seed=42):
    """Pure preparation core, shared with offline fixture tests."""
    groups = Groups(len(rows))
    docs, questions, signatures = {}, {}, set()
    duplicate, unsafe_rows, reasons = set(), set(), {}
    for i, row in enumerate(rows):
        validate_conversation(row)
        for text, seen in [(row["document"], docs), (row["messages"][0]["content"], questions)]:
            key = digest(normalized(text))
            if key in seen:
                groups.join(i, seen[key])
            else:
                seen[key] = i
        signature = digest(row)
        if signature in signatures:
            duplicate.add(i)
        signatures.add(signature)
        fields = [row["document"], *[m["content"] for m in row["messages"]], *row["answers"]]
        if any(META.search(t) for t in fields[1:]):
            reasons.setdefault(i, []).append("generation_metadata")
        if any(len(n := normalized(t)) >= 40 and n in evaluation_texts for t in fields):
            reasons.setdefault(i, []).append("exact_stored_benchmark_field_overlap")
    for entry in review:
        i = entry["source_row"]
        if not 0 <= i < len(rows) or digest(rows[i]) != entry["source_row_sha256"]:
            raise ValueError("Semantic screening record does not match pinned source row")
        if entry["decision"] == "exclude_group":
            reasons.setdefault(i, []).append("review:" + entry["reason"])
        elif entry["decision"] != "retain_no_confirmed_error":
            raise ValueError("Unknown or unresolved semantic screening decision")
    unsafe_rows.update(reasons)
    unsafe_groups = {groups.find(i) for i in unsafe_rows}

    @lru_cache(maxsize=150000)
    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    eligible, excluded = [], Counter()
    for i, row in enumerate(rows):
        if i in duplicate:
            excluded["exact_duplicate"] += 1
            continue
        if groups.find(i) in unsafe_groups:
            excluded["quality_or_overlap_group"] += 1
            continue
        chunks = conversation_chunks(row, tokenizer)
        role_counts = Counter()
        for chunk in chunks:
            role_counts[chunk.role] += count(chunk.text)
        if not all(role_counts[r] for r in range(4)):
            raise ValueError("Missing semantic role tokens")
        n_tokens = sum(role_counts.values())
        if n_tokens > MAX_LENGTH:
            excluded["complete_sequence_over_2048"] += 1
            continue
        eligible.append({"source_row": i, "group_id": str(groups.find(i)),
                         "source_row_sha256": digest(row), "n_tokens": n_tokens, **row})
    group_ids = sorted({r["group_id"] for r in eligible}, key=int)
    if len(group_ids) < 2:
        raise ValueError("Need at least two independent groups")
    random.Random(seed).shuffle(group_ids)
    validation_groups = set(group_ids[:max(1, len(group_ids) // 10)])
    splits = {"train": [], "validation": []}
    for row in eligible:
        splits["validation" if row["group_id"] in validation_groups else "train"].append(row)
    stats = {}
    for split, data in splits.items():
        stats[split] = {"rows": len(data), "groups": len({r["group_id"] for r in data}),
                        "multi_turn_rows": sum(len(r["messages"]) > 1 for r in data),
                        "canonical_unanswerable_rows": sum(normalized(r["answers"][0]) == NO_ANSWER for r in data),
                        "tokens": sum(r["n_tokens"] for r in data)}
    return splits, {"raw_rows": len(rows), "excluded_rows_by_first_reason": dict(excluded),
                    "quality_trigger_rows": {str(i): reason for i, reason in reasons.items()},
                    "excluded_groups": sorted(unsafe_groups), "split_stats": stats}


def load_prepared(folder):
    """Fail before model loading if source/data/role/tokenizer provenance differs."""
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["dataset"] != DATASET_NAME or manifest["training_recipe"] != RECIPE
            or manifest["max_length"] != MAX_LENGTH or manifest["tokenizer_id"] != TOKENIZER_ID
            or manifest["tokenizer_revision"] != TOKENIZER_REVISION
            or manifest["system_text_sha256"] != hashlib.sha256(PREFACE.encode("utf-8")).hexdigest()
            or manifest["renderer_source_sha256"] != file_hash(ROOT / "ha-aside-paper/experiments/prompts.py")
            or manifest["conversation_source_sha256"] != file_hash(ROOT / "ha-aside-paper/training/chatqa.py")):
        raise ValueError("Prepared ChatQA rendering/tokenizer/recipe contract differs")
    for name, item in manifest["files"].items():
        path = (folder / name).resolve()
        path.relative_to(folder)
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Prepared ChatQA checksum mismatch: {name}")
    splits = {}
    for split in ["train", "validation"]:
        # str.splitlines() also splits literal U+2028/U+0085 inside JSON strings.
        # JSONL separators are physical LF lines, not Unicode paragraph breaks.
        with (folder / f"{split}.jsonl").open(encoding="utf-8") as handle:
            splits[split] = [json.loads(line) for line in handle]
        if len(splits[split]) != manifest["split_stats"][split]["rows"]:
            raise ValueError("Prepared ChatQA row count differs")
        if not splits[split]:
            raise ValueError("Empty prepared split")
        for row in splits[split]:
            validate_conversation(row)
            original = {key: row[key] for key in ["document", "messages", "answers"]}
            if digest(original) != row["source_row_sha256"] or not 1 < row["n_tokens"] <= MAX_LENGTH:
                raise ValueError("Modified source row or invalid sequence budget")
    train, val = splits["train"], splits["validation"]
    if ({r["group_id"] for r in train} & {r["group_id"] for r in val}
            or {r["source_row"] for r in train} & {r["source_row"] for r in val}):
        raise ValueError("Prepared ChatQA train/validation leakage")
    # Check the grouping invariants independently of the supplied group IDs.
    for field in [lambda r: normalized(r["document"]), lambda r: normalized(r["messages"][0]["content"])]:
        if {field(r) for r in train} & {field(r) for r in val}:
            raise ValueError("Document/conversation crosses prepared split")
    return manifest, splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "ha-aside-paper/data/finetuning" / DATASET_NAME)
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        p.error("Output already exists; prepared corpora are immutable")
    source_dir = ROOT / "review-artifacts/dataset-selection-2026-10-26/source-snapshots/chatqa"
    source = source_dir / "synthetic_convqa_train.json"
    if file_hash(source) != SOURCE_SHA or file_hash(source_dir / "README.md") != CARD_SHA:
        raise ValueError("Pinned ChatQA source/card hash mismatch")
    review_path = ROOT / "review-artifacts/dataset-selection-2026-10-27/chatqa_semantic_decisions.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["source_sha256"] != SOURCE_SHA or len(review["records"]) != 212:
        raise ValueError("Missing completed 212-row screening")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION,
                                            local_files_only=args.local_files_only, trust_remote_code=False)
    evaluation_texts, evaluation_manifest = benchmark_texts()
    rows = json.loads(source.read_text(encoding="utf-8"))
    splits, stats = build_corpus(rows, tokenizer, review["records"], evaluation_texts)
    output.mkdir(parents=True)
    for split, data in splits.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8", newline="\n")
    (output / "SOURCE_CARD.md").write_bytes((source_dir / "README.md").read_bytes())
    write_json(output / "semantic_screening.json", review)
    write_json(output / "filter_report.json", stats)
    manifest = {"schema_version": 1, "dataset": DATASET_NAME, "training_recipe": RECIPE,
                "source_repo": "nvidia/ChatQA-Training-Data", "source_config": "synthetic_convqa",
                "source_split": "train", "source_revision": REVISION, "source_sha256": SOURCE_SHA,
                "source_url": f"https://huggingface.co/datasets/nvidia/ChatQA-Training-Data/blob/{REVISION}/synthetic_convqa/train.json",
                "license_note": "Non-commercial use only, subject to linked OpenAI Terms of Use; see SOURCE_CARD.md",
                "tokenizer_id": TOKENIZER_ID, "tokenizer_revision": TOKENIZER_REVISION,
                "max_length": MAX_LENGTH, "truncation": False, "source_strings_rewritten": False,
                "system_text": PREFACE, "system_text_sha256": hashlib.sha256(PREFACE.encode("utf-8")).hexdigest(), "role_policy": ROLE_POLICY,
                "renderer_source_sha256": file_hash(ROOT / "ha-aside-paper/experiments/prompts.py"),
                "conversation_source_sha256": file_hash(ROOT / "ha-aside-paper/training/chatqa.py"),
                "preparer_source_sha256": file_hash(Path(__file__)),
                "split_seed": 42, "split_policy": "10% connected groups by exact normalized document OR first question, before filtering",
                "split_stats": stats["split_stats"], "stored_evaluation_files": evaluation_manifest,
                "contamination_scope": "exact normalized source-field matches >=40 chars only; no semantic/near-duplicate guarantee",
                "quality_scope": "212-row agent evidence screening plus structural/meta checks; not full-corpus semantic certification",
                "files": {name: {"sha256": file_hash(output / name), "bytes": (output / name).stat().st_size}
                          for name in ["train.jsonl", "validation.jsonl", "SOURCE_CARD.md", "semantic_screening.json", "filter_report.json"]}}
    write_json(output / "manifest.json", manifest)
    load_prepared(output)
    print(json.dumps({"dataset": str(output), **stats}, indent=2))


if __name__ == "__main__":
    main()

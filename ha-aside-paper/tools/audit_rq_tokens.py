"""Freeze common source-case cohorts using the pinned tokenizer, without model weights."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import PROJECT
from experiments.rq_selection import freeze_selection
from training.chatqa import TOKENIZER_ID, TOKENIZER_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PROJECT / "data/prepared/rq_public_v1")
    parser.add_argument("--sources", type=Path, default=PROJECT / "data/public_benchmarks/sources_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION,
                                             local_files_only=args.local_files_only, trust_remote_code=False)
    result = freeze_selection(args.prepared, args.sources, args.output or args.prepared / "selection.v2.json", tokenizer)
    print("Frozen selection:", result["fingerprint"])


if __name__ == "__main__":
    main()

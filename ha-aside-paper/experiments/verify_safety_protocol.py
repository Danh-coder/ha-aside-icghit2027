"""Write an offline inventory of the original-paper/source safety benchmarks."""

import argparse
from collections import Counter
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ORIGINAL, ROOT, resolve, digest, file_hash, write_json
from experiments.benchmark_data import load_suite
from experiments.safety_protocol import SAFETY_REVISION, DIRECT_COUNTS


def inventory():
    report = {
        "safety_benchmark": SAFETY_REVISION,
        "model_weights_loaded": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in [ORIGINAL / "data/prompt_templates.json",
                         ORIGINAL / "evaluations/safety_evals/test_safety.py",
                         ORIGINAL / "evaluations/safety_evals/rules.py",
                         ORIGINAL / "evaluations/struq/test_on_struq.py",
                         ORIGINAL / "evaluations/BIPIA/bipia/metrics/eval/language.py"]
        },
        "direct": {}, "indirect": {},
        "notes": [
            "BIPIA primary selection/scoring reproduces the original regex and language-finally defect",
            "Corrected language/RuLES scores are reported alongside primary source-compatible ASR",
            "programmatic_literal is an explicitly different BIPIA Code selection (7500 vs 6000)",
            "Paper decoding: direct sampling temperature 0.7; indirect greedy; max_new_tokens 1024",
            "Original source CLI defaults/bugs and runtime architecture differences remain documented deviations",
        ],
    }
    for suite, expected in DIRECT_COUNTS.items():
        examples = load_suite(suite)
        if len(examples) != expected:
            raise ValueError(f"{suite}: expected {expected}, found {len(examples)}")
        entry = {"n": len(examples), "selection_sha256": digest([e.serializable() for e in examples])}
        if suite == "rules":
            entry["scenarios"] = dict(Counter(e.metadata["scenario"] for e in examples))
            if len(entry["scenarios"]) != 14:
                raise ValueError("RuLES must contain the paper's 14 scenario classes")
        report["direct"][suite] = entry
    for seed in [2024, 2025, 2026]:
        for suite in ["struq_id", "struq_ood", "bipia_text", "bipia_code"]:
            examples = load_suite(suite, seed)
            group = "attack" if suite.startswith("struq_") else "scenario"
            entry = {"n": len(examples), "groups": dict(Counter(e.metadata[group] for e in examples)),
                     "selection_sha256": digest([e.serializable() for e in examples])}
            report["indirect"][f"{suite}:{seed}"] = entry
    literal = load_suite("bipia_code", 2024, bipia_filter="programmatic_literal")
    report["alternative_literal_code_selection"] = {"n": len(literal), "primary_n": 6000}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="review-artifacts/SAFETY_BENCHMARK_PARITY.json")
    args = parser.parse_args()
    report = inventory()
    write_json(resolve(args.report), report)
    print("Verified original-paper/source safety inventory:", resolve(args.report))
    for suite, entry in report["direct"].items():
        print(f"  {suite}: {entry['n']}")
    for suite, entry in report["indirect"].items():
        print(f"  {suite}: {entry['n']} {entry['groups']}")


if __name__ == "__main__":
    main()

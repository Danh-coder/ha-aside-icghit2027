"""Freeze tokenizer/budget cohorts before seeing any defender model outputs."""

import json
from pathlib import Path

from .common import digest, file_hash, write_json
from .rq_data import load_bundle, load_cases
from .rq_sources import rows
from .rq_transcript import POLICY, assistant, encode_transcript, message, rag_user, segment
from .rq_runner import scenario_specs
from training.chatqa import TOKENIZER_ID, TOKENIZER_REVISION


def rendering_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in ["rq_transcript.py", "rq_runner.py", "rq_selection.py"]}


def prefix_tokens(tokenizer, messages, source_root, *, tool=None, reserve=1024, own_history=False):
    encoded = encode_transcript(tokenizer, messages, 10**9, tool=tool, source_root=source_root)
    n = encoded["input_ids"].shape[1] + reserve
    if own_history:
        # Budget the maximum generated response at every previous A position.
        # Source reference answers are used only to count template overhead;
        # they are never fed to the own-history defender at inference.
        for msg in messages:
            if msg["role"] == "assistant":
                text = "".join(s["text"] for s in msg["segments"])
                actual = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                n += max(0, 1024 - actual) + 4  # conservative boundary allowance
    return n


def freeze_selection(prepared, source_root, output, tokenizer):
    prepared, source_root, output = Path(prepared), Path(source_root), Path(output)
    bundle = load_bundle(prepared)
    identity = {"dataset_fingerprint": bundle["fingerprint"], "tokenizer_id": TOKENIZER_ID,
                "tokenizer_revision": TOKENIZER_REVISION, "context_budget": 16384,
                "rendering_sha256": rendering_hashes(),
                "own_history_bound": "1024 generated tokens per preceding assistant response plus boundary allowance"}
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing["identity"] != identity:
            raise ValueError("Budget selection is immutable; select a new audit output")
        verify_selection(existing, bundle)
        return existing
    allowed, exclusions, counts = {}, [], {}
    for suite in ["role_authority", "iheval", "task_controls", "mtrag_reference", "sysbench", "four_role_multiturn"]:
        cases = load_cases(prepared, suite)
        accepted, pair_failures, pair_members = [], set(), {}
        pool = {p["payload_id"]: p for p in rows(prepared / "payloads.json")} if suite == "four_role_multiturn" else {}
        for i, case in enumerate(cases):
            if i and i % 1000 == 0:
                print(f"Budget audit {suite}: {i}/{len(cases)}", flush=True)
            maxima = []
            if suite in {"role_authority", "iheval", "task_controls"}:
                maxima.append(prefix_tokens(tokenizer, case["messages"], source_root, tool=case.get("tool"), reserve=case.get("max_new_tokens", 1024)))
            elif suite == "mtrag_reference":
                task = case["task"]
                messages = [message("system", segment(POLICY, "system", framing=True)), *task["history"],
                            rag_user(task["question"], task["contexts"], question_ref=task["question_ref"])]
                maxima.append(prefix_tokens(tokenizer, messages, source_root))
            elif suite == "sysbench":
                history = []
                for msg in case["raw"]["messages"]:
                    if msg["role"] == "assistant":
                        maxima.append(prefix_tokens(tokenizer, history, source_root, own_history=True))
                    history.append(message(msg["role"], segment(msg["content"], msg["role"])))
            else:
                for scenario in scenario_specs(case, pool, ["fixed", "own"]):
                    payload = pool[scenario["payload_id"]]
                    system = [segment(POLICY, "system", framing=True)]
                    if payload["goal"] == "secret_extraction":
                        system += [segment("\n", "system", framing=True), segment(payload["protected_instruction"], "system")]
                    history = [message("system", *system)]
                    for task in case["turns"]:
                        active = task["turn"] in scenario["attack_turns"]
                        current = rag_user(task["question"], task["contexts"], payload=payload if active else None,
                                           channel=scenario["channel"], question_ref=task["question_ref"])
                        maxima.append(prefix_tokens(tokenizer, [*history, current], source_root,
                                                     own_history=scenario["history_mode"] == "own"))
                        stored = rag_user(task["question"], [], payload=payload if active and scenario["channel"] == "direct" else None,
                                          channel=scenario["channel"], question_ref=task["question_ref"])
                        history += [stored, assistant(task["references"][0])]
            maximum = max(maxima, default=0)
            if maximum and maximum <= 16384:
                accepted.append(case["case_id"])
            else:
                exclusions.append({"suite": suite, "case_id": case["case_id"], "max_total_tokens_bound": maximum,
                                   "reason": "shared_context_budget" if maximum else "no_eligible_attack_turn"})
            if suite == "role_authority" and case["scorer"]["kind"] == "witness_authority":
                pair = case["case_id"].rsplit(":", 1)[0]
                pair_members.setdefault(pair, []).append(case["case_id"])
                if maximum > 16384:
                    pair_failures.add(pair)
        rejected_pair_members = {c for p in pair_failures for c in pair_members[p]}
        accepted = sorted(set(accepted) - rejected_pair_members)
        allowed[suite] = accepted
        counts[suite] = {"source_cases": len(cases), "retained_cases": len(accepted),
                         "excluded_cases": len(cases) - len(accepted)}
        print(f"Budget audit {suite}: retained {len(accepted)}/{len(cases)}", flush=True)
    selection = {"identity": identity, "allowed": allowed, "coverage": counts, "exclusions": exclusions,
                 "interpretation": "Common frozen cohort for all methods; exclusions are not defense successes"}
    selection["fingerprint"] = digest(selection)
    write_json(output, selection)
    return selection


def verify_selection(selection, bundle):
    fingerprint = selection["fingerprint"]
    if digest({k: v for k, v in selection.items() if k != "fingerprint"}) != fingerprint:
        raise ValueError("Frozen selection fingerprint mismatch")
    expected = {"dataset_fingerprint": bundle["fingerprint"], "tokenizer_id": TOKENIZER_ID,
                "tokenizer_revision": TOKENIZER_REVISION, "context_budget": 16384,
                "rendering_sha256": rendering_hashes()}
    if any(selection["identity"].get(k) != value for k, value in expected.items()):
        raise ValueError("Budget audit does not match source/tokenizer/runtime; create a new audit")
    return selection

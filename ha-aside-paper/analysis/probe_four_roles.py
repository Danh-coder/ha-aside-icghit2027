"""Four-source linear probes on matched public clauses, never SFT examples."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from experiments.common import digest, file_hash, write_json
from experiments.rq_transcript import ROLE_IDS, encode_transcript


def probe(model, tokenizer, cases, config):
    # One occurrence per source clause/role prevents repeated SEP applications
    # from dominating the probe. Entire clauses stay on one side of the split.
    groups = defaultdict(dict)
    for case in cases:
        role = case.get("metadata", {}).get("condition")
        if role in ROLE_IDS and case["scorer"]["kind"] == "witness_authority":
            groups[case["group_id"]].setdefault(role, case)
    complete = {g: values for g, values in groups.items() if set(values) == set(ROLE_IDS)}
    selected = sorted(complete, key=lambda g: digest([config["seed"], g]))
    cap = config.get("probe_group_limit")
    if cap:
        selected = selected[:cap]
    if len(selected) < 4:
        raise ValueError("Four-role probing requires at least four complete source-clause groups")
    split = max(2, min(len(selected) - 2, int(len(selected) * 0.7)))
    features, positions, labels, group_ids = None, [], [], []
    cache_root = Path(config["_probe_cache_dir"]) if config.get("_probe_cache_dir") else None
    if cache_root:
        if not config.get("_probe_checkpoint_signature"):
            raise ValueError("Probe caching requires a pinned checkpoint signature")
        cache_root.mkdir(parents=True, exist_ok=True)
    for group in selected:
        signature = digest({"cases": complete[group], "checkpoint": config.get("_probe_checkpoint_signature"),
                            "implementation": file_hash(__file__), "budget": config["max_input_tokens"]})
        path = cache_root / (digest(group) + ".npz") if cache_root else None
        if path and path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["signature"]) != signature:
                    raise ValueError("Cached probe features belong to different source cases/checkpoint/code")
                parts, group_positions = saved["features"], saved["positions"]
                checksum = hashlib.sha256(parts.tobytes() + group_positions.tobytes()).hexdigest()
                if str(saved["checksum"]) != checksum or parts.ndim != 3 or parts.shape[1] != 4 or group_positions.shape != (4, 4) or not np.isfinite(parts).all():
                    raise ValueError("Probe feature cache is corrupt")
        else:
            parts, group_positions = None, []
            for role in ROLE_IDS:
                case = complete[group][role]
                encoded = encode_transcript(tokenizer, case["messages"], config["max_input_tokens"])
                mask = encoded["probe_mask"][0] & encoded["content_mask"][0]
                indices = mask.nonzero().flatten()
                if not len(indices) or not (encoded["segment_ids"][0, indices] == ROLE_IDS[role]).all():
                    raise ValueError("Matched source clause has missing or mixed probe-role tokens")
                with torch.no_grad():
                    output = model(**{k: encoded[k].to(model.device) for k in ["input_ids", "segment_ids", "binary_data_mask", "attention_mask"]},
                                   use_cache=False, output_hidden_states=True, logits_to_keep=1)
                if output.hidden_states is None:
                    raise RuntimeError("Model did not expose actual hidden states")
                if parts is None:
                    parts = [[] for _ in output.hidden_states]
                for layer, state in enumerate(output.hidden_states):
                    parts[layer].append(state[0, indices.to(state.device)].float().mean(0).cpu().numpy())
                n = encoded["input_ids"].shape[1]
                group_positions.append([indices.float().mean().item(), indices.float().mean().item() / n, len(indices), n])
            parts, group_positions = np.asarray(parts, dtype=np.float32), np.asarray(group_positions, dtype=np.float32)
            if path:
                temp = path.with_suffix(".npz.tmp")
                with temp.open("wb") as stream:
                    np.savez(stream, signature=signature, features=parts, positions=group_positions,
                             checksum=hashlib.sha256(parts.tobytes() + group_positions.tobytes()).hexdigest())
                    stream.flush()
                    os.fsync(stream.fileno())
                temp.replace(path)
        if features is None:
            features = [[] for _ in parts]
        for layer, values in enumerate(parts):
            features[layer].extend(values)
        positions.extend(group_positions)
        labels.extend(ROLE_IDS.values())
        group_ids.extend([group] * len(ROLE_IDS))
        if cache_root:
            write_json(cache_root / "progress.json", {"completed_source_groups": len(labels) // 4,
                       "planned_source_groups": len(selected), "status": "features_cached_not_probe_scores"})
        if config.get("_probe_progress_callback"):
            config["_probe_progress_callback"](len(labels) // 4)
    y = np.asarray(labels)
    training_groups = set(selected[:split])
    train = np.array([group in training_groups for group in group_ids])

    def fit(x):
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite real-model probe activations")
        scaler = StandardScaler().fit(x[train])
        classifier = LogisticRegression(max_iter=1000, random_state=config["seed"])
        classifier.fit(scaler.transform(x[train]), y[train])
        pred = classifier.predict(scaler.transform(x[~train]))
        return {"balanced_accuracy": float(balanced_accuracy_score(y[~train], pred)),
                "macro_f1": float(f1_score(y[~train], pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(y[~train], pred, labels=[0, 1, 2, 3]).tolist(),
                "test_predictions": [{"group_id": group, "gold": int(gold), "prediction": int(value)}
                                     for group, gold, value in zip(np.array(group_ids)[~train], y[~train], pred)]}

    return {"protocol": "rq_four_role_public_clause_probe_v1", "seed": config["seed"],
            "feature_unit": "mean over content tokens of the matched original clause; headers/special tokens excluded",
            "source_groups": selected, "train_groups": selected[:split], "test_groups": selected[split:],
            "role_order": list(ROLE_IDS), "n_groups": len(selected), "n_examples": len(labels),
            "train_examples": int(train.sum()), "test_examples": int((~train).sum()),
            "layers": [{"layer": i, **fit(parts)} for i, parts in enumerate(features)],
            "position_only_baseline": fit(positions),
            "interpretation": "Role information in representations; not a behavioral security guarantee"}

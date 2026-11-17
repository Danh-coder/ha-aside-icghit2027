"""Instruction/data probes on real checkpoint activations and adversarial Alpaca.

Prompt-level split, shared semantic vocabulary, and explicit token sampling.
This evaluates binary role separability, not a security guarantee.
"""

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from experiments.common import ORIGINAL, REGISTRY, write_json, file_hash
from experiments.checkpoints import load_checkpoint
from experiments.prompts import encode_task


def probe(model, tokenizer, model_key, prompts=517, seed=42, tokens_per_role=0):
    rows = json.loads(
        (ORIGINAL / "data/train_data/alpaca_52k_shuffled_train.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [r for r in rows if r["input"].strip()]
    if prompts < 4 or prompts > len(rows):
        raise ValueError(
            "Choose between 4 and the available nonempty-input prompt count"
        )
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), prompts)
    split = int(prompts * 0.7)
    features, labels, groups = None, [], []
    prompt_ids = []
    for group, index in enumerate(indices):
        row = rows[index]
        data = (
            rows[rng.randrange(len(rows))]["instruction"]
            if group % 2 == 0
            else row["input"]
        )
        encoded = encode_task(
            tokenizer,
            row["instruction"],
            data,
            16384,
            merge_system=model_key == "mistral7b_v03",
        )
        roles = encoded["segment_ids"][0]
        instruction = (roles == 1).nonzero().flatten().tolist()
        passive = (roles == 2).nonzero().flatten().tolist()
        k = min(len(instruction), len(passive))
        if tokens_per_role:
            k = min(k, tokens_per_role)
        if not k:
            raise ValueError("Prompt has no balanced instruction/data tokens")
        positions = sorted(rng.sample(instruction, k)) + sorted(rng.sample(passive, k))
        with torch.no_grad():
            outputs = model(
                **{
                    k: encoded[k].to(model.device)
                    for k in [
                        "input_ids",
                        "segment_ids",
                        "binary_data_mask",
                        "attention_mask",
                    ]
                },
                use_cache=False,
                output_hidden_states=True,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        if features is None:
            features = [[] for _ in outputs.hidden_states]
        for layer, state in enumerate(outputs.hidden_states):
            features[layer].append(
                state[0, positions].float().cpu().numpy().astype(np.float16)
            )
        labels.extend([1] * k + [0] * k)
        groups.extend([group] * len(positions))
        prompt_ids.append(row["orig_ix"])
    y, group = np.array(labels), np.array(groups)
    train = group < split
    results = []
    for layer, parts in enumerate(features):
        x = np.concatenate(parts).astype(np.float32)
        classifier = LogisticRegression(
            max_iter=1000, fit_intercept=True, random_state=seed
        )
        classifier.fit(x[train], y[train])
        results.append(
            {
                "layer": layer,
                "accuracy": float(
                    accuracy_score(y[~train], classifier.predict(x[~train]))
                ),
                "n_train_tokens": int(train.sum()),
                "n_test_tokens": int((~train).sum()),
            }
        )
        features[layer] = None
    return {
        "protocol": "real_checkpoint_adversarial_alpaca_binary_probe",
        "seed": seed,
        "n_prompts": prompts,
        "train_prompts": split,
        "test_prompts": prompts - split,
        "tokens_per_role_cap": tokens_per_role or None,
        "half_inputs_replaced_with_instructions": True,
        "prompt_ids": prompt_ids,
        "hidden_state_definition": "Hugging Face output_hidden_states; special/template tokens excluded",
        "source_sha256": file_hash(
            ORIGINAL / "data/train_data/alpaca_52k_shuffled_train.json"
        ),
        "layers": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=list(REGISTRY), required=True)
    p.add_argument(
        "--method", choices=["vanilla", "ise", "aside_orig", "ha_aside"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompts", type=int, default=517)
    p.add_argument(
        "--tokens-per-role",
        type=int,
        default=0,
        help="0 uses all balanced tokens; a cap is a documented reduced diagnostic",
    )
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new probing output file")
    model, tokenizer, metadata = load_checkpoint(
        args.checkpoint,
        args.method,
        args.model,
        torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        args.device,
    )
    result = probe(
        model, tokenizer, args.model, args.prompts, tokens_per_role=args.tokens_per_role
    )
    write_json(
        args.output, {"checkpoint": args.checkpoint, "architecture": metadata, **result}
    )


if __name__ == "__main__":
    main()

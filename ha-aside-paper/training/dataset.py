"""Matched SFT data with provenance-aligned token roles and explicit loss scope."""

import torch
from torch.utils.data import Dataset
from experiments.prompts import encode_task
from training.chatqa import encode_conversation


class HierarchicalAlpacaDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_length=768,
        system_instruction=None,
        loss_scope="full_sequence",
        merge_system=False,
    ):
        if loss_scope not in {"full_sequence", "assistant_only"}:
            raise ValueError("Unknown loss scope")
        self.items = []
        self.truncated_examples = 0
        self.loss_scope = loss_scope
        for item in data:
            encoded = encode_task(
                tokenizer,
                item["instruction"],
                item["input"],
                10**9,
                merge_system=merge_system,
                response=item["output"],
            )
            if encoded["input_ids"].shape[1] > max_length:
                self.truncated_examples += 1
            row = {
                k: encoded[k][0, :max_length]
                for k in [
                    "input_ids",
                    "segment_ids",
                    "binary_data_mask",
                    "attention_mask",
                ]
            }
            row["labels"] = (
                row["input_ids"].clone()
                if loss_scope == "full_sequence"
                else encoded["target_labels"][0, :max_length]
            )
            if len(row["input_ids"]) < 2 or (row["labels"][1:] != -100).sum() == 0:
                raise ValueError("Training example contains no next-token target")
            self.items.append(row)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class HierarchicalConversationDataset(Dataset):
    """Lazy encoding of whole conversations; overflow fails rather than truncates."""

    def __init__(self, data, tokenizer, max_length=2048, loss_scope="full_sequence", assistant_header_roles=False):
        if loss_scope != "full_sequence":
            raise ValueError("ChatQA v1 preserves ASIDE full-sequence loss")
        self.data, self.tokenizer, self.max_length = data, tokenizer, max_length
        self.truncated_examples = 0
        self.assistant_header_roles = assistant_header_roles

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        encoded = encode_conversation(row, self.tokenizer, self.max_length)
        if len(encoded["input_ids"][0]) != row["n_tokens"]:
            raise ValueError("Actual tokenizer length differs from prepared ChatQA row")
        if self.assistant_header_roles:
            for previous, current in zip(encoded["spans"], encoded["spans"][1:]):
                if previous["role"] == 0 and not previous["binary_data"] and current["role"] == 3:
                    # A template separator before assistant content. Source text,
                    # token IDs, DATA mask and full-sequence labels are unchanged.
                    encoded["segment_ids"][0, previous["start"]:previous["end"]] = 3
        result = {key: encoded[key][0] for key in ["input_ids", "segment_ids", "binary_data_mask", "attention_mask"]}
        result["labels"] = result["input_ids"].clone()
        return result


def collate_hierarchical_batch(batch, pad_token_id=0):
    if not batch:
        raise ValueError("Empty batch")
    length = max(len(x["input_ids"]) for x in batch)
    fills = {
        "input_ids": pad_token_id,
        "segment_ids": 0,
        "binary_data_mask": False,
        "attention_mask": False,
        "labels": -100,
    }
    result = {
        k: torch.full((len(batch), length), v, dtype=batch[0][k].dtype)
        for k, v in fills.items()
    }
    for i, item in enumerate(batch):
        for key in fills:
            result[key][i, : len(item[key])] = item[key]
    return result

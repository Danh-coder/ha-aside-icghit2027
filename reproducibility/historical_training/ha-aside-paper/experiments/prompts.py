"""Role provenance is supplied by the application, never parsed from attacker text."""

from dataclasses import dataclass
import re
import uuid
import torch
from ha_aside.core import ROLE_SYSTEM, ROLE_USER, ROLE_DATA, ROLE_ASSISTANT

PREFACE = "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\nInstruction:\n"


@dataclass(frozen=True)
class Chunk:
    text: str
    role: int
    binary_data: bool = False
    target: bool = False


def task_messages(instruction, data, direct=False, response=None):
    # Operational task instructions retain a distinct HA user role even when the
    # original ASIDE template places them inside the native system message.
    instruction_role = ROLE_SYSTEM if direct else ROLE_USER
    messages = [
        ("system", [Chunk(PREFACE, ROLE_SYSTEM), Chunk(instruction, instruction_role)]),
        (
            "user",
            [
                Chunk("Input:\n", ROLE_SYSTEM),
                Chunk(data, ROLE_USER if direct else ROLE_DATA, True),
            ],
        ),
    ]
    if response is not None:
        messages.append(
            (
                "assistant",
                [
                    Chunk("Response: ", ROLE_ASSISTANT),
                    Chunk(response, ROLE_ASSISTANT, target=True),
                ],
            )
        )
    return messages


def render_chunks(tokenizer, messages, merge_system=False, generation_prompt=True):
    """Render only opaque placeholders, then substitute provenance-tagged content.

    Payloads are never passed to delimiter detection or to the template engine.
    Tokenizing each resulting chunk independently matches ASIDE's chunked API.
    """
    messages = [(r, list(chunks)) for r, chunks in messages]
    if merge_system and messages[0][0] == "system":
        if len(messages) < 2 or messages[1][0] != "user":
            raise ValueError("System merge requires a following user message")
        messages = [
            ("user", messages[0][1] + [Chunk("\n", ROLE_SYSTEM)] + messages[1][1])
        ] + messages[2:]
    markers, chat = {}, []
    nonce = uuid.uuid4().hex
    for i, (role, chunks) in enumerate(messages):
        key = f"__HA_CONTENT_{nonce}_{i}__"
        markers[key] = chunks
        chat.append({"role": role, "content": key})
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "A model-family chat template is required; plain-text fallback is disabled"
        )
    rendered = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=generation_prompt
    )
    if any(rendered.count(key) != 1 for key in markers):
        raise ValueError(
            "Chat template discarded, duplicated, or transformed a content placeholder"
        )
    result = []
    for text in re.split("(" + "|".join(re.escape(k) for k in markers) + ")", rendered):
        result.extend(markers[text] if text in markers else [Chunk(text, ROLE_SYSTEM)])
    # The last template suffix is the assistant generation header, angle zero.
    if generation_prompt and result and not result[-1].binary_data:
        result[-1] = Chunk(result[-1].text, ROLE_ASSISTANT)
    return result


def encode_chunks(tokenizer, chunks, max_length):
    ids, roles, binary, labels, spans = [], [], [], [], []
    for chunk in chunks:
        token_ids = tokenizer(chunk.text, add_special_tokens=False)["input_ids"]
        start = len(ids)
        ids.extend(token_ids)
        roles.extend([chunk.role] * len(token_ids))
        binary.extend([chunk.binary_data] * len(token_ids))
        labels.extend(token_ids if chunk.target else [-100] * len(token_ids))
        spans.append(
            {
                "start": start,
                "end": len(ids),
                "role": chunk.role,
                "binary_data": chunk.binary_data,
                "target": chunk.target,
                "text": chunk.text,
            }
        )
    if not ids or len(ids) > max_length:
        raise ValueError(
            f"Prompt has {len(ids)} tokens, budget {max_length}; truncation is disabled"
        )
    return {
        "input_ids": torch.tensor([ids]),
        "segment_ids": torch.tensor([roles]),
        "binary_data_mask": torch.tensor([binary], dtype=torch.bool),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.bool),
        "target_labels": torch.tensor([labels]),
        "spans": spans,
        "rendered_prompt": "".join(c.text for c in chunks),
    }


def encode_task(
    tokenizer,
    instruction,
    data,
    max_length,
    direct=False,
    merge_system=False,
    response=None,
):
    return encode_chunks(
        tokenizer,
        render_chunks(
            tokenizer,
            task_messages(instruction, data, direct, response),
            merge_system,
            generation_prompt=response is None,
        ),
        max_length,
    )


def encode_safety_task(tokenizer, instruction, data, max_length, direct=False, merge_system=False):
    """Use ASIDE's exact evaluation template, without changing the SFT encoder.

    Template newlines affect both content and BPE tokenization. Keep the
    application-owned roles separate from the attacker-controlled content.
    """
    from .safety_protocol import original_template

    template = original_template()
    if len(data) < 2:
        data = "No input"  # ASIDE model_api.format_prompt's exact user fallback.
    before_instruction, after_instruction = template["system"].split("{}")
    before_data, after_data = template["user"].split("{}")
    messages = [
        ("system", [Chunk(before_instruction, ROLE_SYSTEM),
                    Chunk(instruction, ROLE_SYSTEM if direct else ROLE_USER),
                    Chunk(after_instruction, ROLE_SYSTEM)]),
        ("user", [Chunk(before_data, ROLE_SYSTEM),
                  Chunk(data, ROLE_USER if direct else ROLE_DATA, True),
                  Chunk(after_data, ROLE_SYSTEM)]),
    ]
    chunks = render_chunks(tokenizer, messages, merge_system)
    text = "".join(chunk.text for chunk in chunks)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Safety evaluation requires a fast tokenizer with character offsets")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if not ids or len(ids) > max_length:
        raise ValueError(f"Prompt has {len(ids)} tokens, budget {max_length}; truncation is disabled")
    boundaries, cursor = [], 0
    for chunk in chunks:
        boundaries.append((cursor, cursor + len(chunk.text), chunk))
        cursor += len(chunk.text)
    roles, binary = [], []
    chunk_index = 0
    for start, end in offsets:
        while chunk_index < len(boundaries) and boundaries[chunk_index][1] <= start:
            chunk_index += 1
        overlaps, index = [], chunk_index
        while index < len(boundaries) and boundaries[index][0] < end:
            a, b, chunk = boundaries[index]
            if a < end and b > start:
                overlaps.append(chunk)
            index += 1
        untrusted = [chunk for chunk in overlaps if chunk.binary_data]
        if untrusted:
            # A BPE token crossing into attacker content cannot gain trust from
            # the application-owned whitespace or delimiters beside it.
            roles.append(untrusted[0].role)
        elif any(chunk.role == ROLE_USER for chunk in overlaps):
            roles.append(ROLE_USER)
        elif any(chunk.role == ROLE_ASSISTANT for chunk in overlaps):
            roles.append(ROLE_ASSISTANT)
        else:
            roles.append(ROLE_SYSTEM)
        binary.append(bool(untrusted))
    spans = []
    for start, end, chunk in boundaries:
        tokens = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        at = sum(b <= start for a, b in offsets)
        spans.append({"start": min(tokens) if tokens else at,
                      "end": max(tokens) + 1 if tokens else at,
                      "char_start": start, "char_end": end,
                      "role": chunk.role, "binary_data": chunk.binary_data,
                      "target": False, "text": chunk.text})
    return {"input_ids": torch.tensor([ids]), "segment_ids": torch.tensor([roles]),
            "binary_data_mask": torch.tensor([binary], dtype=torch.bool),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.bool),
            "target_labels": torch.full((1, len(ids)), -100),
            "spans": spans, "rendered_prompt": text,
            "tokenization": "whole_chat_with_provenance_offsets"}

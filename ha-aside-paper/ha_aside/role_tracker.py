"""
Role Tracker for Hierarchical Multi-Turn Dialogues (HA-ASIDE)

Maintains explicit role states across multi-turn interactions (Turns 1, 2, 3, 4, ...).
Preserves supplied role provenance; this alone is not a security guarantee.
"""

from typing import List, Dict, Tuple, Optional
import torch

from .core import ROLE_SYSTEM, ROLE_USER, ROLE_DATA, ROLE_ASSISTANT, ROLE_NAMES


class SegmentChunk:
    def __init__(self, text: str, role: int):
        self.text = text
        self.role = role

    def __repr__(self):
        return f"SegmentChunk(role={ROLE_NAMES.get(self.role, self.role)}, text={self.text[:30]!r}...)"


class MultiTurnDialogueTracker:
    """
    Tracks and prepares input IDs, attention masks, and hierarchical role segment IDs
    across multi-turn conversations without semantic leakage.
    """

    def __init__(self, system_instruction: str = ""):
        self.system_instruction = system_instruction
        self.turns: List[Dict[str, any]] = []

    def add_turn(
        self,
        user_query: str,
        external_data: Optional[str] = None,
        assistant_response: Optional[str] = None,
    ):
        """
        Adds a conversation turn.
        """
        turn_data = {
            "user_query": user_query,
            "external_data": external_data,
            "assistant_response": assistant_response,
        }
        self.turns.append(turn_data)

    def format_chunks(self) -> List[SegmentChunk]:
        """
        Builds a list of semantic chunks with explicit roles.
        """
        chunks: List[SegmentChunk] = []
        if self.system_instruction:
            chunks.append(
                SegmentChunk(
                    f"[SYSTEM]\n{self.system_instruction}\n[/SYSTEM]\n", ROLE_SYSTEM
                )
            )

        for turn_idx, turn in enumerate(self.turns):
            chunks.append(
                SegmentChunk(
                    f"[TURN {turn_idx + 1} USER]\n{turn['user_query']}\n[/TURN {turn_idx + 1} USER]\n",
                    ROLE_USER,
                )
            )

            if turn["external_data"]:
                chunks.append(
                    SegmentChunk(
                        f"[DATA]\n{turn['external_data']}\n[/DATA]\n", ROLE_DATA
                    )
                )

            if turn["assistant_response"] is not None:
                chunks.append(
                    SegmentChunk(
                        f"[ASSISTANT]\n{turn['assistant_response']}\n[/ASSISTANT]\n",
                        ROLE_ASSISTANT,
                    )
                )

        return chunks

    def tokenize_and_align(
        self, tokenizer, max_length: int = 2048
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenizes each chunk individually to ensure 100% token-to-role alignment
        without any boundary ambiguity.

        Returns:
            input_ids: [1, seq_len]
            attention_mask: [1, seq_len]
            segment_ids: [1, seq_len] (0=System, 1=User, 2=Data, 3=Assistant)
        """
        chunks = self.format_chunks()
        all_ids = []
        all_segments = []

        for chunk in chunks:
            tok = tokenizer(chunk.text, add_special_tokens=False, return_tensors="pt")
            ids = tok["input_ids"][0]
            if len(ids) == 0:
                continue
            seg = torch.full_like(ids, chunk.role, dtype=torch.long)
            all_ids.append(ids)
            all_segments.append(seg)

        if not all_ids:
            return (
                torch.empty((1, 0), dtype=torch.long),
                torch.empty((1, 0), dtype=torch.bool),
                torch.empty((1, 0), dtype=torch.long),
            )

        input_ids = torch.cat(all_ids, dim=0).unsqueeze(0)
        segment_ids = torch.cat(all_segments, dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        if input_ids.shape[1] > max_length:
            raise ValueError(
                f"Dialogue has {input_ids.shape[1]} tokens, budget {max_length}; truncation is disabled"
            )

        return input_ids, attention_mask, segment_ids

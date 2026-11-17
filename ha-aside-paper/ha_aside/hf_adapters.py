"""Shared training/inference path for matched Vanilla, ISE, ASIDE and HA-ASIDE."""

import json
import math
from pathlib import Path
import torch
from torch import nn
from .core import (
    HierarchicalIsoclinicEmbedding,
    apply_isoclinic_rotation_fast,
    ROLE_DATA,
    ROLE_ASSISTANT,
)
from .rgag import RoleGuidedAttentionGating

METHODS = {"vanilla", "ise", "ise_4role", "aside_orig", "ha_aside"}


class HAASIDEModelWrapper(nn.Module):
    def __init__(
        self,
        base_model,
        rgag_penalty_data=6.0,
        rgag_penalty_user_to_sys=2.5,
        enable_rgag=True,
        mode="ha_aside",
        angles=None,
        rotation_direction="right",
    ):
        super().__init__()
        if mode not in METHODS or rotation_direction not in {"right", "left"}:
            raise ValueError("Invalid method or rotation direction")
        self.base_model = base_model
        self.mode, self.enable_rgag = mode, enable_rgag
        self.rotation_direction = rotation_direction
        self.config = base_model.config
        self.hidden_dim = self.config.hidden_size
        self.hierarchical_embedding = HierarchicalIsoclinicEmbedding(
            self.hidden_dim, angles
        )
        self.rgag = RoleGuidedAttentionGating(
            rgag_penalty_data, rgag_penalty_user_to_sys
        )
        self.ise_offset = nn.Parameter(
            torch.zeros(self.hidden_dim), requires_grad=mode == "ise"
        )
        # Created only for this method: legacy checkpoint state dictionaries and
        # RNG streams remain unchanged. All four vectors, including A, are learned.
        self.segment_embeddings = None
        if mode == "ise_4role":
            self.segment_embeddings = nn.Embedding(4, self.hidden_dim)
            nn.init.normal_(self.segment_embeddings.weight, mean=0.0, std=0.01)

    def architecture(self):
        return {
            "schema_version": 2,
            "method": self.mode,
            "angles": self.hierarchical_embedding.angles,
            "rotation_direction": self.rotation_direction,
            "enable_rgag": self.enable_rgag,
            "penalty_data": float(self.rgag.penalty_data),
            "penalty_assistant_to_user": float(self.rgag.penalty_assistant_to_user),
            "hierarchy_policy": "assistant_queries_to_user_keys",
            "generated_role": "assistant_segment_embedding" if self.mode == "ise_4role" else "assistant_identity",
            **({"segment_role_order": ["system", "user", "data", "assistant"],
                "segment_initialization_std": 0.01,
                "assistant_header_policy": "assistant_message_framing"} if self.mode == "ise_4role" else {}),
            "attention_backend": "eager",
        }

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    @property
    def device(self):
        return self.get_input_embeddings().weight.device

    def forward(
        self,
        input_ids=None,
        segment_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        binary_data_mask=None,
        mode=None,
        past_key_values=None,
        **kwargs,
    ):
        mode = mode or self.mode
        if mode not in METHODS:
            raise ValueError(f"Unknown method: {mode}")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds required")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        batch, q_len, _ = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        past_len = 0
        if past_key_values is not None:
            past_len = (
                int(past_key_values.get_seq_length())
                if hasattr(past_key_values, "get_seq_length")
                else past_key_values[0][0].shape[-2]
            )
        k_len = past_len + q_len
        if segment_ids is None or segment_ids.shape != (batch, k_len):
            raise ValueError(
                "segment_ids must cover all cached and current key positions"
            )
        segment_ids = segment_ids.to(device)
        if not ((segment_ids >= 0) & (segment_ids < 4)).all():
            raise ValueError("Role IDs must be in [0,3]")
        query_roles = segment_ids[:, -q_len:]
        if binary_data_mask is None:
            binary_data_mask = segment_ids == ROLE_DATA
        if binary_data_mask.shape != segment_ids.shape:
            raise ValueError("binary_data_mask must align with all key roles")
        data = binary_data_mask.to(device=device, dtype=torch.bool)[:, -q_len:]
        if mode == "aside_orig":
            transformed = inputs_embeds.clone()
            angle = -math.pi / 2 if self.rotation_direction == "right" else math.pi / 2
            transformed[data] = apply_isoclinic_rotation_fast(
                inputs_embeds[data], angle
            )
        elif mode == "ise":
            transformed = inputs_embeds + data.unsqueeze(-1) * self.ise_offset.to(
                device=device, dtype=dtype
            )
        elif mode == "ise_4role":
            if self.segment_embeddings is None:
                raise ValueError("ISE-4 requires its own trained segment embeddings")
            transformed = inputs_embeds + self.segment_embeddings(query_roles).to(dtype=dtype)
        elif mode == "ha_aside":
            sign = -1 if self.rotation_direction == "right" else 1
            transformed = inputs_embeds.clone()
            for role, angle in self.hierarchical_embedding.angles.items():
                selected = query_roles == role
                if angle and selected.any():
                    transformed[selected] = apply_isoclinic_rotation_fast(
                        inputs_embeds[selected], sign * angle
                    )
        else:
            transformed = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones((batch, k_len), dtype=torch.bool, device=device)
        if attention_mask.ndim != 2 or attention_mask.shape != (batch, k_len):
            raise ValueError("Provide a 2D padding mask for cached and current keys")
        padding = attention_mask.to(device=device, dtype=torch.bool)
        keys = torch.arange(k_len, device=device)
        queries = torch.arange(past_len, k_len, device=device)
        allowed = (keys[None, :] <= queries[:, None])[None, None] & padding[
            :, None, None, :
        ]
        mask = torch.zeros((batch, 1, q_len, k_len), device=device, dtype=dtype)
        if mode == "ha_aside" and self.enable_rgag:
            if getattr(self.config, "_attn_implementation", "eager") != "eager":
                raise ValueError("Reference RGAG requires attn_implementation='eager'")
            mask = mask - self.rgag(
                query_roles, segment_ids, dtype=dtype, device=device
            )
        mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
        kwargs.setdefault(
            "position_ids", (padding.long().cumsum(-1) - 1).clamp_min(0)[:, -q_len:]
        )
        return self.base_model(
            inputs_embeds=transformed,
            attention_mask=mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        segment_ids,
        attention_mask=None,
        binary_data_mask=None,
        max_new_tokens=1024,
        mode=None,
        do_sample=False,
        temperature=0.7,
        top_k=0,
        generator=None,
        use_cache=True,
        **kwargs,
    ):
        if max_new_tokens <= 0 or (do_sample and temperature <= 0) or top_k < 0:
            raise ValueError("Invalid generation settings")
        if input_ids.shape[0] != 1:
            raise ValueError("Reference generation uses one unpadded example at a time")
        ids, roles = input_ids.clone(), segment_ids.clone()
        mask = (
            torch.ones_like(ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool().clone()
        )
        if not mask.all():
            raise ValueError("Remove padding before reference generation")
        binary = (
            (roles == ROLE_DATA)
            if binary_data_mask is None
            else binary_data_mask.bool().clone()
        )
        eos = getattr(self.config, "eos_token_id", None)
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        cache = None
        for _ in range(max_new_tokens):
            current = ids[:, -1:] if cache is not None else ids
            out = self(
                input_ids=current,
                segment_ids=roles,
                binary_data_mask=binary,
                attention_mask=mask,
                mode=mode,
                past_key_values=cache,
                use_cache=use_cache,
                **kwargs,
            )
            logits = out.logits[:, -1].float()
            if do_sample:
                logits = logits / temperature
                if top_k:
                    cutoff = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, -torch.inf)
            token = (
                torch.multinomial(
                    logits.softmax(-1), 1, generator=generator
                )
                if do_sample
                else logits.argmax(-1, keepdim=True)
            )
            ids = torch.cat((ids, token), 1)
            roles = torch.cat((roles, torch.full_like(token, ROLE_ASSISTANT)), 1)
            binary = torch.cat((binary, torch.zeros_like(token, dtype=torch.bool)), 1)
            mask = torch.cat((mask, torch.ones_like(token, dtype=torch.bool)), 1)
            cache = out.past_key_values if use_cache else None
            if token.item() in eos:
                break
        return ids, roles

    def save_experiment(self, path, tokenizer=None, provenance=None, state_dict=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        base_state = (
            None
            if state_dict is None
            else {
                k.removeprefix("base_model."): v
                for k, v in state_dict.items()
                if k.startswith("base_model.")
            }
        )
        self.base_model.save_pretrained(path, state_dict=base_state)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        (path / "architecture.json").write_text(
            json.dumps(
                {**self.architecture(), "provenance": provenance or {}}, indent=2
            ),
            encoding="utf-8",
        )
        if self.mode == "ise":
            torch.save(
                self.ise_offset.detach().cpu()
                if state_dict is None
                else state_dict["ise_offset"].cpu(),
                path / "ise_offset.pt",
            )
        if self.mode == "ise_4role":
            torch.save(
                self.segment_embeddings.weight.detach().cpu() if state_dict is None
                else state_dict["segment_embeddings.weight"].detach().cpu(),
                path / "segment_embeddings.pt",
            )


def wrap_model_with_ha_aside(model, **kwargs):
    return HAASIDEModelWrapper(model, **kwargs)

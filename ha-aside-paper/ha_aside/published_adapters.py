"""Published ASIDE binary transforms, preserving the author's saved tensors."""

import torch

from .core import ROLE_DATA
from .hf_adapters import HAASIDEModelWrapper


class PublishedBinaryWrapper(HAASIDEModelWrapper):
    def __init__(self, base_model, method, *, rotation=None, segments=None):
        if method not in {"vanilla", "aside_orig", "ise"}:
            raise ValueError("Unsupported published binary method")
        super().__init__(base_model, mode="vanilla", enable_rgag=False)
        self.mode = method
        if method == "aside_orig" and (rotation is None or rotation.shape != (self.hidden_dim, self.hidden_dim)):
            raise ValueError("Missing or invalid saved ASIDE rotation_matrix")
        if method == "ise" and (segments is None or segments.shape != (2, self.hidden_dim)):
            raise ValueError("Missing or invalid saved ISE segment_embedding.weight")
        for value in [rotation, segments]:
            if value is not None and not torch.isfinite(value).all():
                raise ValueError("Nonfinite published transform")
        self.register_buffer("published_rotation", rotation)
        self.register_buffer("published_segments", segments)

    def architecture(self):
        return {"source_type": "author_published", "method": self.mode, "enable_rgag": False,
                "encoding": "DATA=1; SYSTEM/USER/ASSISTANT=0",
                "generated_role": "learned_instruction_segment_0" if self.mode == "ise" else "assistant_identity",
                "rotation": "saved_checkpoint_matrix_right_multiply" if self.mode == "aside_orig" else None}

    def save_experiment(self, *args, **kwargs):
        raise ValueError("Published evaluation adapters cannot be relabeled as locally trained checkpoints")

    def forward(self, input_ids=None, segment_ids=None, attention_mask=None, inputs_embeds=None,
                binary_data_mask=None, mode=None, past_key_values=None, **kwargs):
        if mode not in {None, self.mode}:
            raise ValueError("Cannot change a published checkpoint's method at inference")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if segment_ids is None:
            raise ValueError("Application-owned provenance is required")
        n = inputs_embeds.shape[1]
        data = (segment_ids == ROLE_DATA) if binary_data_mask is None else binary_data_mask.bool()
        data = data[:, -n:].to(inputs_embeds.device)
        if self.mode == "aside_orig":
            transformed = inputs_embeds.clone()
            transformed[data] = inputs_embeds[data] @ self.published_rotation.to(inputs_embeds)
        elif self.mode == "ise":
            # Both vectors are learned in the original model. Instruction,
            # SYSTEM, USER and generated/history ASSISTANT all use segment 0.
            transformed = inputs_embeds + self.published_segments.to(inputs_embeds)[data.long()]
        else:
            transformed = inputs_embeds
        return super().forward(input_ids=None, inputs_embeds=transformed, segment_ids=segment_ids,
                               binary_data_mask=binary_data_mask, attention_mask=attention_mask,
                               mode="vanilla", past_key_values=past_key_values, **kwargs)

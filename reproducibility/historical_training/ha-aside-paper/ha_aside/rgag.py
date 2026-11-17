"""Finite pre-softmax penalties, with no unconditional leakage guarantee."""

import torch
from torch import nn
from .core import ROLE_USER, ROLE_DATA, ROLE_ASSISTANT


class RoleGuidedAttentionGating(nn.Module):
    def __init__(
        self,
        penalty_data=6.0,
        penalty_user_to_sys=2.5,
        learnable=False,
        penalty_assistant_to_user=None,
    ):
        super().__init__()
        # Historical keyword remains accepted; corrected direction is explicit in metadata.
        value = (
            penalty_user_to_sys
            if penalty_assistant_to_user is None
            else penalty_assistant_to_user
        )
        if penalty_data < 0 or value < 0:
            raise ValueError("Attention penalties must be nonnegative")
        for name, v in [
            ("penalty_data", penalty_data),
            ("penalty_assistant_to_user", value),
        ]:
            tensor = torch.tensor(float(v))
            if learnable:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

    def forward(
        self, query_segment_ids, key_segment_ids, dtype=torch.float32, device=None
    ):
        device = device or query_segment_ids.device
        q = query_segment_ids.to(device).unsqueeze(2)
        k = key_segment_ids.to(device).unsqueeze(1)
        data = (q != ROLE_DATA) & (k == ROLE_DATA)
        subordinate = (q == ROLE_ASSISTANT) & (k == ROLE_USER)
        bias = data.to(dtype) * self.penalty_data.to(
            device=device, dtype=dtype
        ).clamp_min(0)
        bias = bias + subordinate.to(dtype) * self.penalty_assistant_to_user.to(
            device=device, dtype=dtype
        ).clamp_min(0)
        return bias.unsqueeze(1)

    def apply_to_scores(self, scores, segment_ids):
        return scores - self(
            segment_ids, segment_ids, dtype=scores.dtype, device=scores.device
        )

"""Fixed isoclinic rotations; coordinate frames are not disjoint subspaces."""

import math
import torch
from torch import nn

ROLE_SYSTEM, ROLE_USER, ROLE_DATA, ROLE_ASSISTANT = range(4)
ROLE_NAMES = dict(enumerate(["SYSTEM", "USER", "DATA", "ASSISTANT"]))
# Fixed v2 hypothesis, not a claim of an optimal angle or semantic fidelity.
DEFAULT_ANGLES = {0: 0.0, 1: math.pi / 8, 2: math.pi / 2, 3: 0.0}


def build_isoclinic_matrix(dim, theta, device=None, dtype=torch.float32):
    if dim % 2:
        raise ValueError("Rotation requires even hidden size")
    block = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        device=device,
        dtype=dtype,
    )
    return torch.block_diag(*([block] * (dim // 2)))


def apply_isoclinic_rotation_fast(x, theta):
    """Column convention R*x, implemented on adjacent coordinate pairs."""
    if x.shape[-1] % 2:
        raise ValueError("Rotation requires even hidden size")
    if abs(theta) < 1e-12:
        return x
    a, b = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).unbind(-1)
    return torch.stack(
        (
            math.cos(theta) * a - math.sin(theta) * b,
            math.sin(theta) * a + math.cos(theta) * b,
        ),
        -1,
    ).reshape_as(x)


class HierarchicalIsoclinicEmbedding(nn.Module):
    def __init__(self, hidden_dim, angles=None):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("Rotation requires even hidden size")
        self.hidden_dim = hidden_dim
        self.angles = dict(DEFAULT_ANGLES if angles is None else angles)
        if set(self.angles) != set(ROLE_NAMES):
            raise ValueError("An angle is required for every role")

    def forward(self, embeddings, segment_ids):
        if (
            embeddings.shape[:2] != segment_ids.shape
            or not ((segment_ids >= 0) & (segment_ids < 4)).all()
        ):
            raise ValueError("Invalid role alignment")
        out = embeddings.clone()
        for role, theta in self.angles.items():
            mask = segment_ids == role
            if theta and mask.any():
                out[mask] = apply_isoclinic_rotation_fast(embeddings[mask], theta)
        return out

    def verify_invariants(self, sample_tensor):
        x = sample_tensor.float()
        y = apply_isoclinic_rotation_fast(x, self.angles[ROLE_DATA])
        return {
            "max_norm_error": (x.norm(dim=-1) - y.norm(dim=-1)).abs().max().item(),
            "same_vector_cosine": torch.nn.functional.cosine_similarity(x, y, dim=-1)
            .mean()
            .item(),
            "scope": "same-vector identity; no arbitrary cross-role orthogonality guarantee",
        }

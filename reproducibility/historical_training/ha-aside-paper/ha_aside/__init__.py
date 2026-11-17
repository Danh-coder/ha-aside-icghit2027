"""
HA-ASIDE: Hierarchical Architectural Separation of Instructions and Data
with Role-Guided Attention Gating (RGAG)
"""

from .core import (
    HierarchicalIsoclinicEmbedding,
    build_isoclinic_matrix,
    apply_isoclinic_rotation_fast,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_DATA,
    ROLE_ASSISTANT,
    ROLE_NAMES,
    DEFAULT_ANGLES,
)
from .rgag import RoleGuidedAttentionGating
from .role_tracker import MultiTurnDialogueTracker
from .hf_adapters import HAASIDEModelWrapper, wrap_model_with_ha_aside

__version__ = "2.0.0"
__all__ = [
    "HierarchicalIsoclinicEmbedding",
    "RoleGuidedAttentionGating",
    "MultiTurnDialogueTracker",
    "HAASIDEModelWrapper",
    "wrap_model_with_ha_aside",
    "build_isoclinic_matrix",
    "apply_isoclinic_rotation_fast",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "ROLE_DATA",
    "ROLE_ASSISTANT",
    "ROLE_NAMES",
    "DEFAULT_ANGLES",
]

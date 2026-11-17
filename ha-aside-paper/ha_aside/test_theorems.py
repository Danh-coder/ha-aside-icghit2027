"""Math regressions; passing examples are not universal security proofs."""

import math
import unittest
import torch
from ha_aside.core import apply_isoclinic_rotation_fast
from ha_aside.rgag import RoleGuidedAttentionGating


class RotationTests(unittest.TestCase):
    def test_isometry(self):
        torch.manual_seed(42)
        x, y = (
            torch.randn(10, 64, dtype=torch.float64),
            torch.randn(10, 64, dtype=torch.float64),
        )
        for angle in [math.pi / 8, math.pi / 4, math.pi / 2]:
            a, b = (
                apply_isoclinic_rotation_fast(x, angle),
                apply_isoclinic_rotation_fast(y, angle),
            )
            self.assertTrue(torch.allclose(x.norm(dim=-1), a.norm(dim=-1), atol=1e-12))
            self.assertTrue(
                torch.allclose((x * y).sum(-1), (a * b).sum(-1), atol=1e-12)
            )

    def test_arbitrary_vectors_are_not_orthogonal(self):
        x, y = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
        self.assertEqual(
            float((x * apply_isoclinic_rotation_fast(y, math.pi / 2)).sum()), -1.0
        )

    def test_large_logit_overcomes_finite_penalty(self):
        gate = RoleGuidedAttentionGating(6.0)
        weights = (
            torch.tensor([[[[0.0, 20.0]]]])
            - gate(torch.tensor([[3]]), torch.tensor([[0, 2]]))
        ).softmax(-1)
        self.assertGreater(float(weights[..., 1]), 0.99)


if __name__ == "__main__":
    unittest.main()

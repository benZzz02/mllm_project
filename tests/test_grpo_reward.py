from __future__ import annotations

import json
import unittest
from collections import Counter

from dgm4_pipeline.grpo_reward import (
    CASE_IMAGE_TEXT_MIXED,
    CASE_NORMAL,
    case_name,
    clipped_case_weights,
    make_badcase_aware_reward,
    score_response,
)
from dgm4_pipeline.schema import canonical_answer


class GrpoRewardTest(unittest.TestCase):
    def test_case_weights_boost_tail_without_exploding(self) -> None:
        weights = clipped_case_weights(Counter({CASE_NORMAL: 100, "face_swap": 25, CASE_IMAGE_TEXT_MIXED: 4}))
        self.assertEqual(weights[CASE_NORMAL], 0.6)
        self.assertGreater(weights[CASE_IMAGE_TEXT_MIXED], weights["face_swap"])
        self.assertLessEqual(weights[CASE_IMAGE_TEXT_MIXED], 2.0)

    def test_scores_mixed_case_with_partial_type_and_evidence(self) -> None:
        target = {
            "verdict": "manipulated",
            "types": ["face_swap", "text_attribute"],
            "image_bbox": [100, 100, 600, 600],
            "text_positions": [2, 4],
        }
        response = json.dumps(
            {
                "verdict": "manipulated",
                "types": ["face_swap"],
                "image_bbox": [100, 100, 600, 600],
                "text_positions": [],
            }
        )
        breakdown = score_response(target, response, {CASE_IMAGE_TEXT_MIXED: 1.6})
        self.assertEqual(case_name(target), CASE_IMAGE_TEXT_MIXED)
        self.assertTrue(breakdown.valid)
        self.assertEqual(breakdown.case_weight, 1.6)
        self.assertAlmostEqual(breakdown.verdict_reward, 1.0)
        self.assertAlmostEqual(breakdown.type_reward, 2 / 3)
        self.assertAlmostEqual(breakdown.evidence_reward, 0.5)
        self.assertGreater(breakdown.reward, 1.0)

    def test_penalizes_invalid_and_hacky_outputs(self) -> None:
        normal_target = {"verdict": "pristine", "types": [], "image_bbox": None, "text_positions": []}
        invalid = score_response(normal_target, "not json")
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.reward, -0.5)

        response = json.dumps(
            {
                "verdict": "manipulated",
                "types": ["face_swap", "face_attribute", "text_swap", "text_attribute"],
                "image_bbox": [0, 0, 1000, 1000],
                "text_positions": list(range(40)),
            }
        )
        hacky = score_response(normal_target, response)
        self.assertTrue(hacky.valid)
        self.assertEqual(hacky.penalty, 0.5)
        self.assertLess(hacky.reward, 0.0)

    def test_reward_function_matches_trl_completion_shape(self) -> None:
        target = {"verdict": "pristine", "types": [], "image_bbox": None, "text_positions": []}
        reward = make_badcase_aware_reward()
        scores = reward(
            completions=[[{"role": "assistant", "content": canonical_answer(target)}]],
            target=[target],
        )
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 1.0)

    def test_reward_function_repeats_targets_for_grouped_generations(self) -> None:
        target = {"verdict": "pristine", "types": [], "image_bbox": None, "text_positions": []}
        reward = make_badcase_aware_reward()
        scores = reward(
            completions=[
                [{"role": "assistant", "content": canonical_answer(target)}],
                [{"role": "assistant", "content": canonical_answer(target)}],
            ],
            target=[target],
        )
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()

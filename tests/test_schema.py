from __future__ import annotations

import json
import unittest

from dgm4_pipeline.schema import (
    bbox_iou,
    build_target,
    canonical_answer,
    compare_prediction,
    normalize_caption,
    parse_response,
)


class SchemaTest(unittest.TestCase):
    def test_caption_matches_dgm4_normalization_contract(self) -> None:
        self.assertEqual(normalize_caption("A-B/<person>,  says: 'Hi!'"), "a b person says hi")

    def test_build_and_parse_multimodal_target(self) -> None:
        annotation = {
            "fake_cls": "face_swap&text_attribute",
            "fake_image_box": [10, 20, 60, 80],
            "fake_text_pos": [4, 2, 2],
        }
        target = build_target(annotation, width=100, height=100)
        self.assertEqual(target["image_bbox"], [100, 200, 600, 800])
        self.assertEqual(target["text_positions"], [2, 4])
        parsed = parse_response(canonical_answer(target))
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.verdict, "manipulated")
        self.assertEqual(list(parsed.types), ["face_swap", "text_attribute"])
        self.assertEqual(compare_prediction(target, parsed), [])

    def test_schema_rejects_inconsistent_evidence(self) -> None:
        response = json.dumps(
            {
                "verdict": "pristine",
                "types": [],
                "image_bbox": [1, 2, 3, 4],
                "text_positions": [],
            }
        )
        parsed = parse_response(response)
        self.assertFalse(parsed.valid)
        self.assertIn("invalid_schema", parsed.error or "")

    def test_empty_boxes_follow_official_test_rule(self) -> None:
        self.assertEqual(bbox_iou(None, None), 1.0)
        self.assertEqual(bbox_iou(None, [0, 0, 1, 1]), 0.0)
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 5, 10, 10]), 0.25)


if __name__ == "__main__":
    unittest.main()


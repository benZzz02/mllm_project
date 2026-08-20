from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dgm4_pipeline.io_utils import load_records
from dgm4_pipeline.schema import TYPE_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def annotation(index: int, image_name: str) -> dict:
    labels = (
        "orig",
        "face_swap",
        "face_attribute",
        "text_swap",
        "text_attribute",
        "face_swap&text_swap",
    )
    label = labels[index % len(labels)]
    has_image = "face_" in label
    has_text = "text_" in label
    return {
        "id": f"news-{index}",
        "image": image_name,
        "text": f"Person number {index} appears in the city today.",
        "fake_cls": label,
        "fake_image_box": [8, 4, 40, 28] if has_image else None,
        "fake_text_pos": [2] if has_text else [],
    }


class PipelineIntegrationTest(unittest.TestCase):
    def run_script(self, *arguments: str) -> None:
        subprocess.run(
            [sys.executable, *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_convert_mine_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_dir = root / "metadata"
            image_root = root / "images"
            output_dir = root / "generated"
            metadata_dir.mkdir()
            image_root.mkdir()

            split_sizes = {"train": 60, "val": 12, "test": 12}
            for split, size in split_sizes.items():
                rows = []
                for index in range(size):
                    unique_index = index if split == "train" else 1000 + index + (100 if split == "test" else 0)
                    image_name = f"{split}_{index}.jpg"
                    Image.new("RGB", (64, 32), color=(index % 255, 40, 80)).save(image_root / image_name)
                    rows.append(annotation(unique_index, image_name))
                (metadata_dir / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")

            self.run_script(
                "scripts/convert_dgm4_to_sharegpt.py",
                "--metadata-dir",
                str(metadata_dir),
                "--image-root",
                str(image_root),
                "--output-dir",
                str(output_dir),
                "--preference-pool-ratio",
                "0.2",
                "--seed",
                "42",
            )

            sft = load_records(output_dir / "dgm4_sft_train.jsonl")
            pool = load_records(output_dir / "dgm4_preference_pool.jsonl")
            validation = load_records(output_dir / "dgm4_val.jsonl")
            self.assertTrue(sft)
            self.assertTrue(pool)
            self.assertFalse(
                {row["meta"]["news_id"] for row in sft}.intersection(
                    {row["meta"]["news_id"] for row in pool}
                )
            )
            self.assertTrue(all(row["meta"]["dgm4_split"] == "train" for row in pool))

            pool_predictions = root / "pool_predictions.jsonl"
            pool_predictions.write_text(
                "".join(json.dumps({"id": row["id"], "response": "not json"}) + "\n" for row in pool),
                encoding="utf-8",
            )
            preference_output = output_dir / "dgm4_badcase_preference.jsonl"
            self.run_script(
                "scripts/build_preference_pairs.py",
                "--pool",
                str(output_dir / "dgm4_preference_pool.jsonl"),
                "--predictions",
                str(pool_predictions),
                "--output",
                str(preference_output),
                "--max-pairs",
                "5",
            )
            pairs = load_records(preference_output)
            manipulated_pool_count = sum(
                row["meta"]["target"]["verdict"] == "manipulated" for row in pool
            )
            self.assertEqual(len(pairs), min(5, manipulated_pool_count))
            self.assertTrue(all(row["chosen"]["from"] == "gpt" for row in pairs))
            report = json.loads(preference_output.with_suffix(".report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["badcase_count_before_cap"], manipulated_pool_count)

            perfect_predictions = root / "perfect_val.jsonl"
            prediction_rows = []
            for row in validation:
                target = row["meta"]["target"]
                expected_types = set(target["types"])
                prediction_rows.append(
                    {
                        "id": row["id"],
                        "response": row["conversations"][1]["value"],
                        "manipulated_score": 0.95 if target["verdict"] == "manipulated" else 0.05,
                        "type_scores": {
                            type_name: 0.95 if type_name in expected_types else 0.05 for type_name in TYPE_ORDER
                        },
                    }
                )
            perfect_predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in prediction_rows), encoding="utf-8"
            )
            metrics_path = root / "metrics.json"
            badcases_path = root / "badcases.jsonl"
            self.run_script(
                "scripts/evaluate_dgm4_predictions.py",
                "--ground-truth",
                str(output_dir / "dgm4_val.jsonl"),
                "--predictions",
                str(perfect_predictions),
                "--output",
                str(metrics_path),
                "--badcases-output",
                str(badcases_path),
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["json_valid_rate"], 1.0)
            self.assertEqual(metrics["binary_classification"]["accuracy"], 1.0)
            self.assertEqual(metrics["binary_classification"]["auc"], 1.0)
            self.assertEqual(metrics["binary_classification"]["eer"], 0.0)
            self.assertEqual(metrics["multilabel_classification"]["map"], 1.0)
            self.assertEqual(metrics["image_grounding"]["mean_iou"], 1.0)
            self.assertEqual(metrics["text_grounding"]["word_level"]["f1"], 1.0)
            self.assertEqual(metrics["badcase_analysis"]["badcase_count"], 0)

            invalid_generation_rows = json.loads(json.dumps(prediction_rows))
            invalid_generation_rows[0]["response"] = "not json"
            invalid_generation_path = root / "invalid_generation.jsonl"
            invalid_generation_path.write_text(
                "".join(json.dumps(row) + "\n" for row in invalid_generation_rows), encoding="utf-8"
            )
            invalid_metrics_path = root / "invalid_metrics.json"
            self.run_script(
                "scripts/evaluate_dgm4_predictions.py",
                "--ground-truth",
                str(output_dir / "dgm4_val.jsonl"),
                "--predictions",
                str(invalid_generation_path),
                "--output",
                str(invalid_metrics_path),
            )
            invalid_metrics = json.loads(invalid_metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(invalid_metrics["binary_classification"]["accuracy"], 1.0)
            self.assertLess(invalid_metrics["json_valid_rate"], 1.0)
            self.assertLess(invalid_metrics["structured_output"]["verdict_accuracy"], 1.0)

            before_metrics = json.loads(json.dumps(metrics))
            before_metrics["binary_classification"]["auc"] = 0.5
            before_metrics["binary_classification"]["accuracy"] = 0.5
            before_metrics["binary_classification"]["eer"] = 0.5
            before_metrics["badcase_analysis"]["badcase_rate"] = 0.5
            before_path = root / "before_metrics.json"
            before_path.write_text(json.dumps(before_metrics), encoding="utf-8")
            comparison_path = root / "comparison.json"
            self.run_script(
                "scripts/compare_runs.py",
                "--before",
                str(before_path),
                "--after",
                str(metrics_path),
                "--output",
                str(comparison_path),
            )
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            self.assertEqual(comparison["metrics"]["binary_auc"]["absolute_delta"], 0.5)
            self.assertEqual(comparison["metrics"]["badcase_rate"]["relative_error_reduction"], 1.0)


if __name__ == "__main__":
    unittest.main()

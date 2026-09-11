from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from activevision_workbench import (
    ContractValidationError,
    EvaluationProtocol,
    InferenceRequest,
    PredictionRecord,
    read_predictions_jsonl,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "golden" / "v1"


def _json(name: str) -> dict[str, object]:
    value = json.loads((GOLDEN_ROOT / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{name} must contain a JSON object")
    return value


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(text for child in value.values() for text in _strings(child))
    if isinstance(value, list):
        return tuple(text for child in value for text in _strings(child))
    return ()


class GoldenFixtureTest(unittest.TestCase):
    def test_request_is_canonical_and_deterministic(self) -> None:
        data = _json("request.json")
        request = InferenceRequest.from_dict(data)
        self.assertEqual(request.to_dict(), data)
        self.assertEqual(request.request_id, request.deterministic_id())

    def test_prediction_variants_round_trip(self) -> None:
        names = (
            "deterministic_prediction.json",
            "temporal_prediction.json",
            "observer_conditioned_prediction.json",
            "instruction_text_prediction.json",
        )
        records = {
            name: PredictionRecord.from_dict(_json(name)) for name in names
        }
        for name, record in records.items():
            self.assertEqual(record.to_dict(), _json(name))

        temporal = records["temporal_prediction.json"]
        self.assertTrue(
            all(
                fixation.timestamp_s is not None
                and fixation.duration_s is not None
                and fixation.native_timing["unit"] == "milliseconds"
                for fixation in temporal.fixations
            )
        )
        observer = records["observer_conditioned_prediction.json"]
        self.assertEqual(observer.observer_id, "subject-01")
        instruction = records["instruction_text_prediction.json"]
        self.assertEqual(instruction.task_text, "Find the yellow mug")
        self.assertIsNotNone(instruction.explanation_text)

    def test_stochastic_samples_remain_separate(self) -> None:
        records = read_predictions_jsonl(
            GOLDEN_ROOT / "stochastic_predictions.jsonl"
        )
        self.assertEqual(len(records), 2)
        self.assertEqual([record.sample_index for record in records], [0, 1])
        self.assertEqual([record.seed for record in records], [31, 32])
        self.assertEqual(len({record.sample_id for record in records}), 2)
        self.assertNotEqual(records[0].fixations, records[1].fixations)

    def test_run_and_evaluation_artifact_shapes(self) -> None:
        manifest = _json("run_manifest.json")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertTrue(
            {
                "activevision_git",
                "configuration_hash",
                "dataset",
                "environment",
                "model",
                "resolved_configuration",
                "seeds",
                "slurm",
            }.issubset(manifest)
        )
        summary = _json("evaluation_summary.json")
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["status"], "completed")
        self.assertIn("applicable", summary["metric_status_counts"])
        self.assertIn("not_applicable", summary["metric_status_counts"])
        self.assertEqual(summary["groups"][0]["metric_version"], "1.0.0")

        for value in (manifest, summary):
            for text in _strings(value):
                if text.startswith(("http://", "https://")):
                    continue
                self.assertNotIn("/home/", text)
                self.assertNotIn("lchen", text.lower())
                self.assertFalse(text.startswith("/"), text)

    def test_unknown_contract_versions_are_rejected(self) -> None:
        request = _json("request.json")
        request["schema_version"] = 2
        with self.assertRaises(ContractValidationError):
            InferenceRequest.from_dict(request)

        prediction = _json("deterministic_prediction.json")
        prediction["schema_version"] = 2
        with self.assertRaises(ContractValidationError):
            PredictionRecord.from_dict(prediction)


class ReleaseAuditTest(unittest.TestCase):
    def test_specialized_evaluation_examples_are_strict(self) -> None:
        expected = {
            "stochastic_dummy.json": {
                "endpoint_pairwise_distance_normalized",
                "endpoint_dispersion_normalized",
                "fixation_density_kde",
                "auc_judd_fixation_density",
                "nss_fixation_density",
                "fixation_count",
            },
            "observer_dummy.json": {
                "fixation_count",
                "mean_fixation_duration_s",
                "scanpath_length_normalized",
            },
            "temporal_dummy.json": {
                "duration_mae_s",
                "mean_fixation_duration_s",
                "timestamp_span_s",
                "total_fixation_duration_s",
            },
        }
        for name, metric_names in expected.items():
            protocol = EvaluationProtocol.from_file(
                REPOSITORY_ROOT / "examples" / "evaluation" / name
            )
            self.assertEqual(
                {metric.name for metric in protocol.metrics}, metric_names
            )

    def test_release_audit_script(self) -> None:
        result = subprocess.run(
            (
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "check_release.py"),
                "--repository",
                str(REPOSITORY_ROOT),
            ),
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

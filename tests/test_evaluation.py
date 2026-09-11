from __future__ import annotations

import csv
import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from activevision_workbench.adapters.dummy import create_dummy_registry
from activevision_workbench.contracts import (
    ArtifactReference,
    Capability,
    FixationEvent,
    ModelIdentity,
    PredictionRecord,
)
from activevision_workbench.datasets import (
    DatasetItem,
    GroundTruthScanpath,
    TargetBoundingBox,
)
from activevision_workbench.cli import main as cli_main
from activevision_workbench.evaluation import (
    BestOfNConfig,
    ConfidenceIntervalConfig,
    CoordinateConversionConfig,
    EvaluationEngine,
    EvaluationProtocol,
    ExistingOutputPolicy,
    GroundTruthConfig,
    MetricDefinition,
    MetricContext,
    MetricDirection,
    MetricKind,
    MetricRegistry,
    MetricRequest,
    MetricScope,
    ObserverAggregationConfig,
    SampleAggregationConfig,
    TemporalAlignmentConfig,
    create_default_metric_registry,
)
from activevision_workbench.evaluation.metrics import MetricValue
from activevision_workbench.run_config import RunConfig, RunItem, RunPolicy
from activevision_workbench.run_engine import RunEngine


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "datasets" / "tiny_gaze"


def _run_item(
    *,
    item_id: str,
    observer_id: str,
    task_text: str,
    image_name: str = "wide.png",
    width: int = 100,
    height: int = 50,
) -> RunItem:
    return RunItem(
        item_id=item_id,
        image_ref=str(FIXTURE_ROOT / "images" / image_name),
        image_width=width,
        image_height=height,
        task_text=task_text,
        target_description="circle",
        observer_id=observer_id,
        observer_metadata={"fixture": True},
    )


def _completed_run(
    root: Path,
    *,
    items: tuple[RunItem, ...] | None = None,
    split: str = "train",
    num_samples: int = 2,
) -> Path:
    config = RunConfig.from_file(
        REPOSITORY_ROOT / "examples" / "dummy_tiny_dataset_run.yaml"
    )
    if items is None:
        items = (
            _run_item(
                item_id="scene-a",
                observer_id="observer-a",
                task_text="find the yellow circle",
            ),
        )
    config = replace(
        config,
        run_id="evaluation-test-run",
        output_root=str(root),
        run_policy=RunPolicy.CREATE,
        items=items,
        item_subset=None,
        dataset_split=split,
        num_samples=num_samples,
    )
    result = RunEngine(create_dummy_registry(), console=None).run(config)
    return result.run_directory


def _protocol(
    run_directory: Path,
    metrics: tuple[MetricRequest, ...],
    *,
    split: str = "train",
    temporal_policy: str = "strict_index",
    best_of_n: BestOfNConfig | None = None,
) -> EvaluationProtocol:
    return EvaluationProtocol(
        schema_version=1,
        run_directory=str(run_directory),
        dataset=GroundTruthConfig(
            adapter="json",
            dataset_id="tiny_gaze",
            version="fixture-v1",
            root=str(FIXTURE_ROOT),
            split=split,
            task_type="auto",
            options={"annotation_name": "annotations.json"},
        ),
        metrics=metrics,
        sample_aggregation=SampleAggregationConfig(
            policy="mean_std",
            standard_deviation="population",
            confidence_interval=ConfidenceIntervalConfig(
                method="normal", level=0.95
            ),
            best_of_n=best_of_n or BestOfNConfig(),
        ),
        observer_aggregation=ObserverAggregationConfig(
            policy="macro",
            include_micro=False,
            unknown_observer_policy="separate",
        ),
        temporal_alignment=TemporalAlignmentConfig(
            policy=temporal_policy,
            unit="seconds",
        ),
        coordinate_conversion=CoordinateConversionConfig(
            policy="canonical_pixel_center",
            out_of_bounds="invalid",
        ),
        output_directory=str(run_directory / "metrics"),
        existing_output=ExistingOutputPolicy.CREATE,
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_all_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_all_keys(child))
        return result
    return set()


class EvaluationProtocolTest(unittest.TestCase):
    def test_strict_round_trip_and_dry_run_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _completed_run(Path(temporary))
            protocol = _protocol(
                run,
                (
                    MetricRequest("fixation_count", {}),
                    MetricRequest("nss", {}),
                ),
            )
            restored = EvaluationProtocol.from_dict(protocol.to_dict())
            self.assertEqual(restored.protocol_hash, protocol.protocol_hash)
            plan = EvaluationEngine().dry_run(protocol)
            statuses = {metric["name"]: metric["status"] for metric in plan.metrics}
            self.assertEqual(statuses["fixation_count"], "applicable")
            self.assertEqual(statuses["nss"], "not_applicable")
            self.assertEqual(tuple((run / "metrics").iterdir()), ())

    def test_evaluate_cli_dry_run_reports_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _completed_run(root)
            protocol = _protocol(
                run,
                (
                    MetricRequest("fixation_count", {}),
                    MetricRequest("nss", {}),
                ),
            )
            protocol_path = root / "evaluation.json"
            protocol_path.write_text(
                json.dumps(protocol.to_dict()), encoding="utf-8"
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    (
                        "evaluate",
                        "--run",
                        str(run),
                        "--protocol",
                        str(protocol_path),
                        "--dry-run",
                    )
                )
            self.assertEqual(code, 0)
            self.assertIn('"applicable"', output.getvalue())
            self.assertIn('"not_applicable"', output.getvalue())
            self.assertIn("Evaluation directory:", output.getvalue())
            self.assertEqual(tuple((run / "metrics").iterdir()), ())

    def test_missing_metric_dependency_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _completed_run(Path(temporary))
            registry = MetricRegistry()
            registry.register(
                MetricDefinition(
                    name="dependency_probe",
                    implementation_version="1.0.0",
                    kind=MetricKind.DIAGNOSTIC,
                    scope=MetricScope.SAMPLE,
                    required_prediction_fields=("scanpath_record",),
                    required_ground_truth_fields=(),
                    supported_task_types=frozenset({"any"}),
                    compatible_capabilities=frozenset(
                        {Capability.PRODUCES_SCANPATHS}
                    ),
                    parameter_schema={},
                    aggregation_semantics="fixture",
                    direction=MetricDirection.INTERPRET_ONLY,
                    unit="fixture",
                    references=(),
                    evaluator=lambda context: MetricValue(1.0, "fixture"),
                    optional_dependencies=("avrw_dependency_that_does_not_exist",),
                )
            )
            plan = EvaluationEngine(registry).dry_run(
                _protocol(run, (MetricRequest("dependency_probe", {}),))
            )
            self.assertEqual(plan.metrics[0]["status"], "unavailable")


class MetricDefinitionTest(unittest.TestCase):
    def test_known_tfp_auc_nss_and_auc_judd_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saliency = root / "saliency.json"
            saliency.write_text(json.dumps([[1.0, 0.0], [0.0, 0.0]]), encoding="utf-8")
            truth = GroundTruthScanpath(
                scanpath_id="tiny:observer",
                observer_id="observer",
                task_text="find target",
                target_label="target",
                fixations=(
                    FixationEvent(
                        x_px=0.0,
                        y_px=0.0,
                        x_norm=0.0,
                        y_norm=0.0,
                        sequence_index=0,
                    ),
                ),
            )
            item = DatasetItem(
                dataset_id="tiny",
                dataset_version="v1",
                official_split="test",
                item_id="item",
                image_id="image",
                image_ref="image.png",
                image_width=2,
                image_height=2,
                scanpaths=(truth,),
                task_text="find target",
                target_label="target",
                target_metadata={"target_present": True},
                target_boxes=(
                    TargetBoundingBox(
                        x_px=1.0,
                        y_px=1.0,
                        width_px=1.0,
                        height_px=1.0,
                        target_label="target",
                    ),
                ),
            )
            prediction = PredictionRecord(
                run_id="run",
                request_id="request",
                model=ModelIdentity(
                    model_id="fixture",
                    model_variant="v1",
                    adapter_version="v1",
                    upstream_version="fixture",
                    checkpoint_id="none",
                ),
                dataset_id="tiny",
                dataset_split="test",
                item_id="item",
                sample_id="sample",
                sample_index=0,
                seed=1,
                observer_id="observer",
                task_text="find target",
                target_description="target",
                fixations=(
                    FixationEvent(x_px=0.0, y_px=0.0, sequence_index=0),
                    FixationEvent(x_px=1.0, y_px=1.0, sequence_index=1),
                    FixationEvent(x_px=0.0, y_px=1.0, sequence_index=2),
                ),
                saliency_artifact=ArtifactReference(
                    uri="saliency.json",
                    kind="saliency_map",
                    media_type="application/json",
                ),
            )
            protocol = EvaluationProtocol(
                schema_version=1,
                run_directory=str(root),
                dataset=GroundTruthConfig(
                    adapter="json",
                    dataset_id="tiny",
                    version="v1",
                    root=str(root),
                    split="test",
                    task_type="visual_search_target_present",
                    options={"annotation_name": "unused.json"},
                ),
                metrics=(MetricRequest("fixation_count", {}),),
                sample_aggregation=SampleAggregationConfig(),
                observer_aggregation=ObserverAggregationConfig(),
                temporal_alignment=TemporalAlignmentConfig(),
                coordinate_conversion=CoordinateConversionConfig(),
                output_directory=str(root / "metrics"),
                existing_output=ExistingOutputPolicy.CREATE,
            )
            registry = create_default_metric_registry()

            def context(
                name: str,
                parameters: dict[str, object],
                predictions: tuple[PredictionRecord, ...] = (prediction,),
            ) -> MetricContext:
                definition = registry.get(name)
                return MetricContext(
                    predictions=predictions,
                    item=item,
                    ground_truth=(truth,),
                    run_directory=root,
                    output_directory=root / "metrics",
                    protocol=protocol,
                    parameters=definition.resolve_parameters(parameters),
                    artifact_stem="known",
                )

            tfp = registry.get("tfp_auc").evaluator(
                context(
                    "tfp_auc",
                    {
                        "max_fixations": 3,
                        "include_initial_fixation": True,
                        "normalization": "raw_trapezoid",
                    },
                )
            )
            nss = registry.get("nss").evaluator(context("nss", {}))
            auc = registry.get("auc_judd").evaluator(context("auc_judd", {}))
            self.assertEqual(tfp.value, 1.5)
            self.assertAlmostEqual(nss.value, math.sqrt(3.0))
            self.assertEqual(auc.value, 1.0)

            density_prediction = replace(
                prediction,
                fixations=(
                    FixationEvent(x_px=0.0, y_px=0.0, sequence_index=0),
                ),
            )
            density_parameters = {
                "kernel": "gaussian",
                "bandwidth_norm": 0.01,
                "grid_width": 3,
                "grid_height": 3,
            }
            density_auc = registry.get("auc_judd_fixation_density").evaluator(
                context(
                    "auc_judd_fixation_density",
                    {**density_parameters, "tie_policy": "average_rank"},
                    (density_prediction,),
                )
            )
            density_nss = registry.get("nss_fixation_density").evaluator(
                context(
                    "nss_fixation_density",
                    density_parameters,
                    (density_prediction,),
                )
            )
            self.assertEqual(density_auc.value, 1.0)
            self.assertAlmostEqual(density_nss.value, math.sqrt(8.0))
            self.assertEqual(
                density_auc.artifact_reference,
                density_nss.artifact_reference,
            )


class EvaluationEngineTest(unittest.TestCase):
    def test_stochastic_outputs_kde_and_no_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _completed_run(Path(temporary))
            protocol = _protocol(
                run,
                (
                    MetricRequest("fixation_count", {}),
                    MetricRequest("nss", {}),
                    MetricRequest("endpoint_pairwise_distance_normalized", {}),
                    MetricRequest("endpoint_dispersion_normalized", {}),
                    MetricRequest(
                        "fixation_density_kde",
                        {
                            "kernel": "gaussian",
                            "bandwidth_norm": 0.1,
                            "grid_width": 8,
                            "grid_height": 6,
                        },
                    ),
                    MetricRequest(
                        "auc_judd_fixation_density",
                        {
                            "kernel": "gaussian",
                            "bandwidth_norm": 0.1,
                            "grid_width": 8,
                            "grid_height": 6,
                            "tie_policy": "average_rank",
                        },
                    ),
                    MetricRequest(
                        "nss_fixation_density",
                        {
                            "kernel": "gaussian",
                            "bandwidth_norm": 0.1,
                            "grid_width": 8,
                            "grid_height": 6,
                        },
                    ),
                ),
            )
            result = EvaluationEngine().evaluate(protocol)
            self.assertEqual(result.status, "completed")
            output = result.output_directory
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "artifacts",
                    "metric_versions.json",
                    "per_item.csv",
                    "per_sample.csv",
                    "per_observer.csv",
                    "protocol.yaml",
                    "summary.json",
                    "warnings.jsonl",
                },
            )
            sample_rows = _csv_rows(output / "per_sample.csv")
            fixation_rows = [
                row for row in sample_rows if row["metric_name"] == "fixation_count"
            ]
            self.assertEqual(len(fixation_rows), 2)
            self.assertEqual({row["sample_index"] for row in fixation_rows}, {"0", "1"})
            self.assertEqual(len({row["sample_id"] for row in fixation_rows}), 2)
            nss_rows = [row for row in sample_rows if row["metric_name"] == "nss"]
            self.assertEqual({row["status"] for row in nss_rows}, {"not_applicable"})
            self.assertEqual({row["value"] for row in nss_rows}, {""})
            item_rows = _csv_rows(output / "per_item.csv")
            fixation_item = next(
                row for row in item_rows if row["metric_name"] == "fixation_count"
            )
            self.assertEqual(fixation_item["count"], "2")
            self.assertNotEqual(fixation_item["mean"], "")
            self.assertNotEqual(fixation_item["std"], "")
            density_rows = {
                row["metric_name"]: row
                for row in item_rows
                if row["metric_name"]
                in {"auc_judd_fixation_density", "nss_fixation_density"}
            }
            self.assertEqual(set(density_rows), {
                "auc_judd_fixation_density",
                "nss_fixation_density",
            })
            self.assertEqual(
                {row["status"] for row in density_rows.values()},
                {"applicable"},
            )
            self.assertEqual(
                len(
                    {
                        row["artifact_reference"]
                        for row in density_rows.values()
                    }
                ),
                1,
            )
            versions = json.loads((output / "metric_versions.json").read_text())
            kde = next(
                metric
                for metric in versions["metrics"]
                if metric["name"] == "fixation_density_kde"
            )
            self.assertEqual(kde["resolved_parameters"]["kernel"], "gaussian")
            self.assertEqual(kde["resolved_parameters"]["bandwidth_norm"], 0.1)
            artifact = next((output / "artifacts").glob("*.json"))
            density = json.loads(artifact.read_text())
            self.assertAlmostEqual(
                sum(sum(row) for row in density["density"]), 1.0
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["secondary_best_of_n"]["enabled"])
            self.assertEqual(summary["secondary_best_of_n"]["results"], [])
            self.assertTrue(summary["text_evaluation"]["preserved_outputs"])
            banned = {"rank", "ranking", "winner", "global_score", "overall_score"}
            self.assertFalse(banned.intersection(_all_keys(summary)))

    def test_temporal_ground_truth_absence_and_strict_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation_item = _run_item(
                item_id="scene-b",
                observer_id="observer-a",
                task_text="find the white square",
                image_name="tall.png",
                width=50,
                height=100,
            )
            validation_run = _completed_run(
                root / "validation",
                items=(validation_item,),
                split="validation",
            )
            validation_protocol = _protocol(
                validation_run,
                (MetricRequest("duration_mae_s", {}),),
                split="validation",
            )
            EvaluationEngine().evaluate(validation_protocol)
            rows = _csv_rows(validation_run / "metrics" / "per_sample.csv")
            self.assertEqual({row["status"] for row in rows}, {"not_applicable"})
            self.assertTrue(all("ground-truth timing" in row["message"] for row in rows))

            strict_run = _completed_run(root / "strict")
            strict_protocol = _protocol(
                strict_run,
                (MetricRequest("duration_mae_s", {}),),
                temporal_policy="strict_index",
            )
            EvaluationEngine().evaluate(strict_protocol)
            strict_rows = _csv_rows(strict_run / "metrics" / "per_sample.csv")
            self.assertEqual({row["status"] for row in strict_rows}, {"invalid"})
            self.assertTrue(all("equal sequence lengths" in row["message"] for row in strict_rows))

    def test_single_dataset_task_accepts_model_specific_instruction_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            item = _run_item(
                item_id="scene-a",
                observer_id="observer-a",
                task_text="Question: Is there a circle in the image? Answer: yes.",
            )
            run = _completed_run(Path(temporary), items=(item,))
            protocol = _protocol(
                run,
                (MetricRequest("duration_mae_s", {}),),
                temporal_policy="truncate_to_shorter",
            )
            EvaluationEngine().evaluate(protocol)
            rows = _csv_rows(run / "metrics" / "per_sample.csv")
            self.assertEqual({row["status"] for row in rows}, {"applicable"})

    def test_observers_are_preserved_and_macro_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            items = tuple(
                _run_item(
                    item_id=f"scene-a:{observer}",
                    observer_id=observer,
                    task_text="find the yellow circle",
                )
                for observer in ("observer-a", "observer-b")
            )
            run = _completed_run(Path(temporary), items=items)
            protocol = _protocol(run, (MetricRequest("fixation_count", {}),))
            EvaluationEngine().evaluate(protocol)
            sample_rows = _csv_rows(run / "metrics" / "per_sample.csv")
            self.assertEqual(
                {row["observer_id"] for row in sample_rows},
                {"observer-a", "observer-b"},
            )
            observer_rows = _csv_rows(run / "metrics" / "per_observer.csv")
            self.assertEqual(
                {row["observer_id"] for row in observer_rows},
                {"observer-a", "observer-b"},
            )
            summary = json.loads((run / "metrics" / "summary.json").read_text())
            group = next(
                group
                for group in summary["groups"]
                if group["metric_name"] == "fixation_count"
            )
            self.assertEqual(group["aggregation"], "macro_across_observers")
            self.assertEqual(group["count_observers"], 2)

    def test_exact_task_groups_are_not_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            items = (
                _run_item(
                    item_id="scene-a:observer-a",
                    observer_id="observer-a",
                    task_text="find the yellow circle",
                ),
                _run_item(
                    item_id="scene-a:observer-b",
                    observer_id="observer-b",
                    task_text="find another exact task",
                ),
            )
            run = _completed_run(Path(temporary), items=items, num_samples=1)
            EvaluationEngine().evaluate(
                _protocol(run, (MetricRequest("fixation_count", {}),))
            )
            summary = json.loads((run / "metrics" / "summary.json").read_text())
            tasks = {
                group["task_text"]
                for group in summary["groups"]
                if group["metric_name"] == "fixation_count"
            }
            self.assertEqual(
                tasks,
                {"find the yellow circle", "find another exact task"},
            )

    def test_metric_failure_is_isolated_and_resume_has_no_duplicates(self) -> None:
        def fail_metric(context) -> MetricValue:
            del context
            raise RuntimeError("intentional metric fixture failure")

        with tempfile.TemporaryDirectory() as temporary:
            run = _completed_run(Path(temporary))
            registry = create_default_metric_registry()
            registry.register(
                MetricDefinition(
                    name="failing_fixture_metric",
                    implementation_version="1.0.0",
                    kind=MetricKind.DIAGNOSTIC,
                    scope=MetricScope.SAMPLE,
                    required_prediction_fields=("scanpath_record",),
                    required_ground_truth_fields=(),
                    supported_task_types=frozenset({"any"}),
                    compatible_capabilities=frozenset(
                        {Capability.PRODUCES_SCANPATHS}
                    ),
                    parameter_schema={},
                    aggregation_semantics="fixture",
                    direction=MetricDirection.INTERPRET_ONLY,
                    unit="fixture",
                    references=(),
                    evaluator=fail_metric,
                )
            )
            protocol = _protocol(
                run,
                (
                    MetricRequest("fixation_count", {}),
                    MetricRequest("failing_fixture_metric", {}),
                ),
            )
            engine = EvaluationEngine(registry)
            first = engine.evaluate(protocol)
            self.assertEqual(first.status, "completed_with_metric_errors")
            first_rows = _csv_rows(run / "metrics" / "per_sample.csv")
            self.assertEqual(
                {
                    row["status"]
                    for row in first_rows
                    if row["metric_name"] == "failing_fixture_metric"
                },
                {"error"},
            )
            resumed = engine.evaluate(
                protocol.with_overrides(existing_output=ExistingOutputPolicy.RESUME)
            )
            second_rows = _csv_rows(run / "metrics" / "per_sample.csv")
            self.assertEqual(resumed.sample_row_count, len(first_rows))
            self.assertEqual(len(second_rows), len(first_rows))
            warning_lines = (run / "metrics" / "warnings.jsonl").read_text().splitlines()
            self.assertEqual(len(warning_lines), 2)
            keys = {
                (row["request_id"], row["sample_id"], row["metric_name"])
                for row in second_rows
            }
            self.assertEqual(len(keys), len(second_rows))


if __name__ == "__main__":
    unittest.main()

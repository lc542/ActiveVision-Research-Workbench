from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from PIL import Image

from activevision_workbench.adapters.dummy import DUMMY_CAPABILITIES
from activevision_workbench.cli import main as cli_main
from activevision_workbench.contracts import (
    Capability,
    CapabilitySet,
    FixationEvent,
    ModelIdentity,
    PredictionRecord,
    StoppingReason,
)
from activevision_workbench.errors import InspectionDataError
from activevision_workbench.inspection import (
    CompareRequest,
    FigureFormat,
    InspectRequest,
    InspectionEngine,
    build_instruction_run_config,
    validate_instruction_capability,
)
from activevision_workbench.run_artifacts import RunDirectory, RunStatus
from activevision_workbench.run_config import RunConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ANNOTATIONS = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "datasets" / "tiny_gaze" / "annotations.json"
)


def _capabilities(model_id: str) -> CapabilitySet:
    values = {Capability.PRODUCES_SCANPATHS, Capability.STOCHASTIC_MULTI_SAMPLE}
    if model_id == "tpp_gaze":
        values.update(
            {Capability.CONTINUOUS_TIME_OUTPUT, Capability.FIXATION_DURATION_OUTPUT}
        )
    if model_id == "individualscanpath":
        values.update(
            {Capability.OBSERVER_CONDITIONED, Capability.FIXATION_DURATION_OUTPUT}
        )
    if model_id == "gazexplain":
        values.update(
            {
                Capability.INSTRUCTION_CONDITIONED,
                Capability.TEXT_EXPLANATION_OUTPUT,
                Capability.FIXATION_DURATION_OUTPUT,
            }
        )
    return CapabilitySet(frozenset(values))


def _dataset(root: Path, *, image_size: tuple[int, int] = (100, 50)) -> Path:
    dataset = root / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    Image.new("RGB", image_size, "#E9E5DC").save(images / "scene.png")
    value = json.loads(SOURCE_ANNOTATIONS.read_text(encoding="utf-8"))
    item = value["items"][0]
    item["image_ref"] = "images/scene.png"
    item["image_width"], item["image_height"] = image_size
    value["items"] = [item]
    value["splits"] = {"train": ["scene-a"]}
    (dataset / "annotations.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return dataset


def _config(
    root: Path,
    *,
    run_id: str,
    model_id: str,
    dataset: Path,
    items: list[dict[str, object]],
    num_samples: int,
) -> RunConfig:
    return RunConfig.from_dict(
        {
            "schema_version": 1,
            "run_id": run_id,
            "model": {
                "id": model_id,
                "variant": "fixture-v1",
                "options": {model_id: {}},
                "checkpoint_path": None,
                "checkpoint_fingerprint": None,
                "upstream_code_ref": "fixture",
                "upstream_commit": "fixture-commit",
            },
            "dataset": {
                "id": "tiny_gaze",
                "version": "fixture-v1",
                "split": "train",
                "root": str(dataset),
                "root_fingerprint": "sha256:fixture",
                "items": items,
                "item_subset": None,
            },
            "request": {
                "task_text": None,
                "target_description": None,
                "observer_id": None,
                "observer_metadata": {},
                "num_samples": num_samples,
                "base_seed": 17,
                "seed_strategy": "increment_per_item",
                "max_fixations": 3,
                "time_horizon_s": 1.0 if model_id == "tpp_gaze" else None,
            },
            "device": "cpu",
            "output_root": str(root / "runs"),
            "run_policy": "create",
            "tags": ["fixture"],
            "notes": None,
            "environment": {"name": "fixture", "image_digest": None},
        },
        base_dir=root,
    )


def _item(
    image: Path,
    *,
    item_id: str = "scene-a",
    width: int = 100,
    height: int = 50,
    observer_id: str | None = None,
    task_text: str | None = "find the yellow circle",
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "image_ref": str(image),
        "image_width": width,
        "image_height": height,
        "task_text": task_text,
        "target_description": "circle",
        "observer_id": observer_id,
        "observer_metadata": {},
    }


def _make_run(
    root: Path,
    *,
    run_id: str,
    model_id: str,
    items: list[dict[str, object]] | None = None,
    num_samples: int = 2,
    image_size: tuple[int, int] = (100, 50),
) -> tuple[Path, RunConfig]:
    dataset = _dataset(root / run_id, image_size=image_size)
    image = dataset / "images" / "scene.png"
    selected_items = items or [_item(image, width=image_size[0], height=image_size[1])]
    config = _config(
        root / run_id,
        run_id=run_id,
        model_id=model_id,
        dataset=dataset,
        items=selected_items,
        num_samples=num_samples,
    )
    capabilities = _capabilities(model_id)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": RunStatus.CREATED.value,
        "configuration_hash": config.configuration_hash,
        "model": {
            "id": model_id,
            "variant": "fixture-v1",
            "adapter_version": "fixture",
            "upstream_version": "fixture",
            "capabilities": capabilities.to_dict(),
        },
        "dataset": {"id": "tiny_gaze", "version": "fixture-v1", "split": "train"},
        "counts": {"total": len(selected_items), "succeeded": len(selected_items), "failed": 0},
        "warnings": [],
    }
    directory = RunDirectory.create(config, manifest)
    records = []
    identity = ModelIdentity(
        model_id=model_id,
        model_variant="fixture-v1",
        adapter_version="fixture",
        upstream_version="fixture",
        checkpoint_id="fixture",
    )
    for request in config.requests():
        for sample_index in range(num_samples):
            offset = sample_index * 5
            timed = model_id == "tpp_gaze"
            fixations = tuple(
                FixationEvent(
                    x_px=float(10 + offset + index * 25),
                    y_px=float(10 + index * 12),
                    x_norm=float(10 + offset + index * 25) / (request.image_width - 1),
                    y_norm=float(10 + index * 12) / (request.image_height - 1),
                    sequence_index=index,
                    timestamp_s=index * 0.2 if timed else None,
                    duration_s=0.08 + index * 0.04,
                    native_timing={"unit": "seconds"},
                )
                for index in range(3)
            )
            explanation = (
                f"Explanation for exact task: {request.task_text}"
                if model_id == "gazexplain"
                else None
            )
            warnings = (
                ("decoding truncated at configured maximum",)
                if model_id == "gazexplain" and sample_index == 0
                else ()
            )
            records.append(
                PredictionRecord(
                    run_id=run_id,
                    request_id=request.request_id,
                    model=identity,
                    dataset_id=request.dataset_id,
                    dataset_split=request.dataset_split,
                    item_id=request.item_id,
                    sample_id=f"{request.item_id}-sample-{sample_index:04d}",
                    sample_index=sample_index,
                    seed=request.base_seed + sample_index,
                    observer_id=request.observer_id,
                    task_text=request.task_text,
                    target_description=request.target_description,
                    fixations=fixations,
                    explanation_text=explanation,
                    warnings=warnings,
                    native_metadata={model_id: {"fixture": True}},
                    stopping_reason=(
                        StoppingReason.TIME_HORIZON
                        if timed
                        else StoppingReason.MODEL_STOP
                    ),
                )
            )
    directory.append_predictions(records)
    manifest["status"] = RunStatus.COMPLETED.value
    directory.write_manifest(manifest)
    return directory.path, config


def _add_evaluation(run: Path) -> None:
    metrics = run / "metrics"
    artifacts = metrics / "artifacts"
    artifacts.mkdir(exist_ok=True)
    prediction = RunDirectory.open(run).read_predictions()[0]
    fields = (
        "item_id",
        "request_id",
        "sample_id",
        "metric_name",
        "status",
        "mean",
        "value",
        "unit",
        "artifact_reference",
    )
    with (metrics / "per_item.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "item_id": prediction.item_id,
                "request_id": prediction.request_id,
                "sample_id": "",
                "metric_name": "endpoint_dispersion_normalized",
                "status": "applicable",
                "mean": "0.125",
                "value": "",
                "unit": "image_diagonals",
                "artifact_reference": "",
            }
        )
        writer.writerow(
            {
                "item_id": prediction.item_id,
                "request_id": prediction.request_id,
                "sample_id": "",
                "metric_name": "fixation_density_kde",
                "status": "applicable",
                "mean": "",
                "value": "",
                "unit": "probability_density_artifact",
                "artifact_reference": "artifacts/kde.json",
            }
        )
    (artifacts / "kde.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metric_name": "fixation_density_kde",
                "grid_width": 2,
                "grid_height": 2,
                "density": [[0.1, 0.2], [0.3, 0.4]],
            }
        ),
        encoding="utf-8",
    )


class InspectionTest(unittest.TestCase):
    def test_scandiff_headless_kde_samples_and_exact_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = _make_run(root, run_id="scandiff-run", model_id="scandiff", num_samples=3)
            _add_evaluation(run)
            engine = InspectionEngine()
            result = engine.inspect(
                InspectRequest(run_directory=run, item_id="scene-a")
            )
            digest = sha256(b"scene-a").hexdigest()[:10]
            self.assertEqual(result.figure_path.name, f"inspect-scene-a-{digest}.png")
            self.assertEqual(
                result.manifest_path.name,
                f"inspect-scene-a-{digest}.manifest.json",
            )
            with Image.open(result.figure_path) as figure:
                self.assertEqual(figure.format, "PNG")
                self.assertGreater(figure.width, 1000)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            kinds = [panel["kind"] for panel in manifest["figure"]["panels"]]
            self.assertEqual(kinds.count("sample_small_multiple"), 3)
            self.assertTrue(manifest["figure"]["panels"][0]["kde_rendered"])
            self.assertEqual(len(manifest["runs"][0]["predictions"]), 3)
            self.assertNotIn("ranking", manifest)
            repeated = engine.inspect(
                InspectRequest(
                    run_directory=run,
                    item_id="scene-a",
                    overwrite=True,
                )
            )
            self.assertEqual(repeated.figure_sha256, result.figure_sha256)

    def test_missing_image_and_record_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = _make_run(root, run_id="missing-run", model_id="scandiff")
            with self.assertRaisesRegex(InspectionDataError, "no prediction record"):
                InspectionEngine().dry_run_inspect(
                    InspectRequest(run_directory=run, item_id="absent")
                )
            image = next((run.parent.parent / "dataset").glob("images/scene.png"), None)
            if image is None:
                config = RunConfig.from_file(run / "config.yaml")
                image = Path(config.items[0].image_ref)
            image.unlink()
            with self.assertRaisesRegex(InspectionDataError, "source image is missing"):
                InspectionEngine().dry_run_inspect(
                    InspectRequest(run_directory=run, item_id="scene-a")
                )

    def test_compare_rejects_mismatched_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = _make_run(root, run_id="run-a", model_id="scandiff")
            second, _ = _make_run(
                root,
                run_id="run-b",
                model_id="scandiff",
                image_size=(120, 60),
            )
            with self.assertRaisesRegex(InspectionDataError, "same image bytes and dimensions"):
                InspectionEngine().dry_run_compare(
                    CompareRequest(
                        run_directories=(first, second),
                        item_id="scene-a",
                    )
                )

    def test_individual_observer_selection_and_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _dataset(root / "source")
            image = dataset / "images" / "scene.png"
            items = [
                _item(image, item_id="scene-a:observer-a", observer_id="observer-a"),
                _item(image, item_id="scene-a:observer-b", observer_id="observer-b"),
            ]
            run, _ = _make_run(
                root,
                run_id="observer-run",
                model_id="individualscanpath",
                items=items,
                num_samples=1,
            )
            output = root / "observer-view"
            result = InspectionEngine().inspect(
                InspectRequest(
                    run_directory=run,
                    item_id="scene-a",
                    observer_id="observer-b",
                    output_directory=output,
                )
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            prediction = manifest["runs"][0]["predictions"][0]
            self.assertEqual(prediction["observer_id"], "observer-b")
            self.assertEqual(
                manifest["runs"][0]["ground_truth_scanpaths"][0]["observer_id"],
                "observer-b",
            )

    def test_gazexplain_exact_task_and_explanation_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = _make_run(root, run_id="gaze-run", model_id="gazexplain", num_samples=1)
            exact = "find the yellow circle"
            result = InspectionEngine().inspect(
                InspectRequest(
                    run_directory=run,
                    item_id="scene-a",
                    task_text=exact,
                )
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            prediction = manifest["runs"][0]["predictions"][0]
            self.assertEqual(prediction["task_text"], exact)
            self.assertIn(exact, prediction["explanation_text"])
            self.assertIn("decoding truncated", " ".join(manifest["warnings"]))

    def test_tpp_timeline_and_temporal_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = _make_run(root, run_id="tpp-run", model_id="tpp_gaze", num_samples=2)
            result = InspectionEngine().inspect(
                InspectRequest(run_directory=run, item_id="scene-a")
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            panels = manifest["figure"]["panels"]
            timeline = next(
                panel
                for panel in panels
                if panel["kind"] == "continuous_time_timeline"
            )
            self.assertEqual(timeline["unit"], "seconds")
            self.assertEqual(len(timeline["sample_ids"]), 2)

    def test_compare_manifest_references_selected_metric_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = _make_run(root, run_id="first", model_id="scandiff")
            second, _ = _make_run(root, run_id="second", model_id="gazexplain")
            _add_evaluation(first)
            output = root / "comparison"
            result = InspectionEngine().compare(
                CompareRequest(
                    run_directories=(first, second),
                    item_id="scene-a",
                    output_directory=output,
                )
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["command"], "compare")
            self.assertEqual(len(manifest["runs"]), 2)
            self.assertGreater(len(manifest["runs"][0]["evaluation_rows"]), 0)
            self.assertFalse(manifest["inference_rerun"])

    def test_cli_inspect_dry_run_has_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = _make_run(root, run_id="cli-run", model_id="scandiff")
            output = root / "dry"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "inspect",
                        "--run",
                        str(run),
                        "--item-id",
                        "scene-a",
                        "--output-dir",
                        str(output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            self.assertTrue(json.loads(stdout.getvalue())["dry_run"])

    def test_instruction_requests_are_distinct_and_comparison_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _dataset(root / "base")
            image = dataset / "images" / "scene.png"
            base = _config(
                root,
                run_id="base",
                model_id="dummy",
                dataset=dataset,
                items=[_item(image, observer_id="observer-a")],
                num_samples=1,
            )
            exact = ("inspect the cup", "inspect the clock")
            generated = build_instruction_run_config(
                base,
                model_id="dummy",
                image_path=image,
                instructions=exact,
                output_root=root / "instruction-runs",
            )
            requests = generated.requests()
            self.assertEqual(tuple(request.task_text for request in requests), exact)
            self.assertEqual(len({request.request_id for request in requests}), 2)
            validate_instruction_capability(DUMMY_CAPABILITIES)

            items = [
                _item(image, item_id=f"instruction-{index:04d}", task_text=text)
                for index, text in enumerate(exact)
            ]
            run, _ = _make_run(
                root,
                run_id="instruction-run",
                model_id="gazexplain",
                items=items,
                num_samples=1,
            )
            result = InspectionEngine().compare_instructions(
                run,
                exact,
                output_directory=root / "instruction-figures",
                figure_format=FigureFormat.PNG,
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["selectors"]["instructions"], list(exact))
            self.assertEqual(result.figure_path.name, "compare-instructions.png")
            self.assertEqual(
                result.manifest_path.name,
                "compare-instructions.manifest.json",
            )
            self.assertEqual(len(manifest["figure"]["panels"]), 2)

    def test_compare_instructions_cli_runs_explicit_dummy_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _dataset(root / "base")
            image = dataset / "images" / "scene.png"
            base = _config(
                root,
                run_id="base",
                model_id="dummy",
                dataset=dataset,
                items=[_item(image, observer_id="observer-a")],
                num_samples=1,
            )
            base = replace(base, model_variant="deterministic-v1")
            config_path = root / "base.json"
            config_path.write_text(
                json.dumps(base.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            output = root / "experiment"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "compare-instructions",
                        "--config",
                        str(config_path),
                        "--model",
                        "dummy",
                        "--image",
                        str(image),
                        "--instruction",
                        "inspect the cup",
                        "--instruction",
                        "inspect the clock",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            manifest_path = output / "figures" / "compare-instructions.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["selectors"]["instructions"],
                ["inspect the cup", "inspect the clock"],
            )
            self.assertTrue(manifest["inference_rerun"])
            self.assertTrue((output / "figures" / "compare-instructions.png").is_file())


if __name__ == "__main__":
    unittest.main()

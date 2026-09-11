
from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from activevision_workbench import (
    COCOSearch18DatasetAdapter,
    CoordinateClippedWarning,
    CoordinateOutOfBoundsError,
    DataRootError,
    DatasetAnnotationError,
    DatasetItem,
    DatasetItemNotFoundError,
    JsonDatasetAdapter,
    OSIEDatasetAdapter,
    RunConfig,
    RunEngine,
    RunStatus,
    normalized_to_pixel,
    pixel_to_normalized,
    read_predictions_jsonl,
)
from activevision_workbench.adapters.dummy import create_dummy_registry
from activevision_workbench.cli import main as cli_main

REPOSITORY = Path(__file__).resolve().parents[1]
TINY_ROOT = REPOSITORY / "tests" / "fixtures" / "datasets" / "tiny_gaze"
MALFORMED_ROOT = (
    REPOSITORY / "tests" / "fixtures" / "datasets" / "malformed"
)


class CoordinateTests(unittest.TestCase):
    def test_pixel_normalized_round_trip_uses_width_and_height_correctly(self) -> None:
        coordinate = pixel_to_normalized(
            x_px=49.5,
            y_px=24.5,
            image_width=100,
            image_height=50,
        )
        self.assertAlmostEqual(coordinate.x_norm, 0.5, places=12)
        self.assertAlmostEqual(coordinate.y_norm, 0.5, places=12)
        restored = normalized_to_pixel(
            x_norm=coordinate.x_norm,
            y_norm=coordinate.y_norm,
            image_width=100,
            image_height=50,
        )
        self.assertAlmostEqual(restored.x_px, 49.5, places=12)
        self.assertAlmostEqual(restored.y_px, 24.5, places=12)

    def test_out_of_bounds_rejected_unless_clipping_is_explicit(self) -> None:
        with self.assertRaises(CoordinateOutOfBoundsError):
            pixel_to_normalized(
                x_px=100.0,
                y_px=10.0,
                image_width=100,
                image_height=50,
            )
        with self.assertWarns(CoordinateClippedWarning):
            clipped = pixel_to_normalized(
                x_px=100.0,
                y_px=-2.0,
                image_width=100,
                image_height=50,
                clip=True,
            )
        self.assertTrue(clipped.clipped)
        self.assertEqual((clipped.x_px, clipped.y_px), (99.0, 0.0))
        self.assertEqual(
            (clipped.original_x_px, clipped.original_y_px), (100.0, -2.0)
        )


class JsonDatasetAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = JsonDatasetAdapter(TINY_ROOT)

    def test_stable_iteration_lookup_and_split_filtering(self) -> None:
        first = tuple(
            item.item_id for item in self.adapter.iter_items(split="train")
        )
        second = tuple(
            item.item_id for item in self.adapter.iter_items(split="train")
        )
        self.assertEqual(first, second)
        self.assertEqual(first, ("scene-a",))
        self.assertEqual(self.adapter.item_ids(split="validation"), ("scene-b",))
        item = self.adapter.get_item("scene-b", split="validation")
        self.assertEqual((item.image_width, item.image_height), (50, 100))
        with self.assertRaises(DatasetItemNotFoundError):
            self.adapter.get_item("missing")

    def test_observer_and_task_filters_preserve_metadata(self) -> None:
        filtered = tuple(
            self.adapter.iter_items(
                split="train",
                observer_ids=("observer-b",),
                tasks=("circle",),
            )
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(filtered[0].scanpaths), 1)
        self.assertEqual(filtered[0].scanpaths[0].observer_id, "observer-b")
        self.assertEqual(filtered[0].scanpaths[0].target_label, "circle")
        self.assertEqual(
            tuple(self.adapter.iter_items(split="train", tasks=("square",))), ()
        )

    def test_contract_round_trip_keeps_native_provenance_and_timing(self) -> None:
        item = self.adapter.get_item("scene-a", split="train")
        restored = DatasetItem.from_dict(item.to_dict())
        self.assertEqual(restored, item)
        self.assertEqual(len(restored.scanpaths), 2)
        fixation = restored.scanpaths[0].fixations[0]
        self.assertEqual(fixation.native_timing["unit"], "ms")
        self.assertIn("annotations.json#", restored.scanpaths[0].native_annotation.uri)

    def test_manifest_has_annotation_and_split_fingerprints(self) -> None:
        manifest = self.adapter.create_manifest()
        data = manifest.to_dict()
        self.assertEqual(data["item_counts"], {"train": 1, "validation": 1})
        self.assertTrue(data["split_definition_fingerprint"].startswith("sha256:"))
        annotation = data["annotation_fingerprints"]["annotations.json"]
        self.assertEqual(len(annotation["sha256"]), 64)
        self.assertEqual(data["configured_root"], str(TINY_ROOT.resolve()))

    def test_missing_root_and_malformed_annotation_have_clear_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "absent"
            with self.assertRaisesRegex(DataRootError, "does not exist"):
                JsonDatasetAdapter(missing)
        with self.assertRaisesRegex(DatasetAnnotationError, "items\[0\]"):
            JsonDatasetAdapter(MALFORMED_ROOT)


class COCOSearch18FixtureTests(unittest.TestCase):
    def test_source_loader_preserves_task_observer_bbox_and_oob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_coco_fixture(root)
            adapter = COCOSearch18DatasetAdapter(root)
            self.assertEqual(adapter.item_ids(split="train"), ("train:bottle:0001",))
            item = adapter.get_item("train:bottle:0001", split="train")
            self.assertEqual((item.image_width, item.image_height), (100, 50))
            self.assertEqual(item.target_label, "bottle")
            self.assertEqual(len(item.scanpaths), 2)
            self.assertEqual(item.observer_ids, ("subject-01", "subject-02"))
            self.assertEqual(item.target_boxes[0].width_px, 20.0)
            self.assertIsNone(item.scanpaths[0].fixations[0].duration_s)
            out_of_bounds = item.scanpaths[1].fixations[-1]
            self.assertEqual(out_of_bounds.x_px, 105.0)
            self.assertIsNone(out_of_bounds.x_norm)
            self.assertTrue(any("out-of-bounds" in value for value in item.warnings))
            task_filtered = tuple(
                adapter.iter_items(split="train", tasks=("bottle",))
            )
            self.assertEqual(len(task_filtered), 1)

    def test_explicit_source_clipping_warns_and_keeps_raw_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_coco_fixture(root)
            adapter = COCOSearch18DatasetAdapter(root, clip_out_of_bounds=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                item = adapter.get_item("train:bottle:0001", split="train")
            self.assertTrue(
                any(
                    issubclass(value.category, CoordinateClippedWarning)
                    for value in caught
                )
            )
            fixation = item.scanpaths[1].fixations[-1]
            self.assertEqual(fixation.x_px, 99.0)
            self.assertEqual(
                fixation.native_metadata["source_coordinate"]["x_px"],
                105.0,
            )

    def test_source_manifest_and_malformed_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_coco_fixture(root)
            adapter = COCOSearch18DatasetAdapter(root)
            manifest = adapter.create_manifest().to_dict()
            self.assertEqual(manifest["item_counts"], {"train": 1, "validation": 1})
            self.assertEqual(len(manifest["annotation_fingerprints"]), 2)
            train_path = root / "coco_search18_fixations_TP_train_split1.json"
            train_path.write_text('[{"name":"missing-fields.jpg"}]', encoding="utf-8")
            with self.assertRaisesRegex(DatasetAnnotationError, "missing field"):
                COCOSearch18DatasetAdapter(root).metadata


class OSIEFixtureTests(unittest.TestCase):
    def test_source_loader_converts_matlab_coordinates_and_keeps_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_osie_fixture(root)
            subject = SimpleNamespace(
                fix_x=[1.5, 100.0],
                fix_y=[1.5, 50.0],
                fix_duration=[120.0, 240.0],
            )
            annotation = SimpleNamespace(img="1001.jpg", subjects=[subject])
            with mock.patch(
                "activevision_workbench.datasets.osie._load_fixations_mat",
                return_value=[annotation],
            ):
                adapter = OSIEDatasetAdapter(root)
                item = adapter.get_item("1001", split="all")
            self.assertEqual((item.image_width, item.image_height), (100, 50))
            fixation = item.scanpaths[0].fixations[0]
            self.assertEqual((fixation.x_px, fixation.y_px), (0.5, 0.5))
            self.assertEqual(fixation.native_metadata["source_x_px"], 1.5)
            self.assertEqual(
                fixation.native_metadata["canonical_coordinate_transform"],
                "subtract_one",
            )
            edge = item.scanpaths[0].fixations[1]
            self.assertEqual((edge.x_px, edge.y_px), (99.0, 49.0))
            self.assertFalse(any("out-of-bounds" in value for value in item.warnings))


class DatasetRunIntegrationTests(unittest.TestCase):
    def test_fixture_observer_and_task_survive_run_engine(self) -> None:
        adapter = JsonDatasetAdapter(TINY_ROOT)
        item = adapter.get_item(
            "scene-a", split="train", observer_ids=("observer-a",)
        )
        with tempfile.TemporaryDirectory() as temporary:
            config = _run_config(
                item.to_run_item_config(observer_id="observer-a"),
                output_root=Path(temporary) / "runs",
                run_id="dataset-engine-run",
                dataset_manifest=adapter.create_manifest().to_dict(),
            )
            result = RunEngine(create_dummy_registry(), console=None).run(config)
            self.assertEqual(result.status, RunStatus.COMPLETED)
            prediction = read_predictions_jsonl(
                result.run_directory / "predictions.jsonl"
            )[0]
            self.assertEqual(prediction.observer_id, "observer-a")
            self.assertEqual(prediction.task_text, "find the yellow circle")

    def test_tiny_fixture_runs_through_activevision_cli(self) -> None:
        adapter = JsonDatasetAdapter(TINY_ROOT)
        item = adapter.get_item(
            "scene-a", split="train", observer_ids=("observer-b",)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _run_config(
                item.to_run_item_config(observer_id="observer-b"),
                output_root=root / "runs",
                run_id="dataset-cli-run",
                dataset_manifest=adapter.create_manifest().to_dict(),
            )
            config_path = root / "run.yaml"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(("run", "--config", str(config_path)))
            self.assertEqual(code, 0)
            self.assertIn("Run directory:", output.getvalue())
            self.assertTrue(
                (
                    root
                    / "runs"
                    / "dataset-cli-run"
                    / "manifest.json"
                ).is_file()
            )


def _run_config(
    run_item: dict[str, object],
    *,
    output_root: Path,
    run_id: str,
    dataset_manifest: dict[str, object],
) -> RunConfig:
    return RunConfig.from_dict(
        {
            "schema_version": 1,
            "run_id": run_id,
            "model": {
                "id": "dummy",
                "variant": "deterministic-v1",
                "options": {"dummy": {"explanation_prefix": "Dataset fixture"}},
                "checkpoint_path": None,
                "checkpoint_fingerprint": None,
                "upstream_code_ref": "builtin",
                "upstream_commit": None,
            },
            "dataset": {
                "id": "tiny_gaze",
                "version": "fixture-v1",
                "split": "train",
                "root": dataset_manifest["configured_root"],
                "root_fingerprint": dataset_manifest[
                    "split_definition_fingerprint"
                ],
                "items": [run_item],
                "item_subset": None,
            },
            "request": {
                "task_text": None,
                "target_description": None,
                "observer_id": None,
                "observer_metadata": {},
                "num_samples": 1,
                "base_seed": 7,
                "seed_strategy": "increment_per_item",
                "max_fixations": 2,
                "time_horizon_s": 1.0,
            },
            "device": "cpu",
            "output_root": str(output_root),
            "run_policy": "create",
            "tags": ["dataset-fixture"],
            "notes": "AVRW-005 integration fixture",
            "environment": {"name": "cpu-test", "image_digest": None},
        }
    )


def _write_coco_fixture(root: Path) -> None:
    image_dir = root / "images" / "bottle"
    image_dir.mkdir(parents=True)
    _write_png_header(image_dir / "0001.jpg", width=100, height=50)
    _write_png_header(image_dir / "0002.jpg", width=80, height=40)
    train = [
        _coco_record(name="0001.jpg", subject=1, split="train"),
        _coco_record(
            name="0001.jpg",
            subject=2,
            split="train",
            xs=(50.0, 105.0),
        ),
    ]
    validation = [_coco_record(name="0002.jpg", subject=1, split="valid")]
    (root / "coco_search18_fixations_TP_train_split1.json").write_text(
        json.dumps(train), encoding="utf-8"
    )
    (root / "coco_search18_fixations_TP_validation_split1.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )


def _write_osie_fixture(root: Path) -> None:
    eye_dir = root / "eye"
    stimuli_dir = root / "stimuli"
    eye_dir.mkdir(parents=True)
    stimuli_dir.mkdir(parents=True)
    (eye_dir / "fixations.mat").write_bytes(b"synthetic-test-placeholder")
    _write_png_header(stimuli_dir / "1001.jpg", width=100, height=50)


def _coco_record(
    *,
    name: str,
    subject: int,
    split: str,
    xs: tuple[float, ...] = (50.0,),
) -> dict[str, object]:
    return {
        "name": name,
        "subject": subject,
        "task": "bottle",
        "condition": "present",
        "bbox": [10, 5, 20, 25],
        "X": list(xs),
        "Y": [20.0] * len(xs),
        "T": [100] * len(xs),
        "length": len(xs),
        "correct": 1,
        "RT": 500,
        "split": split,
    }


def _write_png_header(path: Path, *, width: int, height: int) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height)
    )


if __name__ == "__main__":
    unittest.main()

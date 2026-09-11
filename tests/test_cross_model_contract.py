
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from activevision_workbench import (
    CompareRequest,
    ConfidenceIntervalConfig,
    CoordinateConversionConfig,
    DatasetItem,
    EvaluationEngine,
    EvaluationProtocol,
    ExistingOutputPolicy,
    FixationEvent,
    GazeXplainAdapterConfig,
    GroundTruthConfig,
    GroundTruthScanpath,
    IndividualScanpathAdapterConfig,
    InspectRequest,
    InspectionEngine,
    MetricRequest,
    ObserverAggregationConfig,
    RunConfig,
    RunEngine,
    RunItem,
    RunPolicy,
    SampleAggregationConfig,
    ScanDiffAdapterConfig,
    TemporalAlignmentConfig,
    TPPGazeAdapterConfig,
    gazexplain_descriptor,
    individual_scanpath_descriptor,
    read_predictions_jsonl,
    scandiff_descriptor,
    tpp_gaze_descriptor,
)
from activevision_workbench.adapters.gazexplain import (
    GAZEXPLAIN_JOINT_VARIANT,
    GAZEXPLAIN_UPSTREAM_COMMIT,
)
from activevision_workbench.adapters.individual_scanpath import (
    INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_CHECKPOINT_SHA256,
    INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
    INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT,
)
from activevision_workbench.adapters.scandiff import (
    SCANDIFF_FREE_VIEWING_VARIANT,
    SCANDIFF_UPSTREAM_COMMIT,
)
from activevision_workbench.adapters.tpp_gaze import (
    TPP_GAZE_TRANSFORMER_VARIANT,
    TPP_GAZE_UPSTREAM_COMMIT,
)
from activevision_workbench.registry import AdapterRegistry


class CrossModelContractTest(unittest.TestCase):

    def test_four_models_share_engine_jsonl_and_canonical_record_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "shared-image.jpg"
            image.write_bytes(b"shared-fake-image")
            registry, assets = self._registry(root)
            engine = RunEngine(registry, repository_root=root, console=None)

            results = {}
            for model_id in (
                "scandiff",
                "tpp_gaze",
                "individualscanpath",
                "gazexplain",
            ):
                result = engine.run(
                    self._run_config(
                        root=root,
                        image=image,
                        model_id=model_id,
                        assets=assets,
                    )
                )
                results[model_id] = read_predictions_jsonl(
                    result.run_directory / "predictions.jsonl"
                )

            for model_id, records in results.items():
                self.assertEqual(len(records), 2)
                self.assertEqual(
                    [record.model.model_id for record in records],
                    [model_id, model_id],
                )
                self.assertEqual(
                    [record.sample_index for record in records], [0, 1]
                )
                self.assertEqual([record.seed for record in records], [11, 12])
                self.assertTrue(
                    all(record.dataset_id == "osie" for record in records)
                )
                self.assertTrue(
                    all(record.item_id == "shared-image" for record in records)
                )
                self.assertTrue(all(record.native_artifacts for record in records))
                for record in records:
                    self.assertEqual(
                        [event.sequence_index for event in record.fixations],
                        list(range(len(record.fixations))),
                    )

            raw_key_sets = []
            for model_id in results:
                path = root / "runs" / f"cross-{model_id}" / "predictions.jsonl"
                first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                raw_key_sets.append(set(first))
            self.assertTrue(all(keys == raw_key_sets[0] for keys in raw_key_sets))

            scandiff = results["scandiff"]
            tpp_gaze = results["tpp_gaze"]
            individual = results["individualscanpath"]
            gazexplain = results["gazexplain"]
            self.assertTrue(
                all(
                    event.timestamp_s is None and event.duration_s is not None
                    for record in scandiff
                    for event in record.fixations
                )
            )
            self.assertTrue(
                all(
                    event.timestamp_s is not None and event.duration_s is not None
                    for record in tpp_gaze
                    for event in record.fixations
                )
            )
            self.assertTrue(
                all(
                    event.timestamp_s is None and event.duration_s is not None
                    for record in individual
                    for event in record.fixations
                )
            )
            self.assertTrue(all(record.observer_id is None for record in scandiff))
            self.assertTrue(all(record.observer_id is None for record in tpp_gaze))
            self.assertTrue(
                all(record.observer_id == "subject-01" for record in individual)
            )
            self.assertTrue(
                all(record.observer_id is None for record in gazexplain)
            )
            self.assertTrue(
                all(record.task_text == "Question: What do you see?" for record in gazexplain)
            )
            self.assertTrue(
                all(record.explanation_text is not None for record in gazexplain)
            )
            self.assertTrue(
                all(
                    record.explanation_text is None
                    for model_id in ("scandiff", "tpp_gaze", "individualscanpath")
                    for record in results[model_id]
                )
            )

    def test_fake_workers_complete_run_evaluate_inspect_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "shared-image.png"
            Image.new("RGB", (100, 80), "#ece7dc").save(image)
            dataset_root = self._evaluation_dataset(root, image)
            registry, assets = self._registry(root)
            engine = RunEngine(registry, repository_root=root, console=None)
            inspection = InspectionEngine()
            run_directories: dict[str, Path] = {}

            for model_id in (
                "scandiff",
                "tpp_gaze",
                "individualscanpath",
                "gazexplain",
            ):
                config = self._matrix_config(
                    self._run_config(
                        root=root,
                        image=image,
                        model_id=model_id,
                        assets=assets,
                    ),
                    model_id=model_id,
                    image=image,
                )
                result = engine.run(config)
                self.assertEqual(result.status.value, "completed")
                run_directories[model_id] = result.run_directory
                records = read_predictions_jsonl(
                    result.run_directory / "predictions.jsonl"
                )
                self.assertEqual(len(records), len(config.items) * config.num_samples)
                manifest = json.loads(
                    (result.run_directory / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(manifest["activevision_git"]["commit"], None)
                self.assertEqual(
                    manifest["model"]["upstream_commit"],
                    config.upstream_commit,
                )
                self.assertIsNotNone(
                    manifest["model"]["checkpoint"]["fingerprint"]
                )
                self.assertEqual(manifest["dataset"]["id"], "osie")
                self.assertEqual(
                    manifest["dataset"]["version"], "release-fixture-v1"
                )
                self.assertEqual(
                    manifest["configuration_hash"], config.configuration_hash
                )
                self.assertEqual(
                    len(manifest["seeds"]["actual_per_request"]),
                    len(config.items),
                )
                self.assertEqual(manifest["environment"]["name"], "fake")
                self.assertTrue(all(record.schema_version == 1 for record in records))

                evaluation = EvaluationEngine().evaluate(
                    self._evaluation_protocol(
                        result.run_directory,
                        dataset_root,
                    )
                )
                self.assertEqual(evaluation.status, "completed")
                summary = json.loads(
                    (evaluation.output_directory / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(summary["input_prediction_count"], len(records))
                self.assertIn("applicable", summary["metric_status_counts"])
                metric_versions = json.loads(
                    (evaluation.output_directory / "metric_versions.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(metric_versions["schema_version"], 1)
                self.assertEqual(len(metric_versions["metrics"]), 5)

                if model_id == "individualscanpath":
                    observers = {record.observer_id for record in records}
                    self.assertEqual(observers, {"subject-01", "subject-02"})
                    artifact = inspection.inspect(
                        InspectRequest(
                            result.run_directory,
                            "scene-a",
                            output_directory=root / "figures" / model_id,
                        )
                    )
                elif model_id == "gazexplain":
                    tasks = ("Find the yellow mug", "Find the wall clock")
                    self.assertEqual({record.task_text for record in records}, set(tasks))
                    self.assertTrue(
                        all(record.explanation_text is not None for record in records)
                    )
                    artifact = inspection.compare_instructions(
                        result.run_directory,
                        tasks,
                        output_directory=root / "figures" / model_id,
                    )
                    self.assertEqual(
                        len(summary["text_evaluation"]["preserved_outputs"]),
                        len(records),
                    )
                else:
                    artifact = inspection.inspect(
                        InspectRequest(
                            result.run_directory,
                            "scene-a",
                            output_directory=root / "figures" / model_id,
                        )
                    )
                self.assertTrue(artifact.figure_path.is_file())
                self.assertTrue(artifact.manifest_path.is_file())

            tpp_records = read_predictions_jsonl(
                run_directories["tpp_gaze"] / "predictions.jsonl"
            )
            self.assertTrue(
                all(
                    fixation.timestamp_s is not None
                    and fixation.duration_s is not None
                    and fixation.native_timing
                    for record in tpp_records
                    for fixation in record.fixations
                )
            )
            comparison = inspection.compare(
                CompareRequest(
                    (
                        run_directories["scandiff"],
                        run_directories["tpp_gaze"],
                    ),
                    "scene-a",
                    output_directory=root / "figures" / "comparison",
                )
            )
            self.assertTrue(comparison.figure_path.is_file())

            scandiff = self._matrix_config(
                self._run_config(
                    root=root,
                    image=image,
                    model_id="scandiff",
                    assets=assets,
                ),
                model_id="scandiff",
                image=image,
            )
            resumed = engine.run(scandiff, resume=True)
            resume_manifest = json.loads(
                (resumed.run_directory / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                resume_manifest["resume"]["last_decision"]["skipped_completed"],
                1,
            )
            self.assertEqual(
                len(read_predictions_jsonl(resumed.run_directory / "predictions.jsonl")),
                3,
            )

    def _matrix_config(
        self,
        base: RunConfig,
        *,
        model_id: str,
        image: Path,
    ) -> RunConfig:
        items = (
            RunItem(
                item_id="scene-a",
                image_ref=str(image),
                image_width=100,
                image_height=80,
                task_text=None,
                target_description=None,
                observer_id=None,
                observer_metadata={},
            ),
        )
        if model_id == "individualscanpath":
            items = tuple(
                RunItem(
                    item_id=f"scene-a:{observer}",
                    image_ref=str(image),
                    image_width=100,
                    image_height=80,
                    task_text=None,
                    target_description=None,
                    observer_id=observer,
                    observer_metadata={"source_subject_index": index},
                )
                for index, observer in enumerate(("subject-01", "subject-02"))
            )
        elif model_id == "gazexplain":
            items = tuple(
                RunItem(
                    item_id=f"instruction-{index:04d}",
                    image_ref=str(image),
                    image_width=100,
                    image_height=80,
                    task_text=task,
                    target_description=target,
                    observer_id=None,
                    observer_metadata={},
                )
                for index, (task, target) in enumerate(
                    (
                        ("Find the yellow mug", "yellow mug"),
                        ("Find the wall clock", "wall clock"),
                    )
                )
            )
        return replace(
            base,
            run_id=f"release-e2e-{model_id}",
            dataset_version="release-fixture-v1",
            items=items,
            item_subset=None,
            num_samples=3 if model_id == "scandiff" else 2,
            run_policy=RunPolicy.CREATE,
        )

    def _evaluation_dataset(self, root: Path, image: Path) -> Path:
        fixations = (
            FixationEvent(
                x_px=20.0,
                y_px=20.0,
                x_norm=20.0 / 99.0,
                y_norm=20.0 / 79.0,
                sequence_index=0,
                timestamp_s=0.0,
                duration_s=0.1,
            ),
            FixationEvent(
                x_px=70.0,
                y_px=45.0,
                x_norm=70.0 / 99.0,
                y_norm=45.0 / 79.0,
                sequence_index=1,
                timestamp_s=0.2,
                duration_s=0.15,
            ),
        )

        def scanpath(observer: str, task: str) -> GroundTruthScanpath:
            return GroundTruthScanpath(
                scanpath_id=f"scene-a:{observer}:{hashlib.sha256(task.encode()).hexdigest()[:8]}",
                fixations=fixations,
                observer_id=observer,
                observer_metadata={"group": "release-fixture"},
                task_text=task,
                target_label="fixture target",
            )

        tasks = ("Find the yellow mug", "Find the wall clock")
        items = [
            DatasetItem(
                dataset_id="osie",
                dataset_version="release-fixture-v1",
                official_split="test",
                item_id="scene-a",
                image_id="scene-a",
                image_ref=image.name,
                image_width=100,
                image_height=80,
                scanpaths=tuple(
                    scanpath(observer, "")
                    for observer in ("subject-01", "subject-02")
                ),
            )
        ]
        items.extend(
            DatasetItem(
                dataset_id="osie",
                dataset_version="release-fixture-v1",
                official_split="test",
                item_id=f"instruction-{index:04d}",
                image_id="scene-a",
                image_ref=image.name,
                image_width=100,
                image_height=80,
                scanpaths=(scanpath("reference", task),),
                task_text=task,
                target_label=target,
            )
            for index, (task, target) in enumerate(
                zip(tasks, ("yellow mug", "wall clock"), strict=True)
            )
        )
        data = {
            "schema_version": 1,
            "dataset_id": "osie",
            "dataset_version": "release-fixture-v1",
            "splits": {"test": [item.item_id for item in items]},
            "items": [item.to_dict() for item in items],
        }
        (root / "annotations.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        return root

    @staticmethod
    def _evaluation_protocol(
        run_directory: Path,
        dataset_root: Path,
    ) -> EvaluationProtocol:
        return EvaluationProtocol(
            schema_version=1,
            run_directory=str(run_directory),
            dataset=GroundTruthConfig(
                adapter="json",
                dataset_id="osie",
                version="release-fixture-v1",
                root=str(dataset_root),
                split="test",
            ),
            metrics=(
                MetricRequest("fixation_count"),
                MetricRequest("total_fixation_duration_s"),
                MetricRequest("timestamp_span_s"),
                MetricRequest("endpoint_dispersion_normalized"),
                MetricRequest(
                    "fixation_density_kde",
                    {
                        "kernel": "gaussian",
                        "bandwidth_norm": 0.08,
                        "grid_width": 16,
                        "grid_height": 12,
                    },
                ),
            ),
            sample_aggregation=SampleAggregationConfig(
                confidence_interval=ConfidenceIntervalConfig()
            ),
            observer_aggregation=ObserverAggregationConfig(),
            temporal_alignment=TemporalAlignmentConfig(),
            coordinate_conversion=CoordinateConversionConfig(),
            output_directory=str(run_directory / "metrics"),
            existing_output=ExistingOutputPolicy.CREATE,
        )

    def _registry(
        self, root: Path
    ) -> tuple[AdapterRegistry, dict[str, Path | str]]:
        runs = root / "runs"
        scandiff_root = root / "scandiff-upstream"
        scandiff_root.mkdir()
        scandiff_checkpoint = root / "scandiff.pth"
        scandiff_checkpoint.write_bytes(b"fake-scandiff-checkpoint")
        embeddings = root / "task-embeddings.npy"
        embeddings.write_bytes(b"fake-task-embeddings")

        tpp_root = root / "tpp-upstream"
        tpp_root.mkdir()
        tpp_checkpoint = root / "tpp.pth"
        tpp_checkpoint.write_bytes(b"fake-tpp-checkpoint")
        tpp_config = root / "tpp-config.yaml"
        tpp_config.write_text("context: transformer\n", encoding="utf-8")

        individual_root = root / "individual-upstream"
        individual_root.mkdir()
        individual_checkpoint = root / "individual.pth"
        individual_checkpoint.write_bytes(b"fake-individual-checkpoint")
        mapping = root / "observers.json"
        mapping.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_id": "osie",
                    "model_variant": INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
                    "checkpoint_sha256": (
                        "sha256:"
                        f"{INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_CHECKPOINT_SHA256}"
                    ),
                    "identity_source": "cross-model fixture",
                    "observers": [
                        {
                            "observer_id": f"subject-{index + 1:02d}",
                            "model_observer_index": index,
                            "known_checkpoint_observer": True,
                        }
                        for index in range(15)
                    ],
                }
            ),
            encoding="utf-8",
        )
        mapping_fingerprint = f"sha256:{self._sha256(mapping)}"

        gazexplain_root = root / "gazexplain-upstream"
        gazexplain_root.mkdir()
        gazexplain_checkpoint = root / "gazexplain.safetensors"
        gazexplain_checkpoint.write_bytes(b"fake-gazexplain-checkpoint")
        gazexplain_hparams = root / "gazexplain-hparams.json"
        gazexplain_hparams.write_text("{}\n", encoding="utf-8")

        registry = AdapterRegistry()
        registry.register(
            scandiff_descriptor(
                ScanDiffAdapterConfig.for_fake(
                    variant=SCANDIFF_FREE_VIEWING_VARIANT,
                    upstream_root=scandiff_root,
                    checkpoint_path=scandiff_checkpoint,
                    task_embeddings_path=embeddings,
                    output_root=runs,
                )
            )
        )
        registry.register(
            tpp_gaze_descriptor(
                TPPGazeAdapterConfig.for_fake(
                    upstream_root=tpp_root,
                    checkpoint_path=tpp_checkpoint,
                    model_config_path=tpp_config,
                    output_root=runs,
                    safety_max_fixations=8,
                )
            )
        )
        registry.register(
            individual_scanpath_descriptor(
                IndividualScanpathAdapterConfig.for_fake(
                    upstream_root=individual_root,
                    checkpoint_path=individual_checkpoint,
                    observer_mapping_path=mapping,
                    observer_mapping_fingerprint=mapping_fingerprint,
                    output_root=runs,
                )
            )
        )
        registry.register(
            gazexplain_descriptor(
                GazeXplainAdapterConfig.for_fake(
                    upstream_root=gazexplain_root,
                    checkpoint_path=gazexplain_checkpoint,
                    hparams_path=gazexplain_hparams,
                    output_root=runs,
                )
            )
        )
        return registry, {
            "scandiff_checkpoint": scandiff_checkpoint,
            "tpp_checkpoint": tpp_checkpoint,
            "individual_checkpoint": individual_checkpoint,
            "individual_root": individual_root,
            "mapping": mapping,
            "mapping_fingerprint": mapping_fingerprint,
            "gazexplain_checkpoint": gazexplain_checkpoint,
        }

    def _run_config(
        self,
        *,
        root: Path,
        image: Path,
        model_id: str,
        assets: dict[str, Path | str],
    ) -> RunConfig:
        variants = {
            "scandiff": SCANDIFF_FREE_VIEWING_VARIANT,
            "tpp_gaze": TPP_GAZE_TRANSFORMER_VARIANT,
            "individualscanpath": INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
            "gazexplain": GAZEXPLAIN_JOINT_VARIANT,
        }
        commits = {
            "scandiff": SCANDIFF_UPSTREAM_COMMIT,
            "tpp_gaze": TPP_GAZE_UPSTREAM_COMMIT,
            "individualscanpath": INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT,
            "gazexplain": GAZEXPLAIN_UPSTREAM_COMMIT,
        }
        checkpoint = assets[
            {
                "scandiff": "scandiff_checkpoint",
                "tpp_gaze": "tpp_checkpoint",
                "individualscanpath": "individual_checkpoint",
                "gazexplain": "gazexplain_checkpoint",
            }[model_id]
        ]
        options: dict[str, object] = {}
        observer_id = None
        observer_metadata: dict[str, object] = {}
        time_horizon = None
        max_fixations = 16 if model_id == "scandiff" else 8
        task_text = None
        target_description = None
        if model_id == "tpp_gaze":
            time_horizon = 1.0
        if model_id == "individualscanpath":
            observer_id = "subject-01"
            observer_metadata = {"source_subject_index": 0}
            options = {
                "upstream_root": str(assets["individual_root"]),
                "observer_mapping_path": str(assets["mapping"]),
                "observer_mapping_fingerprint": assets["mapping_fingerprint"],
                "unknown_observer_policy": "error",
            }
        if model_id == "gazexplain":
            task_text = "Question: What do you see?"
            target_description = "scene content"
        return RunConfig.from_dict(
            {
                "schema_version": 1,
                "run_id": f"cross-{model_id}",
                "model": {
                    "id": model_id,
                    "variant": variants[model_id],
                    "options": {model_id: options},
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_fingerprint": None,
                    "upstream_code_ref": f"fixture:{model_id}",
                    "upstream_commit": commits[model_id],
                },
                "dataset": {
                    "id": "osie",
                    "version": "cross-model-fixture",
                    "split": "test",
                    "root": None,
                    "root_fingerprint": "fixture",
                    "items": [
                        {
                            "item_id": "shared-image",
                            "image_ref": str(image),
                            "image_width": 100,
                            "image_height": 80,
                            "task_text": task_text,
                            "target_description": target_description,
                            "observer_id": observer_id,
                            "observer_metadata": observer_metadata,
                        }
                    ],
                    "item_subset": None,
                },
                "request": {
                    "task_text": None,
                    "target_description": None,
                    "observer_id": None,
                    "observer_metadata": {},
                    "num_samples": 2,
                    "base_seed": 11,
                    "seed_strategy": "fixed_per_item",
                    "max_fixations": max_fixations,
                    "time_horizon_s": time_horizon,
                },
                "device": "cpu",
                "output_root": str(root / "runs"),
                "run_policy": "create",
                "tags": ["cross-model-contract", model_id],
                "notes": "Shared canonical contract CPU regression",
                "environment": {"name": "fake", "image_digest": None},
            }
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()

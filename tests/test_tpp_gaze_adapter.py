
from __future__ import annotations

import json
import os
import shlex
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from activevision_workbench import (
    AdapterOutputError,
    ContractValidationError,
    InferenceRequest,
    JsonDatasetAdapter,
    RunConfig,
    RunEngine,
    StoppingReason,
    TPPGazeAdapter,
    TPPGazeAdapterConfig,
    TPPGazeWorkerError,
    create_tpp_gaze_registry,
    tpp_gaze_run_item_from_dataset_item,
)
from activevision_workbench.adapters.tpp_gaze import (
    TPP_GAZE_NATIVE_ARTIFACT_POLICY,
    TPP_GAZE_TRANSFORMER_CHECKPOINT_SHA256,
    TPP_GAZE_TRANSFORMER_VARIANT,
    TPP_GAZE_UPSTREAM_COMMIT,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TINY_GAZE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "datasets" / "tiny_gaze"


class TPPGazeAdapterTest(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.checkpoint = self.root / "model_transformer.pth"
        self.checkpoint.write_bytes(b"fake-tpp-checkpoint")
        self.model_config = self.root / "config.yaml"
        self.model_config.write_text("context: transformer\n", encoding="utf-8")
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"fake-image")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(
        self,
        *,
        behavior: str = "normal",
        timeout: float = 5.0,
        safety_max: int = 8,
        checkpoint: Path | None = None,
    ) -> TPPGazeAdapterConfig:
        return TPPGazeAdapterConfig.for_fake(
            upstream_root=self.upstream,
            checkpoint_path=self.checkpoint if checkpoint is None else checkpoint,
            model_config_path=self.model_config,
            output_root=self.root / "runs",
            safety_max_fixations=safety_max,
            worker_timeout_s=timeout,
            fake_behavior=behavior,
        )

    def request(
        self,
        *,
        num_samples: int = 3,
        base_seed: int = 8,
        max_fixations: int | None = 8,
        time_horizon_s: float | None = 1.0,
        task_text: str | None = None,
    ) -> InferenceRequest:
        return InferenceRequest.create(
            model_id="tpp_gaze",
            model_variant=TPP_GAZE_TRANSFORMER_VARIANT,
            dataset_id="fixture",
            dataset_split="test",
            item_id="image-1",
            image_ref=str(self.image),
            image_width=100,
            image_height=80,
            task_text=task_text,
            num_samples=num_samples,
            base_seed=base_seed,
            max_fixations=max_fixations,
            time_horizon_s=time_horizon_s,
        )

    def test_multiple_simulations_stay_separate_and_variable_length(self) -> None:
        adapter = TPPGazeAdapter(self.config())
        try:
            request = self.request()
            records = adapter.predict(request, run_id="three")
        finally:
            adapter.close()
        self.assertEqual(len(records), 3)
        self.assertEqual([record.sample_index for record in records], [0, 1, 2])
        self.assertEqual([record.seed for record in records], [8, 9, 10])
        self.assertEqual([len(record.fixations) for record in records], [1, 2, 3])
        self.assertEqual(
            [record.sample_id for record in records],
            [f"{request.request_id}:sample:{index}" for index in range(3)],
        )

    def test_millisecond_inter_event_values_are_explicitly_converted(self) -> None:
        adapter = TPPGazeAdapter(self.config())
        try:
            record = adapter.predict(
                self.request(num_samples=1, base_seed=0), run_id="conversion"
            )[0]
        finally:
            adapter.close()
        fixation = record.fixations[0]
        self.assertAlmostEqual(fixation.duration_s, 0.1)
        self.assertAlmostEqual(fixation.timestamp_s, 0.1)
        timing = fixation.native_timing
        self.assertEqual(timing["inter_event_time"]["value"], 100.0)
        self.assertEqual(timing["inter_event_time"]["unit"], "milliseconds")
        self.assertEqual(timing["inter_event_time"]["scale_to_seconds"], 0.001)

    def test_absolute_time_without_inter_event_does_not_fabricate_duration(self) -> None:
        adapter = TPPGazeAdapter(self.config(behavior="absolute_only"))
        try:
            fixation = adapter.predict(
                self.request(num_samples=1), run_id="absolute-only"
            )[0].fixations[0]
        finally:
            adapter.close()
        self.assertAlmostEqual(fixation.timestamp_s, 0.1)
        self.assertIsNone(fixation.duration_s)
        self.assertIsNone(
            fixation.native_timing["inter_event_time"]["canonical_field"]
        )

    def test_negative_inter_event_time_is_rejected(self) -> None:
        adapter = TPPGazeAdapter(self.config(behavior="negative_time"))
        try:
            with self.assertRaisesRegex(AdapterOutputError, "negative inter-event"):
                adapter.predict(self.request(num_samples=1), run_id="negative")
        finally:
            adapter.close()

    def test_non_monotonic_absolute_time_is_rejected(self) -> None:
        adapter = TPPGazeAdapter(self.config(behavior="decreasing_time"))
        try:
            with self.assertRaisesRegex(AdapterOutputError, "strictly increasing"):
                adapter.predict(self.request(num_samples=1), run_id="decreasing")
        finally:
            adapter.close()

    def test_horizon_termination_is_traceable(self) -> None:
        adapter = TPPGazeAdapter(self.config())
        try:
            record = adapter.predict(
                self.request(
                    num_samples=1,
                    base_seed=0,
                    time_horizon_s=0.1,
                ),
                run_id="horizon",
            )[0]
        finally:
            adapter.close()
        self.assertEqual(record.stopping_reason, StoppingReason.TIME_HORIZON)
        self.assertEqual(len(record.fixations), 1)
        self.assertLessEqual(record.fixations[-1].timestamp_s, 0.1)

    def test_safety_cap_termination_is_traceable(self) -> None:
        adapter = TPPGazeAdapter(self.config(safety_max=2))
        try:
            record = adapter.predict(
                self.request(
                    num_samples=1,
                    base_seed=3,
                    max_fixations=None,
                    time_horizon_s=10.0,
                ),
                run_id="cap",
            )[0]
        finally:
            adapter.close()
        self.assertEqual(record.stopping_reason, StoppingReason.MAX_FIXATIONS)
        self.assertEqual(len(record.fixations), 2)
        self.assertEqual(
            record.native_metadata["tpp_gaze"]["effective_max_fixations"], 2
        )

    def test_empty_sequence_has_no_fake_fixation_and_preserves_terminal_event(
        self,
    ) -> None:
        adapter = TPPGazeAdapter(self.config(behavior="empty"))
        try:
            record = adapter.predict(
                self.request(num_samples=1), run_id="empty"
            )[0]
        finally:
            adapter.close()
        self.assertEqual(record.fixations, ())
        self.assertEqual(record.stopping_reason, StoppingReason.TIME_HORIZON)
        self.assertTrue(any("no fixation was fabricated" in w for w in record.warnings))
        native_path = Path(urlparse(record.native_artifacts[0].uri).path)
        native = json.loads(native_path.read_text(encoding="utf-8"))
        self.assertEqual(len(native["sample"]["events"]), 1)
        self.assertFalse(native["sample"]["events"][0]["included"])

    def test_malformed_temporal_unit_and_conversion_are_rejected(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "native_time_unit"):
            replace(self.config(), native_time_unit="frames")
        with self.assertRaisesRegex(ContractValidationError, "scale_to_seconds"):
            replace(self.config(), temporal_scale_to_seconds=0.0)
        with self.assertRaisesRegex(ContractValidationError, "must be 0.001"):
            replace(self.config(), temporal_scale_to_seconds=1.0)

    def test_official_temperature_default_and_override_are_traceable(self) -> None:
        default_adapter = TPPGazeAdapter(self.config())
        try:
            default_record = default_adapter.predict(
                self.request(num_samples=1), run_id="temperature-default"
            )[0]
        finally:
            default_adapter.close()
        default_metadata = default_record.native_metadata["tpp_gaze"]
        self.assertEqual(default_metadata["temperature"], 2.0)
        self.assertEqual(default_metadata["temperature_source"], "model_config")

        override_adapter = TPPGazeAdapter(
            replace(self.config(), temperature=1.25)
        )
        try:
            override_record = override_adapter.predict(
                self.request(num_samples=1), run_id="temperature-override"
            )[0]
        finally:
            override_adapter.close()
        override_metadata = override_record.native_metadata["tpp_gaze"]
        self.assertEqual(override_metadata["temperature"], 1.25)
        self.assertEqual(
            override_metadata["temperature_source"], "adapter_override"
        )

    def test_invalid_temperature_override_is_rejected(self) -> None:
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ContractValidationError, "positive finite"
                ):
                    replace(self.config(), temperature=value)

    def test_native_artifact_preserves_every_sample_and_temporal_semantics(
        self,
    ) -> None:
        adapter = TPPGazeAdapter(self.config())
        try:
            records = adapter.predict(
                self.request(num_samples=2), run_id="native"
            )
        finally:
            adapter.close()
        paths = []
        for record in records:
            artifact = record.native_artifacts[0]
            path = Path(urlparse(artifact.uri).path)
            native = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(native["sample"]["sample_id"], record.sample_id)
            self.assertEqual(
                native["native_artifact_policy"], TPP_GAZE_NATIVE_ARTIFACT_POLICY
            )
            self.assertEqual(
                native["temporal_semantics"]["duration"], "inter_event_time"
            )
            paths.append(path)
        self.assertEqual(len(set(paths)), 2)

    def test_unsupported_task_and_missing_horizon_are_rejected(self) -> None:
        adapter = TPPGazeAdapter(self.config())
        try:
            with self.assertRaisesRegex(ContractValidationError, "task_text"):
                adapter.validate_request(self.request(task_text="find car"))
            with self.assertRaisesRegex(ContractValidationError, "time_horizon_s"):
                adapter.validate_request(self.request(time_horizon_s=None))
        finally:
            adapter.close()

    def test_missing_assets_and_worker_failures_are_actionable(self) -> None:
        with self.assertRaisesRegex(TPPGazeWorkerError, "checkpoint_path"):
            TPPGazeAdapter(
                self.config(checkpoint=self.root / "missing.pth")
            ).prepare()
        malformed = TPPGazeAdapter(self.config(behavior="malformed"))
        try:
            with self.assertRaisesRegex(AdapterOutputError, "missing field"):
                malformed.predict(self.request(num_samples=1), run_id="malformed")
        finally:
            malformed.close()
        timeout = TPPGazeAdapter(self.config(behavior="timeout", timeout=0.1))
        try:
            with self.assertRaisesRegex(TPPGazeWorkerError, "timed out"):
                timeout.predict(self.request(num_samples=1), run_id="timeout")
            self.assertIsNone(timeout._process)
        finally:
            timeout.close()

    def test_registry_and_dataset_bridge_are_import_light_and_anonymous(self) -> None:
        registry = create_tpp_gaze_registry(self.config())
        self.assertEqual(registry.model_ids, ("tpp_gaze",))
        self.assertEqual(registry.variants("tpp_gaze"), ("transformer",))
        self.assertFalse(registry.describe("tpp_gaze").optional_dependencies)
        item = JsonDatasetAdapter(TINY_GAZE_ROOT).get_item("scene-a", split="train")
        run_item = tpp_gaze_run_item_from_dataset_item(item)
        self.assertEqual(run_item["item_id"], item.item_id)
        self.assertIsNone(run_item["task_text"])
        self.assertIsNone(run_item["target_description"])
        self.assertIsNone(run_item["observer_id"])

    def test_resume_does_not_duplicate_samples(self) -> None:
        run_config = self._run_config()
        engine = RunEngine(
            create_tpp_gaze_registry(self.config()),
            repository_root=self.root,
            console=None,
        )
        first = engine.run(run_config)
        resumed = engine.run(run_config, resume=True)
        records = [
            json.loads(line)
            for line in (first.run_directory / "predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(records), 3)
        self.assertEqual(resumed.succeeded, 1)

    def _run_config(self) -> RunConfig:
        return RunConfig.from_dict(
            {
                "schema_version": 1,
                "run_id": "tpp-gaze-resume",
                "model": {
                    "id": "tpp_gaze",
                    "variant": "transformer",
                    "options": {"tpp_gaze": {}},
                    "checkpoint_path": str(self.checkpoint),
                    "checkpoint_fingerprint": None,
                    "upstream_code_ref": "https://github.com/phuselab/tppgaze.git",
                    "upstream_commit": TPP_GAZE_UPSTREAM_COMMIT,
                },
                "dataset": {
                    "id": "fixture",
                    "version": "1",
                    "split": "test",
                    "root": None,
                    "root_fingerprint": "fixture-v1",
                    "items": [
                        {
                            "item_id": "image-1",
                            "image_ref": str(self.image),
                            "image_width": 100,
                            "image_height": 80,
                            "task_text": None,
                            "target_description": None,
                            "observer_id": None,
                            "observer_metadata": {},
                        }
                    ],
                    "item_subset": None,
                },
                "request": {
                    "task_text": None,
                    "target_description": None,
                    "observer_id": None,
                    "observer_metadata": {},
                    "num_samples": 3,
                    "base_seed": 8,
                    "seed_strategy": "fixed_per_item",
                    "max_fixations": 8,
                    "time_horizon_s": 1.0,
                },
                "device": "cpu",
                "output_root": str(self.root / "runs"),
                "run_policy": "create",
                "tags": ["fake-tpp-gaze"],
                "notes": "CPU fake-worker integration test",
                "environment": {"name": "fake", "image_digest": None},
            }
        )


_REAL_VARIABLES = (
    "AVRW_TPP_GAZE_UPSTREAM_ROOT",
    "AVRW_TPP_GAZE_CHECKPOINT",
    "AVRW_TPP_GAZE_CONFIG",
    "AVRW_TPP_GAZE_IMAGE",
    "AVRW_TPP_GAZE_IMAGE_WIDTH",
    "AVRW_TPP_GAZE_IMAGE_HEIGHT",
    "AVRW_TPP_GAZE_LAUNCHER",
)


@unittest.skipUnless(
    all(os.environ.get(name) for name in _REAL_VARIABLES),
    "set all AVRW_TPP_GAZE_* smoke variables to run the real model",
)
class RealTPPGazeSmokeTest(unittest.TestCase):

    def test_real_transformer_sample(self) -> None:
        output = os.environ.get("AVRW_TPP_GAZE_OUTPUT_ROOT")
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if output is None:
            temporary = tempfile.TemporaryDirectory()
            output = temporary.name
        config = TPPGazeAdapterConfig(
            variant=TPP_GAZE_TRANSFORMER_VARIANT,
            upstream_root=Path(
                os.environ["AVRW_TPP_GAZE_UPSTREAM_ROOT"]
            ).resolve(),
            upstream_commit=TPP_GAZE_UPSTREAM_COMMIT,
            checkpoint_path=Path(os.environ["AVRW_TPP_GAZE_CHECKPOINT"]).resolve(),
            checkpoint_fingerprint=(
                f"sha256:{TPP_GAZE_TRANSFORMER_CHECKPOINT_SHA256}"
            ),
            model_config_path=Path(os.environ["AVRW_TPP_GAZE_CONFIG"]).resolve(),
            device=os.environ.get("AVRW_TPP_GAZE_DEVICE", "cuda"),
            output_root=Path(output).resolve(),
            launcher=tuple(shlex.split(os.environ["AVRW_TPP_GAZE_LAUNCHER"])),
            safety_max_fixations=64,
        )
        adapter = TPPGazeAdapter(config)
        try:
            request = InferenceRequest.create(
                model_id="tpp_gaze",
                model_variant="transformer",
                dataset_id=os.environ.get("AVRW_TPP_GAZE_DATASET_ID", "real-smoke"),
                dataset_split="smoke",
                item_id="one-image",
                image_ref=os.environ["AVRW_TPP_GAZE_IMAGE"],
                image_width=int(os.environ["AVRW_TPP_GAZE_IMAGE_WIDTH"]),
                image_height=int(os.environ["AVRW_TPP_GAZE_IMAGE_HEIGHT"]),
                num_samples=2,
                base_seed=17,
                max_fixations=64,
                time_horizon_s=0.5,
            )
            records = adapter.predict(request, run_id="real-tpp-gaze-smoke")
        finally:
            adapter.close()
            if temporary is not None:
                temporary.cleanup()
        self.assertEqual(len(records), 2)
        self.assertEqual([record.seed for record in records], [17, 18])
        for record in records:
            self.assertTrue(record.native_artifacts)
            for fixation in record.fixations:
                self.assertLessEqual(fixation.timestamp_s, 0.5)


if __name__ == "__main__":
    unittest.main()

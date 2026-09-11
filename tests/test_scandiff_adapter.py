
from __future__ import annotations

import json
import os
import shlex
import subprocess
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
    ScanDiffAdapter,
    ScanDiffAdapterConfig,
    ScanDiffWorkerError,
    create_scandiff_registry,
    scandiff_run_item_from_dataset_item,
)
from activevision_workbench.adapters.scandiff import (
    SCANDIFF_FREE_VIEWING_VARIANT,
    SCANDIFF_NATIVE_ARTIFACT_POLICY,
    SCANDIFF_UPSTREAM_COMMIT,
    SCANDIFF_VISUAL_SEARCH_VARIANT,
)
from activevision_workbench.adapters.scandiff.adapter import _upstream_commit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TINY_GAZE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "datasets" / "tiny_gaze"


class ScanDiffAdapterTest(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.checkpoint = self.root / "scandiff.pth"
        self.checkpoint.write_bytes(b"fake-checkpoint")
        self.embeddings = self.root / "task_embeddings.npy"
        self.embeddings.write_bytes(b"fake-embeddings")
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"fake-image")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(
        self,
        *,
        variant: str = SCANDIFF_FREE_VIEWING_VARIANT,
        behavior: str = "normal",
        checkpoint: Path | None = None,
        feature_root: Path | None = None,
        timeout: float = 5.0,
    ) -> ScanDiffAdapterConfig:
        return ScanDiffAdapterConfig.for_fake(
            variant=variant,
            upstream_root=self.upstream,
            checkpoint_path=self.checkpoint if checkpoint is None else checkpoint,
            task_embeddings_path=self.embeddings,
            output_root=self.root / "runs",
            feature_root=feature_root,
            worker_timeout_s=timeout,
            fake_behavior=behavior,
        )

    def request(
        self,
        *,
        num_samples: int = 3,
        base_seed: int = 8,
        variant: str = SCANDIFF_FREE_VIEWING_VARIANT,
        task_text: str | None = None,
        target_description: str | None = None,
    ) -> InferenceRequest:
        return InferenceRequest.create(
            model_id="scandiff",
            model_variant=variant,
            dataset_id="fixture",
            dataset_split="test",
            item_id="image-1",
            image_ref=str(self.image),
            image_width=100,
            image_height=80,
            task_text=task_text,
            target_description=target_description,
            num_samples=num_samples,
            base_seed=base_seed,
            max_fixations=16,
        )

    def test_three_samples_keep_stable_ids_seeds_and_variable_lengths(self) -> None:
        adapter = ScanDiffAdapter(self.config())
        try:
            request = self.request()
            records = adapter.predict(request, run_id="fake-three")
        finally:
            adapter.close()
        self.assertEqual(len(records), 3)
        self.assertEqual([record.sample_index for record in records], [0, 1, 2])
        self.assertEqual([record.seed for record in records], [8, 9, 10])
        self.assertEqual(
            [record.sample_id for record in records],
            [f"{request.request_id}:sample:{index}" for index in range(3)],
        )
        self.assertEqual([len(record.fixations) for record in records], [1, 2, 3])

    def test_same_fake_seed_is_deterministic(self) -> None:
        request = self.request(num_samples=2, base_seed=41)
        first_adapter = ScanDiffAdapter(self.config())
        second_adapter = ScanDiffAdapter(self.config())
        try:
            first = first_adapter.predict(request, run_id="deterministic-a")
            second = second_adapter.predict(request, run_id="deterministic-b")
        finally:
            first_adapter.close()
            second_adapter.close()
        self.assertEqual(
            [record.fixations for record in first],
            [record.fixations for record in second],
        )
        self.assertEqual(
            [record.sample_id for record in first],
            [record.sample_id for record in second],
        )

    def test_malformed_worker_response_is_rejected(self) -> None:
        adapter = ScanDiffAdapter(self.config(behavior="malformed"))
        try:
            with self.assertRaisesRegex(AdapterOutputError, "missing field"):
                adapter.predict(self.request(num_samples=1), run_id="malformed")
        finally:
            adapter.close()

    def test_worker_timeout_terminates_process(self) -> None:
        adapter = ScanDiffAdapter(self.config(behavior="timeout", timeout=0.1))
        try:
            with self.assertRaisesRegex(ScanDiffWorkerError, "timed out"):
                adapter.predict(self.request(num_samples=1), run_id="timeout")
            self.assertIsNone(adapter._process)  # worker was cleaned up, not orphaned
        finally:
            adapter.close()

    def test_worker_error_is_actionable(self) -> None:
        adapter = ScanDiffAdapter(self.config(behavior="error"))
        try:
            with self.assertRaisesRegex(
                ScanDiffWorkerError, "deliberate fake ScanDiff worker failure"
            ):
                adapter.predict(self.request(num_samples=1), run_id="worker-error")
        finally:
            adapter.close()

    def test_missing_checkpoint_and_feature_root_fail_clearly(self) -> None:
        missing_checkpoint = self.root / "missing.pth"
        with self.assertRaisesRegex(ScanDiffWorkerError, "checkpoint_path"):
            ScanDiffAdapter(self.config(checkpoint=missing_checkpoint)).prepare()
        missing_features = self.root / "missing-features"
        with self.assertRaisesRegex(ScanDiffWorkerError, "feature_root"):
            ScanDiffAdapter(self.config(feature_root=missing_features)).prepare()

    def test_free_viewing_rejects_unsupported_task_field(self) -> None:
        adapter = ScanDiffAdapter(self.config())
        try:
            with self.assertRaisesRegex(ContractValidationError, "task_text"):
                adapter.validate_request(self.request(task_text="find car"))
        finally:
            adapter.close()

    def test_visual_search_maps_exactly_one_target(self) -> None:
        adapter = ScanDiffAdapter(
            self.config(variant=SCANDIFF_VISUAL_SEARCH_VARIANT)
        )
        try:
            request = self.request(
                num_samples=1,
                variant=SCANDIFF_VISUAL_SEARCH_VARIANT,
                target_description="car",
            )
            record = adapter.predict(request, run_id="visual-search")[0]
        finally:
            adapter.close()
        self.assertEqual(record.target_description, "car")
        self.assertEqual(record.native_metadata["scandiff"]["viewing_task"], "car")

    def test_dataset_bridge_produces_one_image_level_visual_search_request(
        self,
    ) -> None:
        dataset = JsonDatasetAdapter(TINY_GAZE_ROOT)
        item = dataset.get_item("scene-a", split="train")
        run_item = scandiff_run_item_from_dataset_item(
            item,
            variant=SCANDIFF_VISUAL_SEARCH_VARIANT,
        )
        self.assertIsNone(run_item["observer_id"])
        self.assertEqual(run_item["observer_metadata"], {})
        self.assertIsNone(run_item["task_text"])
        self.assertEqual(run_item["target_description"], "circle")

        config = self._run_config(
            run_id="dataset-scandiff-integration",
            variant=SCANDIFF_VISUAL_SEARCH_VARIANT,
            run_item=run_item,
            dataset_id=item.dataset_id,
            dataset_version=item.dataset_version,
            dataset_split=item.official_split,
            dataset_root=str(dataset.data_root),
        )
        engine = RunEngine(
            create_scandiff_registry(
                self.config(variant=SCANDIFF_VISUAL_SEARCH_VARIANT)
            ),
            repository_root=self.root,
            console=None,
        )
        result = engine.run(config)
        records = [
            json.loads(line)
            for line in (
                result.run_directory / "predictions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 3)
        self.assertEqual(
            {record["target_description"] for record in records}, {"circle"}
        )
        self.assertEqual({record["observer_id"] for record in records}, {None})

    def test_native_artifact_reference_preserves_each_sample(self) -> None:
        adapter = ScanDiffAdapter(self.config())
        try:
            records = adapter.predict(self.request(num_samples=2), run_id="native")
        finally:
            adapter.close()
        paths = []
        for record in records:
            self.assertEqual(len(record.native_artifacts), 1)
            artifact = record.native_artifacts[0]
            self.assertEqual(artifact.kind, "model_native_prediction")
            path = Path(urlparse(artifact.uri).path)
            self.assertTrue(path.is_file())
            native = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(native["sample"]["sample_id"], record.sample_id)
            raw_fixation = native["sample"]["fixations"][0]
            canonical = record.fixations[0]
            self.assertAlmostEqual(
                canonical.x_px, raw_fixation["x_norm"] * 100, places=12
            )
            self.assertAlmostEqual(
                canonical.y_px, raw_fixation["y_norm"] * 80, places=12
            )
            if canonical.x_norm is not None:
                self.assertAlmostEqual(
                    canonical.x_norm, canonical.x_px / 99, places=12
                )
            if canonical.y_norm is not None:
                self.assertAlmostEqual(
                    canonical.y_norm, canonical.y_px / 79, places=12
                )
            self.assertEqual(canonical.duration_s, raw_fixation["duration_s"])
            self.assertEqual(
                native["native_artifact_policy"],
                SCANDIFF_NATIVE_ARTIFACT_POLICY,
            )
            paths.append(path)
        self.assertEqual(len(set(paths)), 2)

    def test_worker_is_persistent_within_one_run(self) -> None:
        adapter = ScanDiffAdapter(self.config())
        try:
            adapter.predict(self.request(num_samples=1), run_id="persistent")
            assert adapter._process is not None
            first_pid = adapter._process.pid
            adapter.predict(
                self.request(num_samples=1, base_seed=99), run_id="persistent"
            )
            assert adapter._process is not None
            self.assertEqual(adapter._process.pid, first_pid)
        finally:
            adapter.close()

    def test_registry_discovery_is_import_light(self) -> None:
        registry = create_scandiff_registry(self.config())
        self.assertEqual(registry.model_ids, ("scandiff",))
        self.assertEqual(
            registry.variants("scandiff"), (SCANDIFF_FREE_VIEWING_VARIANT,)
        )
        self.assertFalse(registry.describe("scandiff").optional_dependencies)

    def test_run_resume_does_not_duplicate_samples(self) -> None:
        config = self._run_config()
        engine = RunEngine(
            create_scandiff_registry(self.config()),
            repository_root=self.root,
            console=None,
        )
        first = engine.run(config)
        resume_plan = engine.dry_run(config, resume=True)
        resumed = engine.run(config, resume=True)
        prediction_path = first.run_directory / "predictions.jsonl"
        records = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 3)
        self.assertEqual(resume_plan.skipped_completed, 1)
        self.assertEqual(resumed.succeeded, 1)

    def test_partial_resume_does_not_overwrite_referenced_native_artifact(
        self,
    ) -> None:
        config = self._run_config(run_id="scandiff-partial-resume")
        engine = RunEngine(
            create_scandiff_registry(self.config()),
            repository_root=self.root,
            console=None,
        )
        first = engine.run(config)
        prediction_path = first.run_directory / "predictions.jsonl"
        lines = prediction_path.read_text(encoding="utf-8").splitlines()
        first_record = json.loads(lines[0])
        first_uri = first_record["native_artifacts"][0]["uri"]
        first_path = Path(urlparse(first_uri).path)
        first_bytes = first_path.read_bytes()
        first_checksum = first_record["native_artifacts"][0]["checksum"]

        prediction_path.write_text(f"{lines[0]}\n", encoding="utf-8")
        resumed = engine.run(config, resume=True)

        final_records = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(resumed.succeeded, 1)
        self.assertEqual(len(final_records), 3)
        self.assertEqual(final_records[0], first_record)
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(
            final_records[0]["native_artifacts"][0]["checksum"],
            first_checksum,
        )
        resumed_path = Path(
            urlparse(final_records[1]["native_artifacts"][0]["uri"]).path
        )
        self.assertNotEqual(first_path.parent, resumed_path.parent)

    def test_native_artifact_policy_is_fixed_and_explicit(self) -> None:
        with self.assertRaisesRegex(
            ContractValidationError, "native_artifact_policy"
        ):
            replace(self.config(), native_artifact_policy="discard")

    def test_upstream_checkout_rejects_tracked_changes_only(self) -> None:
        checkout = self.root / "upstream-git"
        checkout.mkdir()
        subprocess.run(
            ["git", "init", "--quiet", str(checkout)],
            check=True,
        )
        tracked = checkout / "demo.py"
        tracked.write_text("print('verified')\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(checkout), "add", "demo.py"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "user.name=AVRW Test",
                "-c",
                "user.email=avrw@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        expected = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        (checkout / "checkpoint.pth").write_bytes(b"untracked-asset")
        self.assertEqual(_upstream_commit(checkout), expected)
        tracked.write_text("print('modified')\n", encoding="utf-8")
        with self.assertRaisesRegex(
            ScanDiffWorkerError, "tracked modifications"
        ):
            _upstream_commit(checkout)

    def _run_config(
        self,
        *,
        run_id: str = "scandiff-resume",
        variant: str = SCANDIFF_FREE_VIEWING_VARIANT,
        run_item: dict[str, object] | None = None,
        dataset_id: str = "fixture",
        dataset_version: str = "1",
        dataset_split: str = "test",
        dataset_root: str | None = None,
    ) -> RunConfig:
        if run_item is None:
            run_item = {
                "item_id": "image-1",
                "image_ref": str(self.image),
                "image_width": 100,
                "image_height": 80,
                "task_text": None,
                "target_description": None,
                "observer_id": None,
                "observer_metadata": {},
            }
        return RunConfig.from_dict(
            {
                "schema_version": 1,
                "run_id": run_id,
                "model": {
                    "id": "scandiff",
                    "variant": variant,
                    "options": {"scandiff": {}},
                    "checkpoint_path": str(self.checkpoint),
                    "checkpoint_fingerprint": None,
                    "upstream_code_ref": "https://github.com/aimagelab/ScanDiff.git",
                    "upstream_commit": SCANDIFF_UPSTREAM_COMMIT,
                },
                "dataset": {
                    "id": dataset_id,
                    "version": dataset_version,
                    "split": dataset_split,
                    "root": dataset_root,
                    "root_fingerprint": "fixture-v1",
                    "items": [run_item],
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
                    "max_fixations": 16,
                    "time_horizon_s": None,
                },
                "device": "cpu",
                "output_root": str(self.root / "runs"),
                "run_policy": "create",
                "tags": ["fake-scandiff"],
                "notes": "CPU fake worker integration test",
                "environment": {"name": "fake", "image_digest": None},
            }
        )


_REAL_VARIABLES = (
    "AVRW_SCANDIFF_UPSTREAM_ROOT",
    "AVRW_SCANDIFF_CHECKPOINT",
    "AVRW_SCANDIFF_TASK_EMBEDDINGS",
    "AVRW_SCANDIFF_IMAGE",
    "AVRW_SCANDIFF_IMAGE_WIDTH",
    "AVRW_SCANDIFF_IMAGE_HEIGHT",
    "AVRW_SCANDIFF_LAUNCHER",
)


@unittest.skipUnless(
    all(os.environ.get(name) for name in _REAL_VARIABLES),
    "set all AVRW_SCANDIFF_* smoke-test variables to run the real GPU model",
)
class RealScanDiffSmokeTest(unittest.TestCase):

    def test_real_free_viewing_sample(self) -> None:
        output = os.environ.get("AVRW_SCANDIFF_OUTPUT_ROOT")
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if output is None:
            temporary = tempfile.TemporaryDirectory()
            output = temporary.name
        config = ScanDiffAdapterConfig(
            variant=SCANDIFF_FREE_VIEWING_VARIANT,
            upstream_root=Path(os.environ["AVRW_SCANDIFF_UPSTREAM_ROOT"]).resolve(),
            upstream_commit=SCANDIFF_UPSTREAM_COMMIT,
            checkpoint_path=Path(os.environ["AVRW_SCANDIFF_CHECKPOINT"]).resolve(),
            checkpoint_fingerprint=os.environ.get(
                "AVRW_SCANDIFF_CHECKPOINT_SHA256"
            ),
            task_embeddings_path=Path(
                os.environ["AVRW_SCANDIFF_TASK_EMBEDDINGS"]
            ).resolve(),
            feature_root=(
                Path(os.environ["AVRW_SCANDIFF_FEATURE_ROOT"]).resolve()
                if os.environ.get("AVRW_SCANDIFF_FEATURE_ROOT")
                else None
            ),
            device="cuda",
            output_root=Path(output).resolve(),
            launcher=tuple(shlex.split(os.environ["AVRW_SCANDIFF_LAUNCHER"])),
        )
        adapter = ScanDiffAdapter(config)
        try:
            request = InferenceRequest.create(
                model_id="scandiff",
                model_variant=SCANDIFF_FREE_VIEWING_VARIANT,
                dataset_id="real-smoke",
                dataset_split="smoke",
                item_id="one-image",
                image_ref=os.environ["AVRW_SCANDIFF_IMAGE"],
                image_width=int(os.environ["AVRW_SCANDIFF_IMAGE_WIDTH"]),
                image_height=int(os.environ["AVRW_SCANDIFF_IMAGE_HEIGHT"]),
                num_samples=1,
                base_seed=0,
                max_fixations=16,
            )
            records = adapter.predict(request, run_id="real-scandiff-smoke")
        finally:
            adapter.close()
            if temporary is not None:
                temporary.cleanup()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].seed, 0)


if __name__ == "__main__":
    unittest.main()

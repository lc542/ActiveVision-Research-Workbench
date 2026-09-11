
from __future__ import annotations

import hashlib
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
    IndividualScanpathAdapter,
    IndividualScanpathAdapterConfig,
    IndividualScanpathWorkerError,
    JsonDatasetAdapter,
    RunConfig,
    RunEngine,
    StoppingReason,
    create_individual_scanpath_registry,
    individual_scanpath_run_items_from_dataset_item,
)
from activevision_workbench.adapters.individual_scanpath import (
    INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_CHECKPOINT_SHA256,
    INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
    INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT,
)

ROOT = Path(__file__).resolve().parents[1]
TINY_GAZE = ROOT / "tests" / "fixtures" / "datasets" / "tiny_gaze"


class IndividualScanpathAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.checkpoint = self.root / "checkpoint_best.pth"
        self.checkpoint.write_bytes(b"fake-individualscanpath-checkpoint")
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"fake-image")
        self.mapping = self.root / "observers.json"
        self._write_mapping(self.mapping)
        self.mapping_fingerprint = f"sha256:{self._sha256(self.mapping)}"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_mapping(self, path: Path) -> None:
        value = {
            "schema_version": 1,
            "dataset_id": "osie",
            "model_variant": INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
            "checkpoint_sha256": (
                f"sha256:{INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_CHECKPOINT_SHA256}"
            ),
            "identity_source": "fixture one-based array position",
            "observers": [
                {
                    "observer_id": f"subject-{index + 1:02d}",
                    "model_observer_index": index,
                    "known_checkpoint_observer": True,
                }
                for index in range(15)
            ],
        }
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def config(
        self,
        *,
        behavior: str = "normal",
        timeout: float = 5.0,
        checkpoint: Path | None = None,
        mapping: Path | None = None,
        mapping_fingerprint: str | None = None,
    ) -> IndividualScanpathAdapterConfig:
        return IndividualScanpathAdapterConfig.for_fake(
            upstream_root=self.upstream,
            checkpoint_path=self.checkpoint if checkpoint is None else checkpoint,
            observer_mapping_path=self.mapping if mapping is None else mapping,
            observer_mapping_fingerprint=(
                self.mapping_fingerprint
                if mapping_fingerprint is None
                else mapping_fingerprint
            ),
            output_root=self.root / "runs",
            worker_timeout_s=timeout,
            fake_behavior=behavior,
        )

    def request(
        self,
        *,
        observer_id: str | None = "subject-01",
        item_id: str = "1001:subject-01",
        num_samples: int = 2,
        base_seed: int = 5,
        max_fixations: int | None = 6,
        dataset_id: str = "osie",
    ) -> InferenceRequest:
        return InferenceRequest.create(
            model_id="individualscanpath",
            model_variant=INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
            dataset_id=dataset_id,
            dataset_split="test",
            item_id=item_id,
            image_ref=str(self.image),
            image_width=800,
            image_height=600,
            observer_id=observer_id,
            observer_metadata={"source_subject_index": 0},
            num_samples=num_samples,
            base_seed=base_seed,
            max_fixations=max_fixations,
        )

    def test_observer_id_is_required_and_unknown_errors(self) -> None:
        adapter = IndividualScanpathAdapter(self.config())
        try:
            with self.assertRaisesRegex(ContractValidationError, "observer_id"):
                adapter.validate_request(self.request(observer_id=None))
            with self.assertRaisesRegex(ContractValidationError, "unknown checkpoint"):
                adapter.validate_request(self.request(observer_id="subject-99"))
            with self.assertRaisesRegex(ContractValidationError, "dataset_id"):
                adapter.validate_request(self.request(dataset_id="other"))
        finally:
            adapter.close()

    def test_unsupported_unknown_observer_fallback_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "no unknown token"):
            replace(self.config(), unknown_observer_policy="mean_embedding")

    def test_same_image_two_observers_are_distinct_and_cache_safe(self) -> None:
        adapter = IndividualScanpathAdapter(self.config())
        try:
            first = adapter.predict(
                self.request(observer_id="subject-01", num_samples=1),
                run_id="two-observers",
            )[0]
            second = adapter.predict(
                self.request(
                    observer_id="subject-02",
                    item_id="1001:subject-02",
                    num_samples=1,
                ),
                run_id="two-observers",
            )[0]
        finally:
            adapter.close()
        self.assertEqual(first.observer_id, "subject-01")
        self.assertEqual(second.observer_id, "subject-02")
        first_meta = first.native_metadata["individualscanpath"]
        second_meta = second.native_metadata["individualscanpath"]
        self.assertEqual(first_meta["model_observer_index"], 0)
        self.assertEqual(second_meta["model_observer_index"], 1)
        self.assertNotEqual(first.fixations, second.fixations)
        self.assertIn("no observer-conditioned outputs", first_meta["image_cache_scope"])

    def test_samples_keep_ids_seeds_observer_mapping_and_variable_lengths(self) -> None:
        adapter = IndividualScanpathAdapter(self.config())
        try:
            request = self.request(num_samples=3, base_seed=5)
            records = adapter.predict(request, run_id="samples")
        finally:
            adapter.close()
        self.assertEqual([record.sample_index for record in records], [0, 1, 2])
        self.assertEqual([record.seed for record in records], [5, 6, 7])
        self.assertEqual([len(record.fixations) for record in records], [2, 3, 4])
        self.assertEqual(
            [record.sample_id for record in records],
            [f"{request.request_id}:sample:{index}" for index in range(3)],
        )
        for record in records:
            metadata = record.native_metadata["individualscanpath"]
            self.assertEqual(metadata["observer_id"], "subject-01")
            self.assertEqual(metadata["model_observer_index"], 0)
            self.assertEqual(
                f"sha256:{metadata['observer_mapping_sha256']}",
                self.mapping_fingerprint,
            )

    def test_duration_coordinates_stop_and_native_artifact_are_preserved(self) -> None:
        adapter = IndividualScanpathAdapter(self.config())
        try:
            record = adapter.predict(
                self.request(num_samples=1, base_seed=1, max_fixations=2),
                run_id="native",
            )[0]
        finally:
            adapter.close()
        self.assertEqual(record.stopping_reason, StoppingReason.MAX_FIXATIONS)
        self.assertEqual(len(record.fixations), 2)
        for fixation in record.fixations:
            self.assertIsNotNone(fixation.duration_s)
            self.assertIsNone(fixation.timestamp_s)
            self.assertEqual(fixation.native_timing["unit"], "seconds")
        artifact = record.native_artifacts[0]
        native = json.loads(Path(urlparse(artifact.uri).path).read_text())
        self.assertEqual(native["sample"]["observer_id"], "subject-01")
        self.assertIn("selected_actions", native["sample"]["native_sequence"])

    def test_missing_and_malformed_mapping_are_actionable(self) -> None:
        missing = IndividualScanpathAdapter(
            self.config(
                mapping=self.root / "missing.json",
                mapping_fingerprint=f"sha256:{'0' * 64}",
            )
        )
        try:
            with self.assertRaisesRegex(ContractValidationError, "does not exist"):
                missing.validate_request(self.request())
        finally:
            missing.close()
        malformed_path = self.root / "malformed.json"
        malformed_path.write_text("{}", encoding="utf-8")
        malformed = IndividualScanpathAdapter(
            self.config(
                mapping=malformed_path,
                mapping_fingerprint=f"sha256:{self._sha256(malformed_path)}",
            )
        )
        try:
            with self.assertRaisesRegex(ContractValidationError, "missing field"):
                malformed.validate_request(self.request())
        finally:
            malformed.close()

    def test_missing_checkpoint_malformed_response_and_timeout(self) -> None:
        missing = IndividualScanpathAdapter(
            self.config(checkpoint=self.root / "missing.pth")
        )
        try:
            with self.assertRaisesRegex(IndividualScanpathWorkerError, "checkpoint_path"):
                missing.predict(self.request(num_samples=1), run_id="missing")
        finally:
            missing.close()
        malformed = IndividualScanpathAdapter(self.config(behavior="malformed"))
        try:
            with self.assertRaisesRegex(AdapterOutputError, "missing field"):
                malformed.predict(self.request(num_samples=1), run_id="malformed")
        finally:
            malformed.close()
        timeout = IndividualScanpathAdapter(
            self.config(behavior="timeout", timeout=0.1)
        )
        try:
            with self.assertRaisesRegex(IndividualScanpathWorkerError, "timed out"):
                timeout.predict(self.request(num_samples=1), run_id="timeout")
        finally:
            timeout.close()

    def test_dataset_bridge_creates_one_item_per_observer(self) -> None:
        item = JsonDatasetAdapter(TINY_GAZE).get_item("scene-a", split="train")
        records = individual_scanpath_run_items_from_dataset_item(
            item, observer_ids=("observer-a", "observer-b")
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["image_ref"], records[1]["image_ref"])
        self.assertEqual(records[0]["observer_id"], "observer-a")
        self.assertEqual(records[1]["observer_id"], "observer-b")
        self.assertNotEqual(records[0]["item_id"], records[1]["item_id"])

    def test_registry_is_import_light_and_resume_keeps_observers_separate(self) -> None:
        registry = create_individual_scanpath_registry(self.config())
        self.assertEqual(registry.model_ids, ("individualscanpath",))
        self.assertTrue(registry.describe("individualscanpath").requirements.observer)
        engine = RunEngine(registry, repository_root=self.root, console=None)
        config = self._run_config()
        first = engine.run(config)
        resumed = engine.run(config, resume=True)
        lines = first.run_directory.joinpath("predictions.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines]
        self.assertEqual(len(records), 4)
        self.assertEqual(
            {record["observer_id"] for record in records},
            {"subject-01", "subject-02"},
        )
        manifest = json.loads(
            first.run_directory.joinpath("manifest.json").read_text()
        )
        resolved = manifest["resolved_configuration"]
        self.assertEqual(
            resolved["model"]["options"]["individualscanpath"][
                "observer_mapping_fingerprint"
            ],
            self.mapping_fingerprint,
        )
        self.assertEqual(
            {
                item["observer_id"]
                for item in resolved["dataset"]["items"]
            },
            {"subject-01", "subject-02"},
        )
        self.assertEqual(resumed.succeeded, 2)

    def _run_config(self) -> RunConfig:
        return RunConfig.from_dict(
            {
                "schema_version": 1,
                "run_id": "individualscanpath-resume",
                "model": {
                    "id": "individualscanpath",
                    "variant": INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
                    "options": {
                        "individualscanpath": {
                            "upstream_root": str(self.upstream),
                            "observer_mapping_path": str(self.mapping),
                            "observer_mapping_fingerprint": (
                                self.mapping_fingerprint
                            ),
                            "unknown_observer_policy": "error",
                        }
                    },
                    "checkpoint_path": str(self.checkpoint),
                    "checkpoint_fingerprint": None,
                    "upstream_code_ref": "https://github.com/chenxy99/IndividualScanpath.git",
                    "upstream_commit": INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT,
                },
                "dataset": {
                    "id": "osie",
                    "version": "fixture",
                    "split": "test",
                    "root": None,
                    "root_fingerprint": "fixture",
                    "items": [
                        {
                            "item_id": f"1001:subject-{index:02d}",
                            "image_ref": str(self.image),
                            "image_width": 800,
                            "image_height": 600,
                            "task_text": None,
                            "target_description": None,
                            "observer_id": f"subject-{index:02d}",
                            "observer_metadata": {"source_subject_index": index - 1},
                        }
                        for index in (1, 2)
                    ],
                    "item_subset": None,
                },
                "request": {
                    "task_text": None,
                    "target_description": None,
                    "observer_id": None,
                    "observer_metadata": {},
                    "num_samples": 2,
                    "base_seed": 5,
                    "seed_strategy": "fixed_per_item",
                    "max_fixations": 4,
                    "time_horizon_s": None,
                },
                "device": "cpu",
                "output_root": str(self.root / "runs"),
                "run_policy": "create",
                "tags": ["fake-individualscanpath"],
                "notes": "observer resume test",
                "environment": {"name": "fake", "image_digest": None},
            }
        )


_REAL_ENV = (
    "AVRW_INDIVIDUAL_SCANPATH_UPSTREAM_ROOT",
    "AVRW_INDIVIDUAL_SCANPATH_CHECKPOINT",
    "AVRW_INDIVIDUAL_SCANPATH_MAPPING",
    "AVRW_INDIVIDUAL_SCANPATH_MAPPING_FINGERPRINT",
    "AVRW_INDIVIDUAL_SCANPATH_IMAGE",
    "AVRW_INDIVIDUAL_SCANPATH_IMAGE_WIDTH",
    "AVRW_INDIVIDUAL_SCANPATH_IMAGE_HEIGHT",
    "AVRW_INDIVIDUAL_SCANPATH_LAUNCHER",
)


@unittest.skipUnless(
    all(os.environ.get(name) for name in _REAL_ENV),
    "set all AVRW_INDIVIDUAL_SCANPATH_* variables for the real GPU smoke",
)
class RealIndividualScanpathSmokeTest(unittest.TestCase):
    def test_real_same_image_two_observers(self) -> None:
        output = os.environ.get("AVRW_INDIVIDUAL_SCANPATH_OUTPUT_ROOT")
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if output is None:
            temporary = tempfile.TemporaryDirectory()
            output = temporary.name
        config = IndividualScanpathAdapterConfig(
            variant=INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
            upstream_root=Path(os.environ[_REAL_ENV[0]]).resolve(),
            upstream_commit=INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT,
            checkpoint_path=Path(os.environ[_REAL_ENV[1]]).resolve(),
            checkpoint_fingerprint=(
                f"sha256:{INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_CHECKPOINT_SHA256}"
            ),
            observer_mapping_path=Path(os.environ[_REAL_ENV[2]]).resolve(),
            observer_mapping_fingerprint=os.environ[_REAL_ENV[3]],
            unknown_observer_policy="error",
            device="cuda",
            output_root=Path(output).resolve(),
            launcher=tuple(shlex.split(os.environ[_REAL_ENV[7]])),
        )
        adapter = IndividualScanpathAdapter(config)
        try:
            records = []
            for observer_index in (1, 2):
                records.append(
                    adapter.predict(
                        InferenceRequest.create(
                            model_id="individualscanpath",
                            model_variant=INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
                            dataset_id="osie",
                            dataset_split="smoke",
                            item_id=f"real:subject-{observer_index:02d}",
                            image_ref=os.environ[_REAL_ENV[4]],
                            image_width=int(os.environ[_REAL_ENV[5]]),
                            image_height=int(os.environ[_REAL_ENV[6]]),
                            observer_id=f"subject-{observer_index:02d}",
                            num_samples=1,
                            base_seed=17,
                            max_fixations=8,
                        ),
                        run_id="real-individualscanpath-smoke",
                    )[0]
                )
        finally:
            adapter.close()
            if temporary is not None:
                temporary.cleanup()
        self.assertEqual([record.observer_id for record in records], ["subject-01", "subject-02"])
        self.assertEqual(
            [record.native_metadata["individualscanpath"]["model_observer_index"] for record in records],
            [0, 1],
        )
        self.assertTrue(all(record.native_artifacts for record in records))
        self.assertNotEqual(records[0].fixations, records[1].fixations)
        self.assertNotEqual(
            records[0].native_artifacts[0].uri,
            records[1].native_artifacts[0].uri,
        )


if __name__ == "__main__":
    unittest.main()

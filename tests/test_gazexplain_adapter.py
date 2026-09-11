
from __future__ import annotations

import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlparse

from activevision_workbench import (
    AdapterOutputError,
    ContractValidationError,
    GAZEXPLAIN_JOINT_CHECKPOINT_SHA256,
    GAZEXPLAIN_JOINT_HPARAMS_SHA256,
    GAZEXPLAIN_JOINT_VARIANT,
    GAZEXPLAIN_UPSTREAM_COMMIT,
    GazeXplainAdapter,
    GazeXplainAdapterConfig,
    GazeXplainWorkerError,
    InferenceRequest,
    JsonDatasetAdapter,
    RunConfig,
    RunEngine,
    create_gazexplain_registry,
    gazexplain_run_item_from_dataset_item,
)

ROOT = Path(__file__).resolve().parents[1]
TINY_GAZE = ROOT / "tests" / "fixtures" / "datasets" / "tiny_gaze"


class GazeXplainAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.checkpoint = self.root / "model.safetensors"
        self.checkpoint.write_bytes(b"fake-gazexplain-checkpoint")
        self.hparams = self.root / "hparams.json"
        self.hparams.write_text("{}\n", encoding="utf-8")
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"fake-image")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(
        self,
        *,
        behavior: str = "normal",
        timeout: float = 5.0,
        checkpoint: Path | None = None,
    ) -> GazeXplainAdapterConfig:
        return GazeXplainAdapterConfig.for_fake(
            upstream_root=self.upstream,
            checkpoint_path=(
                self.checkpoint if checkpoint is None else checkpoint
            ),
            hparams_path=self.hparams,
            output_root=self.root / "runs",
            worker_timeout_s=timeout,
            fake_behavior=behavior,
        )

    def request(
        self,
        *,
        task_text: str | None = "Question: What do you see in the image?",
        target_description: str | None = None,
        item_id: str = "image-1",
        num_samples: int = 2,
        base_seed: int = 7,
        max_fixations: int | None = 6,
        observer_id: str | None = None,
        observer_metadata: object | None = None,
        model_options: object | None = None,
    ) -> InferenceRequest:
        return InferenceRequest.create(
            model_id="gazexplain",
            model_variant=GAZEXPLAIN_JOINT_VARIANT,
            dataset_id="osie",
            dataset_split="test",
            item_id=item_id,
            image_ref=str(self.image),
            image_width=800,
            image_height=600,
            task_text=task_text,
            target_description=target_description,
            observer_id=observer_id,
            observer_metadata=observer_metadata,
            num_samples=num_samples,
            base_seed=base_seed,
            max_fixations=max_fixations,
            model_options=model_options,
        )

    def test_task_is_required_and_unsupported_prompt_fields_are_rejected(self) -> None:
        adapter = GazeXplainAdapter(self.config())
        try:
            for value in (None, ""):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ContractValidationError, "task_text"):
                        adapter.validate_request(self.request(task_text=value))
            with self.assertRaisesRegex(ContractValidationError, "observer"):
                adapter.validate_request(
                    self.request(
                        observer_id="subject-01",
                        observer_metadata={"age": 20},
                    )
                )
            with self.assertRaisesRegex(ContractValidationError, "prompt"):
                adapter.validate_request(
                    self.request(
                        model_options={
                            "gazexplain": {"prompt_template": "find {target}"}
                        }
                    )
                )
        finally:
            adapter.close()

    def test_same_image_two_unicode_tasks_have_distinct_ids_and_records(self) -> None:
        first_request = self.request(
            task_text="Question: 图中有什么？ 👀",
            item_id="same-image:first-task",
            num_samples=1,
        )
        second_request = self.request(
            task_text="Question: 请寻找红色杯子。",
            target_description="红色杯子",
            item_id="same-image:second-task",
            num_samples=1,
        )
        self.assertNotEqual(first_request.request_id, second_request.request_id)
        adapter = GazeXplainAdapter(self.config())
        try:
            first = adapter.predict(first_request, run_id="two-tasks")[0]
            second = adapter.predict(second_request, run_id="two-tasks")[0]
        finally:
            adapter.close()
        self.assertNotEqual(first.request_id, second.request_id)
        self.assertEqual(first.task_text, "Question: 图中有什么？ 👀")
        self.assertEqual(second.task_text, "Question: 请寻找红色杯子。")
        self.assertEqual(second.target_description, "红色杯子")
        self.assertIn("图中有什么", first.explanation_text)
        self.assertIn("请寻找红色杯子", second.explanation_text)
        self.assertNotEqual(first.fixations, second.fixations)
        self.assertIn(
            "no task-conditioned outputs",
            first.native_metadata["gazexplain"]["image_cache_scope"],
        )

    def test_samples_tokens_alignment_and_native_artifacts_are_preserved(self) -> None:
        adapter = GazeXplainAdapter(self.config())
        try:
            request = self.request(num_samples=3, base_seed=11)
            records = adapter.predict(request, run_id="samples")
        finally:
            adapter.close()
        self.assertEqual([record.sample_index for record in records], [0, 1, 2])
        self.assertEqual([record.seed for record in records], [11, 12, 13])
        self.assertEqual(
            [record.sample_id for record in records],
            [f"{request.request_id}:sample:{index}" for index in range(3)],
        )
        self.assertEqual(len({record.native_artifacts[0].uri for record in records}), 3)
        for record in records:
            metadata = record.native_metadata["gazexplain"]
            self.assertEqual(metadata["resolved_task_text"], request.task_text)
            self.assertTrue(metadata["raw_token_ids"])
            self.assertEqual(
                len(metadata["raw_token_ids_all_steps"]),
                metadata["native_sequence_length"],
            )
            self.assertEqual(
                len(metadata["per_fixation_text_alignment"]),
                len(record.fixations),
            )
            path = Path(urlparse(record.native_artifacts[0].uri).path)
            native = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                native["sample"]["explanation"]["raw_token_ids"],
                [list(row) for row in metadata["raw_token_ids"]],
            )
            self.assertIsNone(native["sample"]["explanation"]["token_scores"])
            for fixation in record.fixations:
                native_fixation = fixation.native_metadata["gazexplain"]
                self.assertTrue(native_fixation["explanation_token_ids"])
                self.assertEqual(fixation.native_timing["unit"], "milliseconds")

    def test_text_truncation_and_no_explanation_are_explicit(self) -> None:
        truncated_adapter = GazeXplainAdapter(self.config(behavior="truncated"))
        try:
            truncated = truncated_adapter.predict(
                self.request(num_samples=1), run_id="truncated"
            )[0]
        finally:
            truncated_adapter.close()
        self.assertTrue(
            truncated.native_metadata["gazexplain"]["explanation_truncated"]
        )
        self.assertTrue(
            any("max_generation_length" in warning for warning in truncated.warnings)
        )

        empty_adapter = GazeXplainAdapter(self.config(behavior="no_explanation"))
        try:
            empty = empty_adapter.predict(
                self.request(num_samples=1), run_id="no-explanation"
            )[0]
        finally:
            empty_adapter.close()
        self.assertIsNone(empty.explanation_text)
        self.assertTrue(any("no explanation" in warning for warning in empty.warnings))
        self.assertTrue(
            all(
                fixation.native_metadata["gazexplain"]["explanation_text"] is None
                for fixation in empty.fixations
            )
        )

    def test_malformed_text_missing_checkpoint_and_timeout_are_actionable(self) -> None:
        malformed = GazeXplainAdapter(self.config(behavior="malformed_text"))
        try:
            with self.assertRaisesRegex(AdapterOutputError, "must be a string"):
                malformed.predict(self.request(num_samples=1), run_id="malformed")
        finally:
            malformed.close()
        missing = GazeXplainAdapter(
            self.config(checkpoint=self.root / "missing.safetensors")
        )
        try:
            with self.assertRaisesRegex(GazeXplainWorkerError, "checkpoint_path"):
                missing.predict(self.request(num_samples=1), run_id="missing")
        finally:
            missing.close()
        timeout = GazeXplainAdapter(
            self.config(behavior="timeout", timeout=0.1)
        )
        try:
            with self.assertRaisesRegex(GazeXplainWorkerError, "timed out"):
                timeout.predict(self.request(num_samples=1), run_id="timeout")
            self.assertIsNone(timeout._process)
        finally:
            timeout.close()
        errored = GazeXplainAdapter(self.config(behavior="error"))
        try:
            with self.assertRaisesRegex(
                GazeXplainWorkerError, "deliberate fake GazeXplain error"
            ):
                errored.predict(self.request(num_samples=1), run_id="error")
        finally:
            errored.close()

    def test_registry_dataset_bridge_manifest_and_resume(self) -> None:
        registry = create_gazexplain_registry(self.config())
        descriptor = registry.describe("gazexplain")
        self.assertTrue(descriptor.requirements.instruction)
        self.assertFalse(descriptor.optional_dependencies)
        item = JsonDatasetAdapter(TINY_GAZE).get_item("scene-a", split="train")
        run_item = gazexplain_run_item_from_dataset_item(
            item,
            task_text="Question: 看到了什么？",
            target_description="场景内容",
        )
        self.assertEqual(run_item["task_text"], "Question: 看到了什么？")
        self.assertEqual(run_item["target_description"], "场景内容")
        inherited_target = gazexplain_run_item_from_dataset_item(
            item,
            task_text="Question: 找到黄色圆形。",
        )
        self.assertEqual(inherited_target["target_description"], "circle")

        engine = RunEngine(registry, repository_root=self.root, console=None)
        run_config = self._run_config()
        first = engine.run(run_config)
        resumed = engine.run(run_config, resume=True)
        lines = first.run_directory.joinpath("predictions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        records = [json.loads(line) for line in lines]
        self.assertEqual(len(records), 2)
        self.assertTrue(
            all(record["task_text"] == "Question: 场景里有什么？" for record in records)
        )
        manifest = json.loads(
            first.run_directory.joinpath("manifest.json").read_text(encoding="utf-8")
        )
        resolved_item = manifest["resolved_configuration"]["dataset"]["items"][0]
        self.assertEqual(resolved_item["task_text"], "Question: 场景里有什么？")
        self.assertEqual(resolved_item["target_description"], "场景")
        self.assertEqual(resumed.succeeded, 1)

    def _run_config(self) -> RunConfig:
        return RunConfig.from_dict(
            {
                "schema_version": 1,
                "run_id": "gazexplain-resume",
                "model": {
                    "id": "gazexplain",
                    "variant": GAZEXPLAIN_JOINT_VARIANT,
                    "options": {"gazexplain": {}},
                    "checkpoint_path": str(self.checkpoint),
                    "checkpoint_fingerprint": None,
                    "upstream_code_ref": "fixture:gazexplain",
                    "upstream_commit": GAZEXPLAIN_UPSTREAM_COMMIT,
                },
                "dataset": {
                    "id": "osie",
                    "version": "fixture",
                    "split": "test",
                    "root": None,
                    "root_fingerprint": "fixture",
                    "items": [
                        {
                            "item_id": "image-1",
                            "image_ref": str(self.image),
                            "image_width": 800,
                            "image_height": 600,
                            "task_text": "Question: 场景里有什么？",
                            "target_description": "场景",
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
                    "num_samples": 2,
                    "base_seed": 3,
                    "seed_strategy": "fixed_per_item",
                    "max_fixations": 6,
                    "time_horizon_s": None,
                },
                "device": "cpu",
                "output_root": str(self.root / "runs"),
                "run_policy": "create",
                "tags": ["fake-gazexplain"],
                "notes": "CPU fake-worker integration test",
                "environment": {"name": "fake", "image_digest": None},
            }
        )


_REAL_VARIABLES = (
    "AVRW_GAZEXPLAIN_UPSTREAM_ROOT",
    "AVRW_GAZEXPLAIN_CHECKPOINT",
    "AVRW_GAZEXPLAIN_HPARAMS",
    "AVRW_GAZEXPLAIN_IMAGE",
    "AVRW_GAZEXPLAIN_IMAGE_WIDTH",
    "AVRW_GAZEXPLAIN_IMAGE_HEIGHT",
    "AVRW_GAZEXPLAIN_LAUNCHER",
)


@unittest.skipUnless(
    all(os.environ.get(name) for name in _REAL_VARIABLES),
    "set all AVRW_GAZEXPLAIN_* variables to run the real GPU smoke",
)
class RealGazeXplainSmokeTest(unittest.TestCase):
    def test_real_same_image_two_tasks(self) -> None:
        output = os.environ.get("AVRW_GAZEXPLAIN_OUTPUT_ROOT")
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if output is None:
            temporary = tempfile.TemporaryDirectory()
            output = temporary.name
        config = GazeXplainAdapterConfig(
            variant=GAZEXPLAIN_JOINT_VARIANT,
            upstream_root=Path(os.environ[_REAL_VARIABLES[0]]).resolve(),
            upstream_commit=GAZEXPLAIN_UPSTREAM_COMMIT,
            checkpoint_path=Path(os.environ[_REAL_VARIABLES[1]]).resolve(),
            checkpoint_fingerprint=f"sha256:{GAZEXPLAIN_JOINT_CHECKPOINT_SHA256}",
            hparams_path=Path(os.environ[_REAL_VARIABLES[2]]).resolve(),
            hparams_fingerprint=f"sha256:{GAZEXPLAIN_JOINT_HPARAMS_SHA256}",
            device="cuda",
            output_root=Path(output).resolve(),
            launcher=tuple(shlex.split(os.environ[_REAL_VARIABLES[6]])),
        )
        adapter = GazeXplainAdapter(config)
        try:
            records = []
            for index, task in enumerate(
                (
                    "Question: What do you see in the image?",
                    "Question: Is there a person in the image? Answer: yes.",
                )
            ):
                records.extend(
                    adapter.predict(
                        InferenceRequest.create(
                            model_id="gazexplain",
                            model_variant=GAZEXPLAIN_JOINT_VARIANT,
                            dataset_id="real-smoke",
                            dataset_split="smoke",
                            item_id=f"one-image:task-{index}",
                            image_ref=os.environ[_REAL_VARIABLES[3]],
                            image_width=int(os.environ[_REAL_VARIABLES[4]]),
                            image_height=int(os.environ[_REAL_VARIABLES[5]]),
                            task_text=task,
                            num_samples=2,
                            base_seed=17,
                            max_fixations=4,
                        ),
                        run_id="real-gazexplain-smoke",
                    )
                )
        finally:
            adapter.close()
            if temporary is not None:
                temporary.cleanup()
        self.assertEqual(len(records), 4)
        self.assertEqual(
            [record.sample_index for record in records], [0, 1, 0, 1]
        )
        self.assertEqual([record.seed for record in records], [17, 18, 17, 18])
        self.assertNotEqual(records[0].request_id, records[2].request_id)
        self.assertTrue(all(record.fixations for record in records))
        self.assertTrue(all(record.explanation_text for record in records))
        self.assertTrue(
            all(
                record.native_metadata["gazexplain"]["raw_token_ids"]
                for record in records
            )
        )
        self.assertTrue(all(record.native_artifacts for record in records))


if __name__ == "__main__":
    unittest.main()

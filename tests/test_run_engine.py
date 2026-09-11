
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from activevision_workbench import (
    AdapterDescriptor,
    AdapterRegistry,
    RunConfig,
    RunConfigurationError,
    RunDirectory,
    RunEngine,
    RunExistsError,
    RunResumeError,
    RunStatus,
    read_predictions_jsonl,
    write_predictions_jsonl,
)
from activevision_workbench.adapters.dummy import (
    DUMMY_CAPABILITIES,
    DUMMY_IDENTITY,
    DUMMY_MODEL_ID,
    DUMMY_MODEL_VARIANT,
    DUMMY_REQUIREMENTS,
    DummyAdapter,
    create_dummy_registry,
)
from activevision_workbench.cli import main as cli_main


def config_data(
    output_root: Path,
    *,
    run_id: str = "test-run",
    item_ids: tuple[str, ...] = ("item-1",),
    num_samples: int = 2,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model": {
            "id": DUMMY_MODEL_ID,
            "variant": DUMMY_MODEL_VARIANT,
            "options": {"dummy": {"explanation_prefix": "Run test"}},
            "checkpoint_path": None,
            "checkpoint_fingerprint": None,
            "upstream_code_ref": "builtin",
            "upstream_commit": None,
        },
        "dataset": {
            "id": "fixture-dataset",
            "version": "fixture-v1",
            "split": "test",
            "root": None,
            "root_fingerprint": "fixture-root-v1",
            "items": [
                {
                    "item_id": item_id,
                    "image_ref": f"images/{item_id}.png",
                    "image_width": 64,
                    "image_height": 48,
                }
                for item_id in item_ids
            ],
            "item_subset": None,
        },
        "request": {
            "task_text": "find the target",
            "target_description": None,
            "observer_id": "observer-1",
            "observer_metadata": {"cohort": "fixture"},
            "num_samples": num_samples,
            "base_seed": 100,
            "seed_strategy": "increment_per_item",
            "max_fixations": 2,
            "time_horizon_s": 1.0,
        },
        "device": "cpu",
        "output_root": str(output_root),
        "run_policy": "create",
        "tags": ["unit-test"],
        "notes": "Dependency-free dummy integration.",
        "environment": {"name": "cpu-test", "image_digest": None},
    }


def make_config(output_root: Path, **overrides: object) -> RunConfig:
    return RunConfig.from_dict(
        config_data(output_root, **overrides), base_dir=output_root.parent
    )


def registry_for(adapter_type: type[DummyAdapter]) -> AdapterRegistry:
    registry = AdapterRegistry(allowed_model_ids={DUMMY_MODEL_ID})
    registry.register(
        AdapterDescriptor(
            model_id=DUMMY_MODEL_ID,
            identity=DUMMY_IDENTITY,
            capabilities=DUMMY_CAPABILITIES,
            requirements=DUMMY_REQUIREMENTS,
            factory=adapter_type,
        )
    )
    return registry


class FailingItemDummy(DummyAdapter):

    def _predict(self, request, *, run_id):  # type: ignore[no-untyped-def]
        if request.item_id == "bad-item":
            raise RuntimeError("intentional item failure")
        return super()._predict(request, run_id=run_id)


class InterruptingDummy(DummyAdapter):

    def _predict(self, request, *, run_id):  # type: ignore[no-untyped-def]
        if request.item_id == "interrupt-item":
            raise KeyboardInterrupt
        return super()._predict(request, run_id=run_id)


class CountingDummy(DummyAdapter):
    calls = 0

    def _predict(self, request, *, run_id):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        return super()._predict(request, run_id=run_id)


class RunConfigurationTests(unittest.TestCase):
    def test_resolves_subset_overrides_and_item_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = config_data(root / "runs", item_ids=("one", "two", "three"))
            dataset = data["dataset"]
            self.assertIsInstance(dataset, dict)
            dataset["item_subset"] = ["three", "one"]
            dataset["items"][2]["task_text"] = "item override"
            config = RunConfig.from_dict(data, base_dir=root)
            self.assertEqual(
                [item.item_id for item in config.selected_items], ["three", "one"]
            )
            requests = config.requests()
            self.assertEqual(requests[0].task_text, "item override")
            self.assertEqual([request.base_seed for request in requests], [100, 102])
            self.assertEqual(config, RunConfig.from_dict(config.to_dict()))

    def test_strict_invalid_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = config_data(root / "never-created")
            request = data["request"]
            self.assertIsInstance(request, dict)
            request["num_samples"] = 0
            with self.assertRaisesRegex(RunConfigurationError, "num_samples"):
                RunConfig.from_dict(data)
            self.assertFalse((root / "never-created").exists())

    def test_json_compatible_yaml_file_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "run.yaml"
            path.write_text(
                json.dumps(config_data(Path("relative-runs"))), encoding="utf-8"
            )
            config = RunConfig.from_file(path)
            self.assertEqual(
                config.output_root, str((root / "relative-runs").resolve())
            )
            self.assertTrue(config.configuration_hash.startswith("sha256:"))


class RunEngineTests(unittest.TestCase):
    def test_successful_single_item_run_and_artifact_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary) / "runs")
            result = RunEngine(create_dummy_registry(), console=None).run(config)
            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertEqual((result.succeeded, result.failed), (1, 0))
            directory = result.run_directory
            for name in (
                "config.yaml",
                "manifest.json",
                "predictions.jsonl",
                "failures.jsonl",
                "logs",
                "native",
                "figures",
                "metrics",
                "slurm",
            ):
                self.assertTrue((directory / name).exists(), name)
            predictions = read_predictions_jsonl(directory / "predictions.jsonl")
            self.assertEqual(len(predictions), 2)
            self.assertEqual([record.seed for record in predictions], [100, 101])

    def test_successful_multi_item_run_uses_distinct_seed_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(
                Path(temporary) / "runs",
                item_ids=("item-1", "item-2", "item-3"),
            )
            result = RunEngine(create_dummy_registry(), console=None).run(config)
            predictions = read_predictions_jsonl(
                result.run_directory / "predictions.jsonl"
            )
            self.assertEqual(result.succeeded, 3)
            seeds_by_item = {
                item_id: [
                    record.seed
                    for record in predictions
                    if record.item_id == item_id
                ]
                for item_id in ("item-1", "item-2", "item-3")
            }
            self.assertEqual(seeds_by_item["item-1"], [100, 101])
            self.assertEqual(seeds_by_item["item-2"], [102, 103])
            self.assertEqual(seeds_by_item["item-3"], [104, 105])

    def test_item_failure_is_isolated_and_traceback_is_referenced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(
                Path(temporary) / "runs",
                item_ids=("good-item", "bad-item", "other-item"),
            )
            engine = RunEngine(registry_for(FailingItemDummy), console=None)
            result = engine.run(config)
            self.assertEqual(result.status, RunStatus.COMPLETED_WITH_ITEM_FAILURES)
            self.assertEqual((result.succeeded, result.failed), (2, 1))
            directory = RunDirectory.open(result.run_directory)
            failures = directory.read_failures()
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0].item_id, "bad-item")
            self.assertIn("RuntimeError", failures[0].exception_type)
            self.assertTrue(
                (directory.path / failures[0].traceback_log_reference).is_file()
            )
            self.assertEqual(len(directory.read_predictions()), 4)

    def test_adapter_config_error_fails_before_run_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "runs"
            data = config_data(output_root)
            model = data["model"]
            self.assertIsInstance(model, dict)
            model["options"] = {"dummy": {"unknown": True}}
            config = RunConfig.from_dict(data)
            with self.assertRaisesRegex(Exception, "unknown dummy adapter option"):
                RunEngine(create_dummy_registry(), console=None).run(config)
            self.assertFalse(output_root.exists())

    def test_overwrite_is_protected_and_requires_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary) / "runs")
            engine = RunEngine(create_dummy_registry(), console=None)
            first = engine.run(config)
            manifest_before = (first.run_directory / "manifest.json").read_bytes()
            with self.assertRaises(RunExistsError):
                engine.run(config)
            self.assertEqual(
                (first.run_directory / "manifest.json").read_bytes(), manifest_before
            )
            replacement = engine.run(config, overwrite=True)
            self.assertEqual(replacement.status, RunStatus.COMPLETED)
            self.assertEqual(len(read_predictions_jsonl(
                replacement.run_directory / "predictions.jsonl"
            )), 2)

    def test_resume_skips_completed_records_without_adapter_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            CountingDummy.calls = 0
            config = make_config(
                Path(temporary) / "runs", item_ids=("one", "two")
            )
            engine = RunEngine(registry_for(CountingDummy), console=None)
            first = engine.run(config)
            self.assertEqual(CountingDummy.calls, 2)
            summary = engine.dry_run(config, resume=True)
            self.assertEqual(
                (summary.will_run, summary.skipped_completed), (0, 2)
            )
            second = engine.run(config, resume=True)
            self.assertEqual(second.status, RunStatus.COMPLETED)
            self.assertEqual(CountingDummy.calls, 2)
            self.assertEqual(
                len(read_predictions_jsonl(first.run_directory / "predictions.jsonl")),
                4,
            )

    def test_resume_rejects_incompatible_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "runs"
            config = make_config(output_root)
            result = RunEngine(create_dummy_registry(), console=None).run(config)
            changed = config_data(output_root)
            changed["notes"] = "scientifically different resolved configuration"
            incompatible = RunConfig.from_dict(changed)
            manifest_before = (result.run_directory / "manifest.json").read_bytes()
            with self.assertRaisesRegex(RunResumeError, "configuration hash"):
                RunEngine(create_dummy_registry(), console=None).run(
                    incompatible, resume=True
                )
            self.assertEqual(
                (result.run_directory / "manifest.json").read_bytes(), manifest_before
            )

    def test_partial_samples_resume_without_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary) / "runs", num_samples=3)
            engine = RunEngine(create_dummy_registry(), console=None)
            result = engine.run(config)
            prediction_path = result.run_directory / "predictions.jsonl"
            original = read_predictions_jsonl(prediction_path)
            write_predictions_jsonl(prediction_path, original[:1], overwrite=True)
            resumed = engine.run(config, resume=True)
            restored = read_predictions_jsonl(prediction_path)
            self.assertEqual(resumed.status, RunStatus.COMPLETED)
            self.assertEqual(len(restored), 3)
            self.assertEqual(
                len({(record.request_id, record.sample_index) for record in restored}),
                3,
            )

    def test_resume_recovers_one_torn_trailing_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary) / "runs")
            engine = RunEngine(create_dummy_registry(), console=None)
            result = engine.run(config)
            prediction_path = result.run_directory / "predictions.jsonl"
            with prediction_path.open("ab") as stream:
                stream.write(b'{"torn":')
            resumed = engine.run(config, resume=True)
            self.assertEqual(resumed.status, RunStatus.COMPLETED)
            self.assertEqual(len(read_predictions_jsonl(prediction_path)), 2)

    def test_failed_item_requires_explicit_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(
                Path(temporary) / "runs", item_ids=("good-item", "bad-item")
            )
            failing = RunEngine(registry_for(FailingItemDummy), console=None)
            first = failing.run(config)
            held = failing.dry_run(config, resume=True)
            self.assertEqual((held.will_run, held.skipped_failed), (0, 1))
            second = RunEngine(create_dummy_registry(), console=None).run(
                config, resume=True, retry_failures=True
            )
            self.assertEqual(first.failed, 1)
            self.assertEqual(second.status, RunStatus.COMPLETED)
            self.assertEqual(second.succeeded, 2)
            self.assertEqual(
                len(read_predictions_jsonl(second.run_directory / "predictions.jsonl")),
                4,
            )

    def test_interruption_leaves_readable_resumable_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(
                Path(temporary) / "runs",
                item_ids=("first-item", "interrupt-item", "last-item"),
            )
            interrupted = RunEngine(
                registry_for(InterruptingDummy), console=None
            ).run(config)
            self.assertEqual(interrupted.status, RunStatus.INTERRUPTED)
            self.assertEqual(
                len(read_predictions_jsonl(
                    interrupted.run_directory / "predictions.jsonl"
                )),
                2,
            )
            manifest = json.loads(
                (interrupted.run_directory / "manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "interrupted")
            resumed = RunEngine(create_dummy_registry(), console=None).run(
                config, resume=True
            )
            self.assertEqual(resumed.status, RunStatus.COMPLETED)
            self.assertEqual(resumed.succeeded, 3)

    def test_manifest_contains_required_stable_provenance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary) / "runs")
            result = RunEngine(
                create_dummy_registry(), repository_root=Path.cwd(), console=None
            ).run(config)
            manifest = json.loads(
                (result.run_directory / "manifest.json").read_text()
            )
            required = {
                "schema_version",
                "run_id",
                "project",
                "activevision_git",
                "model",
                "dataset",
                "resolved_configuration",
                "configuration_hash",
                "seeds",
                "environment",
                "started_at",
                "ended_at",
                "status",
                "counts",
                "warnings",
                "slurm",
            }
            self.assertTrue(required.issubset(manifest))
            try:
                git_result = subprocess.run(
                    ("git", "rev-parse", "HEAD"),
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError:
                git_result = None
            if git_result is None or git_result.returncode != 0:
                self.assertIsNone(manifest["activevision_git"]["commit"])
                self.assertIsNone(manifest["activevision_git"]["dirty"])
                return
            expected_commit = git_result.stdout.strip()
            self.assertEqual(
                manifest["activevision_git"]["commit"], expected_commit
            )
            self.assertEqual(manifest["model"]["id"], "dummy")
            self.assertEqual(manifest["dataset"]["id"], "fixture-dataset")
            self.assertEqual(manifest["counts"]["total"], 1)
            self.assertEqual(len(next(iter(
                manifest["seeds"]["actual_per_request"].values()
            ))), 2)
            self.assertEqual(
                set(manifest["slurm"]),
                {"job_id", "array_job_id", "array_task_id", "partition", "node_list"},
            )

    def test_new_dry_run_has_no_filesystem_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "runs"
            config = make_config(output_root)
            summary = RunEngine(create_dummy_registry(), console=None).dry_run(config)
            self.assertEqual(summary.will_run, 1)
            self.assertFalse(output_root.exists())


class CliTests(unittest.TestCase):
    def test_cli_runs_dummy_and_prints_final_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "run.yaml"
            config_path.write_text(
                json.dumps(config_data(root / "runs")), encoding="utf-8"
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(("run", "--config", str(config_path)))
            self.assertEqual(code, 0)
            self.assertIn("Run directory:", output.getvalue())
            self.assertTrue((root / "runs" / "test-run" / "manifest.json").is_file())

    def test_cpu_only_no_site_cli_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "run.yaml"
            config_path.write_text(
                json.dumps(config_data(root / "runs", num_samples=1)),
                encoding="utf-8",
            )
            repository = Path(__file__).resolve().parents[1]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repository / "src")
            environment["PYTHONPYCACHEPREFIX"] = str(root / "pycache")
            result = subprocess.run(
                (
                    sys.executable,
                    "-S",
                    "-m",
                    "activevision_workbench",
                    "run",
                    "--config",
                    str(config_path),
                ),
                cwd=repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Run directory:", result.stdout)


if __name__ == "__main__":
    unittest.main()

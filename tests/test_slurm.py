from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from activevision_workbench.adapters.dummy import create_dummy_registry
from activevision_workbench.cli import main as cli_main
from activevision_workbench.errors import (
    SlurmConfigurationError,
    SlurmMergeError,
    SlurmPlanError,
    SlurmSubmissionError,
)
from activevision_workbench.run_artifacts import FailureRecord, RunDirectory, RunStatus
from activevision_workbench.run_config import RunConfig
from activevision_workbench.serialization import read_predictions_jsonl
from activevision_workbench.slurm import (
    SlurmConfig,
    SlurmExecutor,
    create_plan,
    materialize_plan,
    merge_shards,
    retry_plan,
    run_shard,
)


def run_data(output_root: Path, item_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "slurm-test-run",
        "model": {
            "id": "dummy",
            "variant": "deterministic-v1",
            "options": {"dummy": {"explanation_prefix": "Slurm test"}},
            "checkpoint_path": None,
            "checkpoint_fingerprint": None,
            "upstream_code_ref": "builtin",
            "upstream_commit": None,
        },
        "dataset": {
            "id": "slurm-fixture",
            "version": "1",
            "split": "test",
            "root": None,
            "root_fingerprint": "fixture",
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
            "task_text": "find target",
            "target_description": None,
            "observer_id": "observer-1",
            "observer_metadata": {},
            "num_samples": 2,
            "base_seed": 100,
            "seed_strategy": "increment_per_item",
            "max_fixations": 2,
            "time_horizon_s": 1.0,
        },
        "device": "cpu",
        "output_root": str(output_root),
        "run_policy": "create",
        "tags": ["slurm-test"],
        "notes": None,
        "environment": {"name": "test", "image_digest": None},
    }


def slurm_data(root: Path, shard_count: int = 3) -> dict[str, object]:
    return {
        "schema_version": 1,
        "account": "def-example",
        "partition": None,
        "job_name": "avrw-test",
        "shard_count": shard_count,
        "max_parallel_tasks": None,
        "resources": {
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 2,
            "gpus_per_node": 1,
            "gpu_type": "h100",
            "memory": "8G",
            "time_limit": "00:10:00",
        },
        "environment": {
            "name": "test-python",
            "modules": [],
            "launcher": [sys.executable],
            "apptainer_image": None,
            "apptainer_gpu": False,
            "bind_paths": [],
        },
        "working_directory": str(root),
        "dataset_roots": [str(root / "data root")],
        "checkpoint_roots": [str(root / "checkpoints")],
        "log_directory": str(root / "log files"),
        "email": None,
        "mail_types": [],
        "requeue": False,
        "max_retries": 1,
        "signal_seconds": 120,
        "dependency_job_ids": ["123"],
    }


def configs(
    root: Path, *, items: tuple[str, ...], shards: int = 3
) -> tuple[RunConfig, SlurmConfig]:
    (root / "data root").mkdir(exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    run = RunConfig.from_dict(run_data(root / "runs", items), base_dir=root)
    slurm = SlurmConfig.from_dict(slurm_data(root, shards), base_dir=root)
    return run, slurm


class SlurmConfigurationTests(unittest.TestCase):
    def test_strict_round_trip_and_missing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config = configs(root, items=("one",))
            self.assertEqual(config, SlurmConfig.from_dict(config.to_dict()))
            data = slurm_data(root)
            environment = data["environment"]
            self.assertIsInstance(environment, dict)
            environment["launcher"] = []
            with self.assertRaisesRegex(SlurmConfigurationError, "launcher"):
                SlurmConfig.from_dict(data)

    def test_assignment_strategy_is_optional_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = slurm_data(root)
            legacy = SlurmConfig.from_dict(data)
            self.assertEqual(legacy.assignment_strategy, "hash_bucket")
            self.assertNotIn("assignment_strategy", legacy.to_dict())
            data["assignment_strategy"] = "balanced"
            balanced = SlurmConfig.from_dict(data)
            self.assertEqual(balanced.assignment_strategy, "balanced")
            self.assertEqual(balanced.to_dict()["assignment_strategy"], "balanced")
            data["assignment_strategy"] = "random"
            with self.assertRaisesRegex(
                SlurmConfigurationError, "assignment_strategy"
            ):
                SlurmConfig.from_dict(data)

    def test_missing_apptainer_image_is_rejected_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = configs(root, items=("one",))
            data = slurm_data(root)
            environment = data["environment"]
            self.assertIsInstance(environment, dict)
            environment["launcher"] = []
            environment["apptainer_image"] = "missing.sif"
            slurm = SlurmConfig.from_dict(data, base_dir=root)
            with self.assertRaisesRegex(SlurmPlanError, "does not exist"):
                SlurmExecutor().dry_run(run, slurm)
            self.assertFalse((root / "runs").exists())


class SlurmPlanningTests(unittest.TestCase):
    def test_assignment_is_deterministic_complete_and_handles_empty_shards(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("d", "a", "c", "b"), shards=8)
            first = create_plan(run, slurm)
            second = create_plan(run, slurm)
            self.assertEqual(first.plan_hash, second.plan_hash)
            self.assertEqual(first.shards, second.shards)
            assigned = [item_id for shard in first.shards for item_id in shard.item_ids]
            self.assertCountEqual(assigned, ("d", "a", "c", "b"))
            self.assertEqual(len(assigned), len(set(assigned)))
            self.assertTrue(any(not shard.item_ids for shard in first.shards))

    def test_balanced_assignment_is_deterministic_complete_and_even(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_data_value = run_data(
                root / "runs", tuple(f"item-{index}" for index in range(10))
            )
            slurm_data_value = slurm_data(root, shard_count=3)
            slurm_data_value["assignment_strategy"] = "balanced"
            run = RunConfig.from_dict(run_data_value, base_dir=root)
            slurm = SlurmConfig.from_dict(slurm_data_value, base_dir=root)
            first = create_plan(run, slurm)
            second = create_plan(run, slurm)
            self.assertEqual(first.shards, second.shards)
            assigned = [item_id for shard in first.shards for item_id in shard.item_ids]
            self.assertCountEqual(
                assigned, tuple(f"item-{index}" for index in range(10))
            )
            self.assertEqual(len(assigned), len(set(assigned)))
            self.assertEqual(
                sorted(len(shard.item_ids) for shard in first.shards), [3, 3, 4]
            )

    def test_image_balanced_assignment_keeps_shared_images_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = run_data(
                root / "runs",
                ("a:one", "a:two", "b:one", "b:two", "c:one"),
            )
            items = data["dataset"]
            self.assertIsInstance(items, dict)
            records = items["items"]
            self.assertIsInstance(records, list)
            for record in records:
                self.assertIsInstance(record, dict)
                image = str(record["item_id"]).split(":", 1)[0]
                record["image_ref"] = f"images/{image}.png"
            slurm_data_value = slurm_data(root, shard_count=2)
            slurm_data_value["assignment_strategy"] = "balanced_by_image"
            run = RunConfig.from_dict(data, base_dir=root)
            slurm = SlurmConfig.from_dict(slurm_data_value, base_dir=root)
            plan = create_plan(run, slurm)
            item_to_shard = {
                item_id: shard.index
                for shard in plan.shards
                for item_id in shard.item_ids
            }
            self.assertEqual(item_to_shard["a:one"], item_to_shard["a:two"])
            self.assertEqual(item_to_shard["b:one"], item_to_shard["b:two"])
            self.assertEqual(sorted(len(shard.item_ids) for shard in plan.shards), [2, 3])

    def test_script_fields_and_shell_quoting(self) -> None:
        with tempfile.TemporaryDirectory(prefix="avrw slurm ") as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one", "two"), shards=2)
            plan = create_plan(run, slurm)
            script = plan.array_script
            self.assertIn("#SBATCH --account=def-example", script)
            self.assertIn("#SBATCH --gpus-per-node=h100:1", script)
            self.assertIn("#SBATCH --signal=B:TERM@120", script)
            self.assertIn("#SBATCH --dependency=afterok:123", script)
            self.assertIn(f"cd '{root}'", script)
            self.assertIn("' --shard-index", script)
            self.assertNotIn("lchen", script)

    def test_dry_run_does_not_write_or_call_sbatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one",))
            calls: list[tuple[str, ...]] = []

            def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "1\n", "")

            plan = SlurmExecutor(runner).dry_run(run, slurm)
            summary = plan.summary()
            self.assertEqual(calls, [])
            self.assertFalse(plan.plan_directory.exists())
            self.assertEqual(summary["dataset_roots"], list(slurm.dataset_roots))
            self.assertIn("array_script", summary)

    def test_materialize_only_writes_scripts_without_calling_sbatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one", "two"), shards=2)
            run_path = root / "run.json"
            slurm_path = root / "slurm.json"
            run_path.write_text(json.dumps(run.to_dict()), encoding="utf-8")
            slurm_path.write_text(json.dumps(slurm.to_dict()), encoding="utf-8")
            output = StringIO()
            with patch(
                "activevision_workbench.slurm.execution._run_command"
            ) as command_runner:
                with redirect_stdout(output):
                    result = cli_main(
                        [
                            "submit",
                            "--config",
                            str(run_path),
                            "--slurm",
                            str(slurm_path),
                            "--materialize-only",
                        ]
                    )
                command_runner.assert_not_called()
            plan = create_plan(run, slurm)
            self.assertEqual(result, 0)
            self.assertTrue((plan.plan_directory / "array.sbatch").is_file())
            self.assertTrue((plan.plan_directory / "merge.sbatch").is_file())
            self.assertIn(str(plan.plan_directory), output.getvalue())

    def test_checked_in_scripts_are_valid_path_neutral_profiles(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        profiles = {
            "scandiff.sbatch": (
                "--cpus-per-task=4",
                "--mem=32G",
                "h100",
            ),
            "tpp_gaze.sbatch": (
                "--cpus-per-task=4",
                "--mem=24G",
                "h100",
            ),
            "individual_scanpath.sbatch": (
                "--cpus-per-task=4",
                "--mem=24G",
                "h100",
            ),
            "gazexplain.sbatch": (
                "--cpus-per-task=8",
                "--mem=48G",
                "h100",
            ),
        }
        for name, resource_lines in profiles.items():
            path = repository / "slurm" / name
            script = path.read_text(encoding="utf-8")
            self.assertIn("#SBATCH --array=0-7%4", script)
            self.assertIn("#SBATCH --time=00:30:00", script)
            self.assertIn(
                f"#SBATCH --gpus-per-node={resource_lines[2]}:1", script
            )
            self.assertIn(resource_lines[0], script)
            self.assertIn(resource_lines[1], script)
            self.assertNotIn("/home/", script)
            self.assertNotIn("lchen", script)
            subprocess.run(["bash", "-n", str(path)], check=True)
        merge = repository / "slurm" / "merge.sbatch"
        self.assertNotIn("/home/", merge.read_text(encoding="utf-8"))
        subprocess.run(["bash", "-n", str(merge)], check=True)
        evaluation = repository / "slurm" / "evaluate_final.sbatch"
        evaluation_script = evaluation.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-5%6", evaluation_script)
        self.assertIn("#SBATCH --time=00:30:00", evaluation_script)
        self.assertNotIn("#SBATCH --gpus", evaluation_script)
        self.assertNotIn("/home/", evaluation_script)
        self.assertNotIn("lchen", evaluation_script)
        subprocess.run(["bash", "-n", str(evaluation)], check=True)

    def test_fake_submission_parses_job_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one", "two"), shards=2)
            outputs = iter(("901;rorqual\n", "902\n"))
            calls: list[tuple[str, ...]] = []

            def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, next(outputs), "")

            result = SlurmExecutor(runner).submit(run, slurm)
            self.assertEqual((result.array_job_id, result.merge_job_id), ("901", "902"))
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0:2], ("sbatch", "--parsable"))
            self.assertIn("--dependency=afterany:901", calls[1])
            state = json.loads((result.plan_directory / "plan.json").read_text())
            self.assertEqual(state["array_job_id"], "901")
            self.assertEqual(state["merge_job_id"], "902")

    def test_unavailable_sbatch_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one",), shards=1)

            def unavailable(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
                raise FileNotFoundError(command[0])

            with self.assertRaisesRegex(
                SlurmSubmissionError, "could not execute sbatch"
            ):
                SlurmExecutor(unavailable).submit(run, slurm)


class SlurmExecutionTests(unittest.TestCase):
    def test_merge_accepts_equivalent_item_membership_in_another_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one", "two", "three"), shards=1)
            plan = create_plan(run, slurm)
            materialize_plan(plan)
            run_shard(
                plan.plan_directory,
                0,
                lambda config: create_dummy_registry(),
            )
            manifest_path = (
                plan.plan_directory
                / "shards"
                / "0000"
                / "runs"
                / str(run.run_id)
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selected = manifest["dataset"]["selected_item_ids"]
            manifest["dataset"]["selected_item_ids"] = list(reversed(selected))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            destination = merge_shards(
                plan.plan_directory, create_dummy_registry()
            )
            merged = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(merged["status"], RunStatus.COMPLETED.value)

    def test_per_shard_outputs_preserve_full_config_seeds_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one", "two", "three", "four"), shards=3)
            plan = create_plan(run, slurm)
            outputs = iter(("701\n", "702\n"))

            def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(command, 0, next(outputs), "")

            SlurmExecutor(runner).submit(run, slurm)
            for shard in plan.shards:
                result = run_shard(
                    plan.plan_directory,
                    shard.index,
                    lambda config: create_dummy_registry(),
                )
                if shard.item_ids:
                    self.assertIsNotNone(result)
                    self.assertIn(
                        f"shards/{shard.index:04d}/runs",
                        str(result.run_directory),
                    )
                else:
                    self.assertIsNone(result)
            evidence_shard = next(shard for shard in plan.shards if shard.item_ids)
            evidence_run = (
                plan.plan_directory
                / "shards"
                / f"{evidence_shard.index:04d}"
                / "runs"
                / str(run.run_id)
            )
            request = next(
                request
                for request in run.requests()
                if request.item_id == evidence_shard.item_ids[0]
            )
            failure_log = evidence_run / "logs" / "failures" / "historical.log"
            failure_log.parent.mkdir(exist_ok=True)
            failure_log.write_text("historical test failure\n", encoding="utf-8")
            historical = FailureRecord(
                request_id=request.request_id,
                item_id=request.item_id,
                exception_type="builtins.RuntimeError",
                message="historical test failure",
                traceback_log_reference="logs/failures/historical.log",
                occurred_at="2026-01-01T00:00:00Z",
                attempt=1,
            )
            with (evidence_run / "failures.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(historical.to_dict()) + "\n")
            destination = merge_shards(plan.plan_directory, create_dummy_registry())
            predictions = read_predictions_jsonl(destination / "predictions.jsonl")
            expected_seeds = {
                request.item_id: [request.base_seed, request.base_seed + 1]
                for request in run.requests()
            }
            actual = {
                item_id: sorted(
                    record.seed
                    for record in predictions
                    if record.item_id == item_id
                )
                for item_id in expected_seeds
            }
            self.assertEqual(actual, expected_seeds)
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["status"], RunStatus.COMPLETED.value)
            self.assertEqual(
                manifest["counts"],
                {"total": 4, "succeeded": 4, "failed": 0, "skipped": 0},
            )
            self.assertEqual(manifest["slurm"]["plan_hash"], plan.plan_hash)
            self.assertEqual(manifest["slurm"]["array_job_id"], "701")
            self.assertEqual(manifest["slurm"]["job_id"], "702")
            self.assertTrue((destination / "slurm" / "shards").is_dir())
            merged_failures = RunDirectory.open(destination).read_failures()
            self.assertEqual(len(merged_failures), 1)
            self.assertTrue(
                (destination / merged_failures[0].traceback_log_reference).is_file()
            )

    def test_duplicate_prediction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one", "two", "three", "four"), shards=3)
            plan = create_plan(run, slurm)
            materialize_plan(plan)
            nonempty = [shard for shard in plan.shards if shard.item_ids]
            self.assertEqual(len(nonempty), 2)
            for shard in plan.shards:
                run_shard(
                    plan.plan_directory,
                    shard.index,
                    lambda config: create_dummy_registry(),
                )
            source = (
                plan.plan_directory
                / "shards"
                / f"{nonempty[0].index:04d}"
                / "runs"
                / str(run.run_id)
                / "predictions.jsonl"
            )
            target = (
                plan.plan_directory
                / "shards"
                / f"{nonempty[1].index:04d}"
                / "runs"
                / str(run.run_id)
                / "predictions.jsonl"
            )
            first = source.read_text(encoding="utf-8").splitlines()[0]
            with target.open("a", encoding="utf-8") as stream:
                stream.write(first + "\n")
            with self.assertRaisesRegex(SlurmMergeError, "duplicate"):
                merge_shards(plan.plan_directory, create_dummy_registry())

    def test_manual_submission_job_ids_are_preserved_at_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one",), shards=1)
            plan = SlurmExecutor().materialize(run, slurm)
            run_shard(
                plan.plan_directory,
                0,
                lambda config: create_dummy_registry(),
            )
            with patch.dict(
                os.environ,
                {"AVRW_ARRAY_JOB_ID": "801", "SLURM_JOB_ID": "802"},
            ):
                destination = merge_shards(
                    plan.plan_directory, create_dummy_registry()
                )
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["slurm"]["array_job_id"], "801")
            self.assertEqual(manifest["slurm"]["job_id"], "802")
            evidence = json.loads(
                (destination / "slurm" / "plan.json").read_text()
            )
            self.assertEqual(evidence["array_job_id"], "801")
            self.assertEqual(evidence["merge_job_id"], "802")

    def test_modified_shard_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one",), shards=1)
            plan = create_plan(run, slurm)
            materialize_plan(plan)
            run_shard(
                plan.plan_directory,
                0,
                lambda config: create_dummy_registry(),
            )
            manifest_path = (
                plan.plan_directory
                / "shards"
                / "0000"
                / "runs"
                / str(run.run_id)
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["resolved_configuration"]["notes"] = "modified"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(SlurmMergeError, "modified"):
                merge_shards(plan.plan_directory, create_dummy_registry())

    def test_partial_merge_and_retry_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, slurm = configs(root, items=("one", "two", "three", "four"), shards=3)
            plan = create_plan(run, slurm)
            materialize_plan(plan)
            nonempty = [shard for shard in plan.shards if shard.item_ids]
            self.assertGreaterEqual(len(nonempty), 2)
            first, second = nonempty[:2]
            configured_output_roots: list[str] = []

            def registry_factory(config: RunConfig):
                configured_output_roots.append(config.output_root)
                return create_dummy_registry()

            run_shard(plan.plan_directory, first.index, registry_factory)
            self.assertEqual(
                configured_output_roots,
                [str(plan.plan_directory / "shards" / f"{first.index:04d}" / "runs")],
            )
            second_status = (
                plan.plan_directory
                / "shards"
                / f"{second.index:04d}"
                / "status.json"
            )
            value = json.loads(second_status.read_text())
            value.update({"state": "failed", "attempts": 1, "error": "fixture failure"})
            second_status.write_text(json.dumps(value), encoding="utf-8")
            retry = retry_plan(plan.plan_directory)
            self.assertIn(first.index, retry.completed_shards)
            self.assertIn(second.index, retry.eligible_shards)
            destination = merge_shards(plan.plan_directory, create_dummy_registry())
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(
                manifest["status"],
                RunStatus.COMPLETED_WITH_ITEM_FAILURES.value,
            )
            self.assertGreater(manifest["counts"]["failed"], 0)
            value["attempts"] = 2
            second_status.write_text(json.dumps(value), encoding="utf-8")
            exhausted = retry_plan(plan.plan_directory)
            self.assertIn(second.index, exhausted.exhausted_shards)

    @unittest.skipUnless(
        os.environ.get("AVRW_RUN_REAL_SLURM_SMOKE") == "1"
        and os.environ.get("AVRW_SLURM_SMOKE_ACCOUNT"),
        "set AVRW_RUN_REAL_SLURM_SMOKE=1 and AVRW_SLURM_SMOKE_ACCOUNT "
        "to submit one CPU no-op",
    )
    def test_optional_real_slurm_client_smoke(self) -> None:
        account = os.environ["AVRW_SLURM_SMOKE_ACCOUNT"]
        command = [
            "sbatch",
            "--parsable",
            f"--account={account}",
            "--time=00:01:00",
            "--cpus-per-task=1",
            "--mem=128M",
        ]
        partition = os.environ.get("AVRW_SLURM_SMOKE_PARTITION")
        if partition:
            command.append(f"--partition={partition}")
        command.extend(("--wrap", "true"))
        result = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip().split(";", 1)[0].isdigit())


if __name__ == "__main__":
    unittest.main()

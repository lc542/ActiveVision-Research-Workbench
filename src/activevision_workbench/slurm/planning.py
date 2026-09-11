from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path

from activevision_workbench.errors import SlurmPlanError
from activevision_workbench.run_config import RunConfig, RunPolicy
from activevision_workbench.slurm.config import SlurmConfig


@dataclass(frozen=True, slots=True)
class ShardSpec:
    index: int
    item_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "shard_index": self.index,
            "item_ids": list(self.item_ids),
            "item_count": len(self.item_ids),
        }


@dataclass(frozen=True, slots=True)
class SlurmPlan:
    run_config: RunConfig
    slurm_config: SlurmConfig
    plan_hash: str
    plan_directory: Path
    final_run_directory: Path
    shards: tuple[ShardSpec, ...]
    selected_shards: tuple[int, ...]
    retry: bool = False

    @property
    def array_script(self) -> str:
        return generate_array_script(self)

    @property
    def merge_script(self) -> str:
        return generate_merge_script(self)

    @property
    def environment_command(self) -> tuple[str, ...]:
        return environment_command(self.run_config, self.slurm_config)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_hash": self.plan_hash,
            "plan_directory": str(self.plan_directory),
            "final_run_directory": str(self.final_run_directory),
            "run_id": self.run_config.run_id,
            "run_configuration_hash": self.run_config.configuration_hash,
            "slurm_configuration_hash": self.slurm_config.configuration_hash,
            "retry": self.retry,
            "selected_shards": list(self.selected_shards),
            "shards": [shard.to_dict() for shard in self.shards],
            "array_job_id": None,
            "merge_job_id": None,
            "submission_attempts": 0,
            "submissions": [],
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dry_run": True,
            "plan_hash": self.plan_hash,
            "plan_directory": str(self.plan_directory),
            "output_run_directory": str(self.final_run_directory),
            "resources": self.slurm_config.resources.to_dict(),
            "account": self.slurm_config.account,
            "partition": self.slurm_config.partition,
            "environment_name": self.slurm_config.environment.name,
            "environment_command": list(self.environment_command),
            "dataset_roots": list(self.slurm_config.dataset_roots),
            "checkpoint_roots": list(self.slurm_config.checkpoint_roots),
            "log_directory": self.slurm_config.log_directory,
            "selected_shards": list(self.selected_shards),
            "shards": [shard.to_dict() for shard in self.shards],
            "array_script": self.array_script if self.selected_shards else None,
            "merge_script": self.merge_script,
        }


def create_plan(run_config: RunConfig, slurm_config: SlurmConfig) -> SlurmPlan:
    if run_config.run_id is None:
        raise SlurmPlanError(
            "Slurm submission requires an explicit run_id for reproducible resume"
        )
    if run_config.run_policy is not RunPolicy.CREATE:
        raise SlurmPlanError(
            "Slurm plans require run_policy 'create'; use submit --resume without "
            "changing the run configuration"
        )
    buckets = _assign_items(run_config, slurm_config)
    shards = tuple(
        ShardSpec(index=index, item_ids=tuple(item_ids))
        for index, item_ids in enumerate(buckets)
    )
    payload = {
        "run_configuration_hash": run_config.configuration_hash,
        "slurm_configuration_hash": slurm_config.configuration_hash,
        "shards": [shard.to_dict() for shard in shards],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    plan_hash = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    suffix = plan_hash.split(":", 1)[1][:12]
    plan_directory = (
        Path(run_config.output_root) / ".slurm-work" / f"{run_config.run_id}-{suffix}"
    )
    return SlurmPlan(
        run_config=run_config,
        slurm_config=slurm_config,
        plan_hash=plan_hash,
        plan_directory=plan_directory,
        final_run_directory=Path(run_config.output_root) / run_config.run_id,
        shards=shards,
        selected_shards=tuple(range(slurm_config.shard_count)),
    )


def _assign_items(
    run_config: RunConfig, slurm_config: SlurmConfig
) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(slurm_config.shard_count)]
    if slurm_config.assignment_strategy == "balanced_by_image":
        groups: dict[str, list[str]] = {}
        for item in run_config.selected_items:
            groups.setdefault(item.image_ref, []).append(item.item_id)
        ordered_groups = sorted(
            groups.items(),
            key=lambda group: (
                -len(group[1]),
                hashlib.sha256(group[0].encode("utf-8")).digest(),
                group[0],
            ),
        )
        bucket_sizes = [0] * slurm_config.shard_count
        for _, item_ids in ordered_groups:
            index = min(
                range(slurm_config.shard_count),
                key=lambda value: (bucket_sizes[value], value),
            )
            ordered_ids = sorted(
                item_ids,
                key=lambda item_id: (
                    hashlib.sha256(item_id.encode("utf-8")).digest(),
                    item_id,
                ),
            )
            buckets[index].extend(ordered_ids)
            bucket_sizes[index] += len(ordered_ids)
        return buckets
    if slurm_config.assignment_strategy == "balanced":
        ordered = sorted(
            (item.item_id for item in run_config.selected_items),
            key=lambda item_id: (
                hashlib.sha256(item_id.encode("utf-8")).digest(),
                item_id,
            ),
        )
        for position, item_id in enumerate(ordered):
            buckets[position % slurm_config.shard_count].append(item_id)
        return buckets
    for item in run_config.selected_items:
        digest = hashlib.sha256(item.item_id.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % slurm_config.shard_count
        buckets[index].append(item.item_id)
    return buckets


def environment_command(
    run_config: RunConfig, slurm_config: SlurmConfig
) -> tuple[str, ...]:
    environment = slurm_config.environment
    if environment.apptainer_image is None:
        return (*environment.launcher, "-m", "activevision_workbench")
    command = ["apptainer", "exec", "--cleanenv"]
    if environment.apptainer_gpu:
        command.append("--nv")
    read_only = (
        *environment.bind_paths,
        *slurm_config.dataset_roots,
        *slurm_config.checkpoint_roots,
        slurm_config.working_directory,
    )
    writable = (run_config.output_root, slurm_config.log_directory)
    seen: set[str] = set()
    for path in read_only:
        if path not in seen:
            command.extend(("--bind", f"{path}:{path}:ro"))
            seen.add(path)
    for path in writable:
        if path not in seen:
            command.extend(("--bind", f"{path}:{path}"))
            seen.add(path)
    command.extend(
        (
            "--pwd",
            slurm_config.working_directory,
            environment.apptainer_image,
            "python",
            "-m",
            "activevision_workbench",
        )
    )
    return tuple(command)


def generate_array_script(plan: SlurmPlan) -> str:
    config = plan.slurm_config
    resources = config.resources
    lines = _header(config, job_name=config.job_name)
    lines.extend(
        (
            (
                "#SBATCH --array="
                f"{_array_expression(plan.selected_shards, config.max_parallel_tasks)}"
            ),
            f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
            f"#SBATCH --mem={resources.memory}",
            f"#SBATCH --time={resources.time_limit}",
            f"#SBATCH --signal=B:TERM@{config.signal_seconds}",
        )
    )
    if resources.gpus_per_node:
        gpu = str(resources.gpus_per_node)
        if resources.gpu_type is not None:
            gpu = f"{resources.gpu_type}:{gpu}"
        lines.append(f"#SBATCH --gpus-per-node={gpu}")
    lines.extend(_mail_and_dependencies(config))
    lines.extend(
        (
            "",
            "set -euo pipefail",
            "export PYTHONUNBUFFERED=1",
            f"export AVRW_SLURM_PLAN_HASH={shlex.quote(plan.plan_hash)}",
            *_module_lines(config),
            f"cd {shlex.quote(config.working_directory)}",
            "exec "
            + shlex.join(plan.environment_command)
            + " slurm-run-shard --plan "
            + shlex.quote(str(plan.plan_directory))
            + ' --shard-index "${SLURM_ARRAY_TASK_ID}"',
            "",
        )
    )
    return "\n".join(lines)


def generate_merge_script(plan: SlurmPlan) -> str:
    config = plan.slurm_config
    lines = _header(config, job_name=f"{config.job_name[:121]}-merge")
    lines.extend(
        (
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --mem=4G",
            "#SBATCH --time=00:15:00",
            *_mail_lines(config),
            "",
            "set -euo pipefail",
            "export PYTHONUNBUFFERED=1",
            *_module_lines(config),
            f"cd {shlex.quote(config.working_directory)}",
            "exec "
            + shlex.join(plan.environment_command)
            + " slurm-merge --plan "
            + shlex.quote(str(plan.plan_directory))
            + " --replace-matching-plan",
            "",
        )
    )
    return "\n".join(lines)


def _header(config: SlurmConfig, *, job_name: str) -> list[str]:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --account={config.account}",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --open-mode=append",
        "#SBATCH --requeue" if config.requeue else "#SBATCH --no-requeue",
    ]
    if config.partition is not None:
        lines.append(f"#SBATCH --partition={config.partition}")
    return lines


def _mail_lines(config: SlurmConfig) -> tuple[str, ...]:
    if config.email is None:
        return ()
    return (
        f"#SBATCH --mail-user={config.email}",
        f"#SBATCH --mail-type={','.join(config.mail_types) or 'FAIL'}",
    )


def _mail_and_dependencies(config: SlurmConfig) -> tuple[str, ...]:
    lines = list(_mail_lines(config))
    if config.dependency_job_ids:
        lines.append(
            "#SBATCH --dependency=afterok:"
            + ":".join(config.dependency_job_ids)
        )
    return tuple(lines)


def _module_lines(config: SlurmConfig) -> tuple[str, ...]:
    return tuple(
        f"module load {shlex.quote(name)}"
        for name in config.environment.modules
    )


def _array_expression(indices: tuple[int, ...], maximum: int | None) -> str:
    if not indices:
        raise SlurmPlanError("cannot generate an array script with no selected shards")
    value = ",".join(str(index) for index in indices)
    return value if maximum is None else f"{value}%{maximum}"

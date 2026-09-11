from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from activevision_workbench.errors import SlurmConfigurationError

_SBATCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MEMORY = re.compile(r"^[1-9][0-9]*(?:K|M|G|T)$")
_TIME = re.compile(r"^(?:[0-9]+-)?[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]$")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+$")
_MODULE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+._/-]*$")
_MAIL_TYPES = frozenset(
    {"BEGIN", "END", "FAIL", "REQUEUE", "TIME_LIMIT", "ALL", "NONE"}
)
_ASSIGNMENT_STRATEGIES = frozenset(
    {"hash_bucket", "balanced", "balanced_by_image"}
)


@dataclass(frozen=True, slots=True)
class SlurmResources:
    nodes: int
    tasks_per_node: int
    cpus_per_task: int
    gpus_per_node: int
    gpu_type: str | None
    memory: str
    time_limit: str

    @classmethod
    def from_dict(cls, data: object) -> "SlurmResources":
        values = _mapping(
            data,
            "SlurmConfig.resources",
            {
                "nodes",
                "tasks_per_node",
                "cpus_per_task",
                "gpus_per_node",
                "gpu_type",
                "memory",
                "time_limit",
            },
        )
        nodes = _integer(values["nodes"], "resources.nodes", minimum=1)
        tasks = _integer(
            values["tasks_per_node"], "resources.tasks_per_node", minimum=1
        )
        if nodes != 1 or tasks != 1:
            raise SlurmConfigurationError(
                "resources.nodes and tasks_per_node must both be 1; AVRW shards "
                "are independent single-process jobs"
            )
        memory = _string(values["memory"], "resources.memory")
        if _MEMORY.fullmatch(memory) is None:
            raise SlurmConfigurationError(
                "resources.memory must use a positive Slurm size such as '32G'"
            )
        time_limit = _string(values["time_limit"], "resources.time_limit")
        if _TIME.fullmatch(time_limit) is None:
            raise SlurmConfigurationError(
                "resources.time_limit must use [days-]HH:MM:SS"
            )
        gpu_type = _optional_string(values["gpu_type"], "resources.gpu_type")
        if gpu_type is not None and _SBATCH_NAME.fullmatch(gpu_type) is None:
            raise SlurmConfigurationError(
                "resources.gpu_type contains unsafe characters"
            )
        gpus = _integer(
            values["gpus_per_node"], "resources.gpus_per_node", minimum=0
        )
        if gpu_type is not None and gpus == 0:
            raise SlurmConfigurationError(
                "resources.gpu_type requires gpus_per_node greater than zero"
            )
        return cls(
            nodes=nodes,
            tasks_per_node=tasks,
            cpus_per_task=_integer(
                values["cpus_per_task"], "resources.cpus_per_task", minimum=1
            ),
            gpus_per_node=gpus,
            gpu_type=gpu_type,
            memory=memory,
            time_limit=time_limit,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": self.nodes,
            "tasks_per_node": self.tasks_per_node,
            "cpus_per_task": self.cpus_per_task,
            "gpus_per_node": self.gpus_per_node,
            "gpu_type": self.gpu_type,
            "memory": self.memory,
            "time_limit": self.time_limit,
        }


@dataclass(frozen=True, slots=True)
class SlurmEnvironment:
    name: str
    modules: tuple[str, ...]
    launcher: tuple[str, ...]
    apptainer_image: str | None
    apptainer_gpu: bool
    bind_paths: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: object, base_dir: Path) -> "SlurmEnvironment":
        values = _mapping(
            data,
            "SlurmConfig.environment",
            {
                "name",
                "modules",
                "launcher",
                "apptainer_image",
                "apptainer_gpu",
                "bind_paths",
            },
        )
        modules = _strings(values["modules"], "environment.modules")
        for module in modules:
            if _MODULE.fullmatch(module) is None:
                raise SlurmConfigurationError(
                    f"environment.modules contains unsafe value {module!r}"
                )
        launcher = _strings(values["launcher"], "environment.launcher")
        for token in launcher:
            _safe_text(token, "environment.launcher")
        raw_image = _optional_string(
            values["apptainer_image"], "environment.apptainer_image"
        )
        image = None if raw_image is None else str(_resolve(raw_image, base_dir))
        if image is None and not launcher:
            raise SlurmConfigurationError(
                "environment requires a launcher or an Apptainer image"
            )
        if image is not None and launcher:
            raise SlurmConfigurationError(
                "environment.launcher and apptainer_image are mutually exclusive"
            )
        apptainer_gpu = _boolean(
            values["apptainer_gpu"], "environment.apptainer_gpu"
        )
        if apptainer_gpu and image is None:
            raise SlurmConfigurationError(
                "environment.apptainer_gpu requires an Apptainer image"
            )
        return cls(
            name=_string(values["name"], "environment.name"),
            modules=modules,
            launcher=launcher,
            apptainer_image=image,
            apptainer_gpu=apptainer_gpu,
            bind_paths=tuple(
                str(_resolve(path, base_dir))
                for path in _strings(values["bind_paths"], "environment.bind_paths")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "modules": list(self.modules),
            "launcher": list(self.launcher),
            "apptainer_image": self.apptainer_image,
            "apptainer_gpu": self.apptainer_gpu,
            "bind_paths": list(self.bind_paths),
        }


@dataclass(frozen=True, slots=True)
class SlurmConfig:
    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    schema_version: int
    account: str
    partition: str | None
    job_name: str
    shard_count: int
    max_parallel_tasks: int | None
    assignment_strategy: str
    resources: SlurmResources
    environment: SlurmEnvironment
    working_directory: str
    dataset_roots: tuple[str, ...]
    checkpoint_roots: tuple[str, ...]
    log_directory: str
    email: str | None
    mail_types: tuple[str, ...]
    requeue: bool
    max_retries: int
    signal_seconds: int
    dependency_job_ids: tuple[str, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> "SlurmConfig":
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as error:
            raise SlurmConfigurationError(
                f"cannot read Slurm configuration '{source}': {error}"
            ) from error
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as json_error:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as error:
                raise SlurmConfigurationError(
                    f"{source}: not strict JSON; install PyYAML for block YAML "
                    f"({json_error})"
                ) from error
            try:
                data = yaml.safe_load(text)
            except Exception as error:
                raise SlurmConfigurationError(
                    f"{source}: invalid YAML configuration: {error}"
                ) from error
        return cls.from_dict(data, base_dir=source.resolve().parent)

    @classmethod
    def from_dict(
        cls, data: object, *, base_dir: str | Path | None = None
    ) -> "SlurmConfig":
        root = Path.cwd() if base_dir is None else Path(base_dir)
        fields = _mapping(
            data,
            "SlurmConfig",
            {
                "schema_version",
                "account",
                "partition",
                "job_name",
                "shard_count",
                "max_parallel_tasks",
                "resources",
                "environment",
                "working_directory",
                "dataset_roots",
                "checkpoint_roots",
                "log_directory",
                "email",
                "mail_types",
                "requeue",
                "max_retries",
                "signal_seconds",
                "dependency_job_ids",
            },
            optional={"assignment_strategy"},
        )
        schema = _integer(fields["schema_version"], "schema_version", minimum=1)
        if schema != cls.CURRENT_SCHEMA_VERSION:
            raise SlurmConfigurationError(
                f"schema_version must be {cls.CURRENT_SCHEMA_VERSION}"
            )
        account = _safe_name(fields["account"], "account")
        partition = _optional_string(fields["partition"], "partition")
        if partition is not None and _SBATCH_NAME.fullmatch(partition) is None:
            raise SlurmConfigurationError("partition contains unsafe characters")
        job_name = _safe_name(fields["job_name"], "job_name")
        shard_count = _integer(
            fields["shard_count"], "shard_count", minimum=1, maximum=10000
        )
        maximum = _optional_integer(
            fields["max_parallel_tasks"], "max_parallel_tasks", minimum=1
        )
        if maximum is not None and maximum > shard_count:
            raise SlurmConfigurationError(
                "max_parallel_tasks must not exceed shard_count"
            )
        assignment_strategy = _string(
            fields.get("assignment_strategy", "hash_bucket"),
            "assignment_strategy",
        )
        if assignment_strategy not in _ASSIGNMENT_STRATEGIES:
            supported = ", ".join(sorted(_ASSIGNMENT_STRATEGIES))
            raise SlurmConfigurationError(
                f"assignment_strategy must be one of: {supported}"
            )
        email = _optional_string(fields["email"], "email")
        if email is not None and _EMAIL.fullmatch(email) is None:
            raise SlurmConfigurationError("email is not a safe email address")
        mail_types = tuple(
            value.upper()
            for value in _strings(fields["mail_types"], "mail_types")
        )
        unknown_mail = next(
            (value for value in mail_types if value not in _MAIL_TYPES), None
        )
        if unknown_mail is not None:
            raise SlurmConfigurationError(f"unsupported mail type {unknown_mail!r}")
        if mail_types and email is None:
            raise SlurmConfigurationError("mail_types requires email")
        dependencies = _strings(fields["dependency_job_ids"], "dependency_job_ids")
        if any(not value.isdigit() for value in dependencies):
            raise SlurmConfigurationError(
                "dependency_job_ids must contain numeric job IDs"
            )
        if len(dependencies) != len(set(dependencies)):
            raise SlurmConfigurationError(
                "dependency_job_ids must not contain duplicates"
            )
        return cls(
            schema_version=schema,
            account=account,
            partition=partition,
            job_name=job_name,
            shard_count=shard_count,
            max_parallel_tasks=maximum,
            assignment_strategy=assignment_strategy,
            resources=SlurmResources.from_dict(fields["resources"]),
            environment=SlurmEnvironment.from_dict(fields["environment"], root),
            working_directory=str(
                _resolve(
                    _string(fields["working_directory"], "working_directory"),
                    root,
                )
            ),
            dataset_roots=tuple(
                str(_resolve(path, root))
                for path in _strings(fields["dataset_roots"], "dataset_roots")
            ),
            checkpoint_roots=tuple(
                str(_resolve(path, root))
                for path in _strings(fields["checkpoint_roots"], "checkpoint_roots")
            ),
            log_directory=str(
                _resolve(_string(fields["log_directory"], "log_directory"), root)
            ),
            email=email,
            mail_types=mail_types,
            requeue=_boolean(fields["requeue"], "requeue"),
            max_retries=_integer(
                fields["max_retries"], "max_retries", minimum=0, maximum=20
            ),
            signal_seconds=_integer(
                fields["signal_seconds"],
                "signal_seconds",
                minimum=30,
                maximum=3600,
            ),
            dependency_job_ids=dependencies,
        )

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), separators=(",", ":"), sort_keys=True
        ).encode()
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema_version": self.schema_version,
            "account": self.account,
            "partition": self.partition,
            "job_name": self.job_name,
            "shard_count": self.shard_count,
            "max_parallel_tasks": self.max_parallel_tasks,
            "resources": self.resources.to_dict(),
            "environment": self.environment.to_dict(),
            "working_directory": self.working_directory,
            "dataset_roots": list(self.dataset_roots),
            "checkpoint_roots": list(self.checkpoint_roots),
            "log_directory": self.log_directory,
            "email": self.email,
            "mail_types": list(self.mail_types),
            "requeue": self.requeue,
            "max_retries": self.max_retries,
            "signal_seconds": self.signal_seconds,
            "dependency_job_ids": list(self.dependency_job_ids),
        }
        if self.assignment_strategy != "hash_bucket":
            result["assignment_strategy"] = self.assignment_strategy
        return result


def _mapping(
    data: object,
    field: str,
    required: set[str],
    *,
    optional: set[str] | None = None,
) -> Mapping[str, object]:
    if not isinstance(data, Mapping) or any(type(key) is not str for key in data):
        raise SlurmConfigurationError(f"{field} must be an object with string keys")
    keys = set(data)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - (optional or set()))
    if missing:
        raise SlurmConfigurationError(f"{field}.{missing[0]} is required")
    if unknown:
        raise SlurmConfigurationError(f"{field}.{unknown[0]} is unknown")
    return data


def _safe_text(value: str, field: str) -> None:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise SlurmConfigurationError(f"{field} contains unsafe control characters")


def _string(value: object, field: str) -> str:
    if type(value) is not str or value == "":
        raise SlurmConfigurationError(f"{field} must be a non-empty string")
    _safe_text(value, field)
    return value


def _optional_string(value: object, field: str) -> str | None:
    return None if value is None else _string(value, field)


def _safe_name(value: object, field: str) -> str:
    result = _string(value, field)
    if _SBATCH_NAME.fullmatch(result) is None:
        raise SlurmConfigurationError(f"{field} contains unsafe characters")
    return result


def _integer(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise SlurmConfigurationError(
            f"{field} must be an integer >= {minimum}{suffix}"
        )
    return value


def _optional_integer(value: object, field: str, *, minimum: int) -> int | None:
    return None if value is None else _integer(value, field, minimum=minimum)


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise SlurmConfigurationError(f"{field} must be a boolean")
    return value


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SlurmConfigurationError(f"{field} must be a list of strings")
    return tuple(_string(item, f"{field}[{index}]") for index, item in enumerate(value))


def _resolve(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (base_dir / path).resolve(strict=False)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value!r}")

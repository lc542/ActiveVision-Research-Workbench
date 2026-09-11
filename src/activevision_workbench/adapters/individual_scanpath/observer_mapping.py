
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from activevision_workbench.errors import ContractValidationError


@dataclass(frozen=True, slots=True)
class ObserverEntry:

    observer_id: str
    model_observer_index: int
    known_checkpoint_observer: bool


@dataclass(frozen=True, slots=True)
class ObserverMapping:

    dataset_id: str
    model_variant: str
    checkpoint_sha256: str
    identity_source: str
    entries: tuple[ObserverEntry, ...]
    sha256: str

    def index_for(self, observer_id: str) -> int:
        for entry in self.entries:
            if entry.observer_id == observer_id:
                return entry.model_observer_index
        raise ContractValidationError(
            "InferenceRequest.observer_id",
            f"unknown checkpoint observer '{observer_id}'",
        )

    @property
    def observer_ids(self) -> tuple[str, ...]:
        return tuple(entry.observer_id for entry in self.entries)


def load_observer_mapping(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_dataset_id: str,
    expected_variant: str,
    expected_checkpoint_sha256: str,
    expected_subject_count: int,
) -> ObserverMapping:

    if not path.is_file():
        raise ContractValidationError(
            "IndividualScanpathAdapterConfig.observer_mapping_path",
            f"mapping file does not exist: {path}",
        )
    digest = _sha256(path)
    expected = expected_fingerprint.removeprefix("sha256:").lower()
    if digest != expected:
        raise ContractValidationError(
            "IndividualScanpathAdapterConfig.observer_mapping_fingerprint",
            f"expected sha256:{expected}, found sha256:{digest}",
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractValidationError(
            "IndividualScanpathAdapterConfig.observer_mapping_path",
            f"cannot read strict JSON mapping: {error}",
        ) from error
    if not isinstance(raw, dict):
        raise _mapping_error("must contain a JSON object")
    expected_keys = {
        "schema_version",
        "dataset_id",
        "model_variant",
        "checkpoint_sha256",
        "identity_source",
        "observers",
    }
    _exact_keys(raw, expected_keys, "ObserverMapping")
    if raw["schema_version"] != 1:
        raise _mapping_error("schema_version must equal 1")
    for name in (
        "dataset_id",
        "model_variant",
        "checkpoint_sha256",
        "identity_source",
    ):
        if type(raw[name]) is not str or not raw[name]:
            raise _mapping_error(f"{name} must be a non-empty string")
    if raw["dataset_id"] != expected_dataset_id:
        raise _mapping_error(f"dataset_id must equal '{expected_dataset_id}'")
    if raw["model_variant"] != expected_variant:
        raise _mapping_error(f"model_variant must equal '{expected_variant}'")
    if raw["checkpoint_sha256"].removeprefix("sha256:").lower() != (
        expected_checkpoint_sha256
    ):
        raise _mapping_error("checkpoint_sha256 does not match the selected model")
    values = raw["observers"]
    if not isinstance(values, list) or len(values) != expected_subject_count:
        raise _mapping_error(
            f"observers must contain exactly {expected_subject_count} entries"
        )
    entries: list[ObserverEntry] = []
    ids: set[str] = set()
    indices: set[int] = set()
    for position, value in enumerate(values):
        label = f"ObserverMapping.observers[{position}]"
        if not isinstance(value, dict):
            raise _mapping_error(f"observers[{position}] must be an object")
        _exact_keys(
            value,
            {
                "observer_id",
                "model_observer_index",
                "known_checkpoint_observer",
            },
            label,
        )
        observer_id = value["observer_id"]
        index = value["model_observer_index"]
        known = value["known_checkpoint_observer"]
        if type(observer_id) is not str or not observer_id:
            raise _mapping_error(f"observers[{position}].observer_id is invalid")
        if type(index) is not int or index < 0:
            raise _mapping_error(
                f"observers[{position}].model_observer_index is invalid"
            )
        if type(known) is not bool or not known:
            raise _mapping_error(
                f"observers[{position}] must be a known checkpoint observer"
            )
        if observer_id in ids or index in indices:
            raise _mapping_error("observer IDs and model indices must be unique")
        ids.add(observer_id)
        indices.add(index)
        entries.append(ObserverEntry(observer_id, index, known))
    if indices != set(range(expected_subject_count)):
        raise _mapping_error(
            f"model indices must be exactly 0..{expected_subject_count - 1}"
        )
    entries.sort(key=lambda entry: entry.model_observer_index)
    return ObserverMapping(
        dataset_id=raw["dataset_id"],
        model_variant=raw["model_variant"],
        checkpoint_sha256=expected_checkpoint_sha256,
        identity_source=raw["identity_source"],
        entries=tuple(entries),
        sha256=digest,
    )


def _exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise _mapping_error(f"{label} missing field '{sorted(missing)[0]}'")
    if unknown:
        raise _mapping_error(f"{label} has unknown field '{sorted(unknown)[0]}'")


def _mapping_error(reason: str) -> ContractValidationError:
    return ContractValidationError(
        "IndividualScanpathAdapterConfig.observer_mapping_path", reason
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

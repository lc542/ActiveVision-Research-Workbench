
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

from activevision_workbench.contracts import InferenceRequest, PredictionRecord
from activevision_workbench.errors import ContractValidationError, SerializationError

RecordT = TypeVar("RecordT", InferenceRequest, PredictionRecord)


def _reject_json_constant(value: str) -> None:
    raise SerializationError(f"non-standard JSON numeric constant '{value}'")


def _dumps(data: object) -> str:
    try:
        return json.dumps(
            data,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise SerializationError(f"value is not strict JSON: {error}") from error


def _loads(text: str) -> object:
    if type(text) is not str:
        raise SerializationError("JSON input must be a string")
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise SerializationError(
            f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error


def dumps_request(request: InferenceRequest) -> str:

    if not isinstance(request, InferenceRequest):
        raise SerializationError("request must be an InferenceRequest")
    return _dumps(request.to_dict())


def loads_request(text: str) -> InferenceRequest:

    return InferenceRequest.from_dict(_loads(text))


def dumps_prediction(prediction: PredictionRecord) -> str:

    if not isinstance(prediction, PredictionRecord):
        raise SerializationError("prediction must be a PredictionRecord")
    return _dumps(prediction.to_dict())


def loads_prediction(text: str) -> PredictionRecord:

    return PredictionRecord.from_dict(_loads(text))


def _atomic_write_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[RecordT],
    serializer: Callable[[RecordT], str],
    *,
    overwrite: bool,
) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing JSONL file: {destination}"
        )
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"JSONL parent directory does not exist: {destination.parent}"
        )

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            for record in records:
                temporary.write(serializer(record))
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            try:
                os.link(temporary_name, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite existing JSONL file: {destination}"
                ) from error
            Path(temporary_name).unlink()
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def write_requests_jsonl(
    path: str | os.PathLike[str],
    requests: Iterable[InferenceRequest],
    *,
    overwrite: bool = False,
) -> None:

    _atomic_write_jsonl(path, requests, dumps_request, overwrite=overwrite)


def write_predictions_jsonl(
    path: str | os.PathLike[str],
    predictions: Iterable[PredictionRecord],
    *,
    overwrite: bool = False,
) -> None:

    _atomic_write_jsonl(path, predictions, dumps_prediction, overwrite=overwrite)


def _read_jsonl(
    path: str | os.PathLike[str],
    loader: Callable[[str], RecordT],
) -> tuple[RecordT, ...]:
    source = Path(path)
    records: list[RecordT] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip() == "":
                raise SerializationError(
                    f"{source}: line {line_number}: blank JSONL lines are not allowed"
                )
            try:
                records.append(loader(line))
            except (ContractValidationError, SerializationError) as error:
                raise SerializationError(
                    f"{source}: line {line_number}: {error}"
                ) from error
    return tuple(records)


def read_requests_jsonl(
    path: str | os.PathLike[str],
) -> tuple[InferenceRequest, ...]:

    return _read_jsonl(path, loads_request)


def read_predictions_jsonl(
    path: str | os.PathLike[str],
) -> tuple[PredictionRecord, ...]:

    return _read_jsonl(path, loads_prediction)


from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, TypeAlias

from activevision_workbench.errors import ContractValidationError

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]
JsonMapping: TypeAlias = Mapping[str, JsonValue]

_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")


def require_string(value: object, field: str, *, allow_empty: bool = False) -> str:

    if type(value) is not str:
        raise ContractValidationError(field, "must be a string")
    if not allow_empty and value == "":
        raise ContractValidationError(field, "must not be empty")
    return value


def optional_string(
    value: object,
    field: str,
    *,
    allow_empty: bool = True,
) -> str | None:

    if value is None:
        return None
    return require_string(value, field, allow_empty=allow_empty)


def require_int(value: object, field: str, *, minimum: int | None = None) -> int:

    if type(value) is not int:
        raise ContractValidationError(field, "must be an integer")
    if minimum is not None and value < minimum:
        raise ContractValidationError(field, f"must be >= {minimum}")
    return value


def optional_int(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
) -> int | None:

    if value is None:
        return None
    return require_int(value, field, minimum=minimum)


def require_finite_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:

    if type(value) not in (int, float):
        raise ContractValidationError(field, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(field, "must be finite")
    if minimum is not None and result < minimum:
        raise ContractValidationError(field, f"must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ContractValidationError(field, f"must be <= {maximum}")
    return result


def optional_finite_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:

    if value is None:
        return None
    return require_finite_number(
        value,
        field,
        minimum=minimum,
        maximum=maximum,
    )


def freeze_json_value(value: object, field: str) -> JsonValue:

    if value is None or type(value) in (str, int, bool):
        return value  # type: ignore[return-value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractValidationError(field, "must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ContractValidationError(field, "mapping keys must be strings")
            child = f"{field}.{key}" if key else f"{field}.<empty>"
            frozen[key] = freeze_json_value(item, child)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            freeze_json_value(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise ContractValidationError(field, "must contain only JSON-compatible values")


def freeze_json_mapping(value: object, field: str) -> JsonMapping:

    if not isinstance(value, Mapping):
        raise ContractValidationError(field, "must be a mapping")
    frozen = freeze_json_value(value, field)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
        raise AssertionError("mapping freeze did not return a mapping")
    return frozen


def freeze_namespaced_options(
    value: object,
    field: str,
    *,
    model_id: str,
) -> JsonMapping:

    frozen = freeze_json_mapping(value, field)
    for namespace, options in frozen.items():
        if not _NAMESPACE_PATTERN.fullmatch(namespace):
            raise ContractValidationError(
                f"{field}.{namespace}",
                "namespace must use lowercase letters, digits, '_', '-', or '.'",
            )
        if namespace != model_id and not namespace.startswith(f"{model_id}."):
            raise ContractValidationError(
                f"{field}.{namespace}",
                f"namespace must be '{model_id}' or begin with '{model_id}.'",
            )
        if not isinstance(options, Mapping):
            raise ContractValidationError(
                f"{field}.{namespace}",
                "namespace value must be a mapping",
            )
    return frozen


def thaw_json_value(value: JsonValue | object) -> object:

    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value


def validate_schema_version(value: object, field: str) -> int:

    version = require_int(value, field, minimum=1)
    if version != 1:
        raise ContractValidationError(field, "unsupported schema version; expected 1")
    return version


def validate_dict_keys(
    value: object,
    field: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:

    if not isinstance(value, Mapping):
        raise ContractValidationError(field, "must be a JSON object")
    non_string = [key for key in value if type(key) is not str]
    if non_string:
        raise ContractValidationError(field, "field names must be strings")
    keys = set(value)
    missing = sorted(required - keys)
    if missing:
        raise ContractValidationError(
            f"{field}.{missing[0]}",
            "required field is missing",
        )
    unknown = sorted(keys - required - optional)
    if unknown:
        raise ContractValidationError(
            f"{field}.{unknown[0]}",
            "unknown field under strict schema compatibility policy",
        )
    return value

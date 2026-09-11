
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

from activevision_workbench.contracts import FixationEvent
from activevision_workbench.errors import (
    CoordinateOutOfBoundsError,
    DatasetAnnotationError,
)


class CoordinateClippedWarning(UserWarning):
    pass


@dataclass(frozen=True, slots=True)
class CanonicalCoordinate:

    x_px: float
    y_px: float
    x_norm: float
    y_norm: float
    clipped: bool = False
    original_x_px: float | None = None
    original_y_px: float | None = None


def pixel_to_normalized(
    *,
    x_px: float,
    y_px: float,
    image_width: int,
    image_height: int,
    clip: bool = False,
) -> CanonicalCoordinate:

    width, height = _validate_dimensions(image_width, image_height)
    x_value = _finite(x_px, "x_px")
    y_value = _finite(y_px, "y_px")
    clipped_x = min(max(x_value, 0.0), float(width - 1))
    clipped_y = min(max(y_value, 0.0), float(height - 1))
    changed = clipped_x != x_value or clipped_y != y_value
    if changed and not clip:
        raise CoordinateOutOfBoundsError(
            f"pixel coordinate ({x_value}, {y_value}) is outside the "
            f"{width}x{height} center lattice [0,{width - 1}] x "
            f"[0,{height - 1}]"
        )
    if changed:
        warnings.warn(
            "source gaze coordinate was clipped to the image center lattice; "
            "the original pair remains in provenance",
            CoordinateClippedWarning,
            stacklevel=2,
        )
    return CanonicalCoordinate(
        x_px=clipped_x,
        y_px=clipped_y,
        x_norm=clipped_x / (width - 1),
        y_norm=clipped_y / (height - 1),
        clipped=changed,
        original_x_px=x_value if changed else None,
        original_y_px=y_value if changed else None,
    )


def normalized_to_pixel(
    *,
    x_norm: float,
    y_norm: float,
    image_width: int,
    image_height: int,
    clip: bool = False,
) -> CanonicalCoordinate:

    width, height = _validate_dimensions(image_width, image_height)
    x_value = _finite(x_norm, "x_norm")
    y_value = _finite(y_norm, "y_norm")
    clipped_x = min(max(x_value, 0.0), 1.0)
    clipped_y = min(max(y_value, 0.0), 1.0)
    changed = clipped_x != x_value or clipped_y != y_value
    if changed and not clip:
        raise CoordinateOutOfBoundsError(
            f"normalized coordinate ({x_value}, {y_value}) is outside [0,1]"
        )
    if changed:
        warnings.warn(
            "source normalized gaze coordinate was clipped to [0,1]",
            CoordinateClippedWarning,
            stacklevel=2,
        )
    return CanonicalCoordinate(
        x_px=clipped_x * (width - 1),
        y_px=clipped_y * (height - 1),
        x_norm=clipped_x,
        y_norm=clipped_y,
        clipped=changed,
        original_x_px=None,
        original_y_px=None,
    )


def source_fixation_from_pixel(
    *,
    x_px: float,
    y_px: float,
    image_width: int,
    image_height: int,
    sequence_index: int,
    timestamp_s: float | None = None,
    duration_s: float | None = None,
    native_timing: object | None = None,
    native_metadata: object | None = None,
    clip_out_of_bounds: bool = False,
) -> tuple[FixationEvent, str | None]:

    metadata = {} if native_metadata is None else _mapping_copy(native_metadata)
    timing = {} if native_timing is None else _mapping_copy(native_timing)
    try:
        coordinate = pixel_to_normalized(
            x_px=x_px,
            y_px=y_px,
            image_width=image_width,
            image_height=image_height,
            clip=clip_out_of_bounds,
        )
    except CoordinateOutOfBoundsError:
        metadata["source_coordinate"] = {"x_px": x_px, "y_px": y_px}
        metadata["coordinate_status"] = "out_of_bounds_preserved"
        event = FixationEvent(
            x_px=x_px,
            y_px=y_px,
            sequence_index=sequence_index,
            timestamp_s=timestamp_s,
            duration_s=duration_s,
            native_timing=timing,
            native_metadata=metadata,
        )
        return event, (
            "out-of-bounds source fixation preserved in pixel coordinates; "
            "normalized coordinates are unavailable"
        )

    if coordinate.clipped:
        metadata["source_coordinate"] = {
            "x_px": coordinate.original_x_px,
            "y_px": coordinate.original_y_px,
        }
        metadata["coordinate_status"] = "clipped_explicitly"
        warning = (
            "out-of-bounds source fixation was explicitly clipped; original "
            "coordinates remain in native metadata"
        )
    else:
        metadata["coordinate_status"] = "in_bounds"
        warning = None
    return (
        FixationEvent(
            x_px=coordinate.x_px,
            y_px=coordinate.y_px,
            x_norm=coordinate.x_norm,
            y_norm=coordinate.y_norm,
            sequence_index=sequence_index,
            timestamp_s=timestamp_s,
            duration_s=duration_s,
            native_timing=timing,
            native_metadata=metadata,
        ),
        warning,
    )


def _validate_dimensions(width: object, height: object) -> tuple[int, int]:
    if type(width) is not int or width < 2:
        raise DatasetAnnotationError("image_width must be an integer >= 2")
    if type(height) is not int or height < 2:
        raise DatasetAnnotationError("image_height must be an integer >= 2")
    return width, height


def _finite(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise DatasetAnnotationError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise DatasetAnnotationError(f"{field} must be finite")
    return result


def _mapping_copy(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetAnnotationError("native provenance must be a mapping")
    return dict(value)

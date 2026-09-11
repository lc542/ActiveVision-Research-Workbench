from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from activevision_workbench.errors import InspectionConfigurationError


class FigureFormat(str, Enum):
    PNG = "png"
    PDF = "pdf"


@dataclass(frozen=True, slots=True)
class PlotParameters:
    canvas_width_px: int = 1600
    panel_height_px: int = 720
    margin_px: int = 28
    title_height_px: int = 104
    footer_height_px: int = 180
    marker_radius_px: int = 8
    line_width_px: int = 4
    max_small_multiples: int = 12
    background_color: str = "#F7F7F5"
    predicted_colors: tuple[str, ...] = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
    )
    ground_truth_color: str = "#222222"

    def __post_init__(self) -> None:
        for name in (
            "canvas_width_px",
            "panel_height_px",
            "margin_px",
            "title_height_px",
            "footer_height_px",
            "marker_radius_px",
            "line_width_px",
            "max_small_multiples",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise InspectionConfigurationError(
                    f"plot.{name} must be a positive integer"
                )
        if self.canvas_width_px < 640:
            raise InspectionConfigurationError(
                "plot.canvas_width_px must be at least 640"
            )
        if self.panel_height_px < 320:
            raise InspectionConfigurationError(
                "plot.panel_height_px must be at least 320"
            )
        if not self.predicted_colors:
            raise InspectionConfigurationError(
                "plot.predicted_colors must not be empty"
            )
        for name in ("background_color", "ground_truth_color"):
            _color(getattr(self, name), f"plot.{name}")
        for index, value in enumerate(self.predicted_colors):
            _color(value, f"plot.predicted_colors[{index}]")

    def to_dict(self) -> dict[str, object]:
        return {
            "canvas_width_px": self.canvas_width_px,
            "panel_height_px": self.panel_height_px,
            "margin_px": self.margin_px,
            "title_height_px": self.title_height_px,
            "footer_height_px": self.footer_height_px,
            "marker_radius_px": self.marker_radius_px,
            "line_width_px": self.line_width_px,
            "max_small_multiples": self.max_small_multiples,
            "background_color": self.background_color,
            "predicted_colors": list(self.predicted_colors),
            "ground_truth_color": self.ground_truth_color,
        }


@dataclass(frozen=True, slots=True)
class InspectRequest:
    run_directory: Path
    item_id: str
    sample_id: str | None = None
    observer_id: str | None = None
    task_text: str | None = None
    figure_format: FigureFormat = FigureFormat.PNG
    output_directory: Path | None = None
    overwrite: bool = False
    plot: PlotParameters = field(default_factory=PlotParameters)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "run_directory", _path(self.run_directory, "run_directory")
        )
        object.__setattr__(self, "item_id", _text(self.item_id, "item_id"))
        for name in ("sample_id", "observer_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if self.task_text is not None and type(self.task_text) is not str:
            raise InspectionConfigurationError(
                "task_text must be a string or null"
            )
        _common(self.figure_format, self.output_directory, self.overwrite, self.plot)
        if self.output_directory is not None:
            object.__setattr__(
                self,
                "output_directory",
                _path(self.output_directory, "output_directory"),
            )


@dataclass(frozen=True, slots=True)
class CompareRequest:
    run_directories: tuple[Path, ...]
    item_id: str
    sample_id: str | None = None
    observer_id: str | None = None
    task_text: str | None = None
    figure_format: FigureFormat = FigureFormat.PNG
    output_directory: Path | None = None
    overwrite: bool = False
    plot: PlotParameters = field(default_factory=PlotParameters)

    def __post_init__(self) -> None:
        if not isinstance(self.run_directories, tuple) or len(self.run_directories) < 2:
            raise InspectionConfigurationError(
                "run_directories must contain at least two runs"
            )
        paths = tuple(
            _path(value, f"run_directories[{index}]")
            for index, value in enumerate(self.run_directories)
        )
        if len(paths) != len(set(paths)):
            raise InspectionConfigurationError(
                "run_directories must not contain duplicates"
            )
        object.__setattr__(self, "run_directories", paths)
        object.__setattr__(self, "item_id", _text(self.item_id, "item_id"))
        for name in ("sample_id", "observer_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if self.task_text is not None and type(self.task_text) is not str:
            raise InspectionConfigurationError(
                "task_text must be a string or null"
            )
        _common(self.figure_format, self.output_directory, self.overwrite, self.plot)
        if self.output_directory is not None:
            object.__setattr__(
                self,
                "output_directory",
                _path(self.output_directory, "output_directory"),
            )


def _common(
    figure_format: FigureFormat,
    output_directory: Path | None,
    overwrite: bool,
    plot: PlotParameters,
) -> None:
    if not isinstance(figure_format, FigureFormat):
        raise InspectionConfigurationError("figure_format must be png or pdf")
    if output_directory is not None and not isinstance(
        output_directory, (str, Path)
    ):
        raise InspectionConfigurationError(
            "output_directory must be a path or null"
        )
    if type(overwrite) is not bool:
        raise InspectionConfigurationError("overwrite must be a boolean")
    if not isinstance(plot, PlotParameters):
        raise InspectionConfigurationError("plot must be PlotParameters")


def _path(value: object, field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise InspectionConfigurationError(f"{field_name} must be a path")
    text = str(value)
    if not text:
        raise InspectionConfigurationError(f"{field_name} must not be empty")
    return Path(text).expanduser().resolve(strict=False)


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise InspectionConfigurationError(
            f"{field_name} must be a non-empty string"
        )
    return value


def _color(value: object, field_name: str) -> str:
    if type(value) is not str or len(value) != 7 or not value.startswith("#"):
        raise InspectionConfigurationError(
            f"{field_name} must be a #RRGGBB color"
        )
    try:
        int(value[1:], 16)
    except ValueError as error:
        raise InspectionConfigurationError(
            f"{field_name} must be a #RRGGBB color"
        ) from error
    return value

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from activevision_workbench.contracts import (
    Capability,
    CapabilitySet,
    FixationEvent,
    PredictionRecord,
)
from activevision_workbench.datasets import (
    COCOSearch18DatasetAdapter,
    DatasetAdapter,
    DatasetItem,
    GroundTruthScanpath,
    JsonDatasetAdapter,
    OSIEDatasetAdapter,
)
from activevision_workbench.errors import (
    InspectionConfigurationError,
    InspectionDataError,
    InspectionOutputError,
    RunConfigurationError,
    RunResumeError,
)
from activevision_workbench.inspection.config import (
    CompareRequest,
    FigureFormat,
    InspectRequest,
    PlotParameters,
)
from activevision_workbench.run_artifacts import RunDirectory, RunStatus
from activevision_workbench.run_config import RunConfig, RunPolicy


_MODEL_LABELS = {
    "scandiff": "ScanDiff",
    "tpp_gaze": "TPP-Gaze",
    "individualscanpath": "IndividualScanpath",
    "gazexplain": "GazeXplain",
}


@dataclass(frozen=True, slots=True)
class InspectionPlan:
    command: str
    figure_path: Path
    manifest_path: Path
    run_ids: tuple[str, ...]
    item_id: str
    prediction_count: int
    sample_ids: tuple[str, ...]
    observers: tuple[str, ...]
    tasks: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "command": self.command,
            "figure_path": str(self.figure_path),
            "manifest_path": str(self.manifest_path),
            "run_ids": list(self.run_ids),
            "item_id": self.item_id,
            "prediction_count": self.prediction_count,
            "sample_ids": list(self.sample_ids),
            "observers": list(self.observers),
            "tasks": list(self.tasks),
            "warnings": list(self.warnings),
            "dry_run": True,
            "inference_rerun": False,
        }


@dataclass(frozen=True, slots=True)
class InspectionResult:
    command: str
    figure_path: Path
    manifest_path: Path
    figure_sha256: str
    prediction_count: int
    warning_count: int


@dataclass(frozen=True, slots=True)
class _RunView:
    directory: RunDirectory
    config: RunConfig
    manifest: Mapping[str, object]
    predictions: tuple[PredictionRecord, ...]
    image_path: Path
    image_width: int
    image_height: int
    image_sha256: str
    ground_truth: tuple[GroundTruthScanpath, ...]
    metric_rows: tuple[Mapping[str, str], ...]
    density: Mapping[str, object] | None
    warnings: tuple[str, ...]

    @property
    def run_id(self) -> str:
        value = self.manifest.get("run_id")
        return str(value)

    @property
    def model_id(self) -> str:
        return self.predictions[0].model.model_id


class InspectionEngine:
    def dry_run_inspect(self, request: InspectRequest) -> InspectionPlan:
        view = self._prepare_view(
            request.run_directory,
            item_id=request.item_id,
            sample_id=request.sample_id,
            observer_id=request.observer_id,
            task_text=request.task_text,
        )
        figure, manifest = _inspect_paths(request, view.directory.path)
        return _plan("inspect", figure, manifest, (view,), request.item_id)

    def inspect(self, request: InspectRequest) -> InspectionResult:
        view = self._prepare_view(
            request.run_directory,
            item_id=request.item_id,
            sample_id=request.sample_id,
            observer_id=request.observer_id,
            task_text=request.task_text,
        )
        figure_path, manifest_path = _inspect_paths(request, view.directory.path)
        _preflight((figure_path, manifest_path), request.overwrite)
        image, render_warnings, panels = _render_inspection(view, request.plot)
        warnings = tuple(sorted(set((*view.warnings, *render_warnings))))
        _atomic_image(figure_path, image, request.figure_format)
        digest = _sha256_file(figure_path)
        manifest = _artifact_manifest(
            command="inspect",
            figure_path=figure_path,
            figure_sha256=digest,
            figure_format=request.figure_format,
            plot=request.plot,
            views=(view,),
            selectors={
                "item_id": request.item_id,
                "sample_id": request.sample_id,
                "observer_id": request.observer_id,
                "task_text": request.task_text,
            },
            panels=panels,
            warnings=warnings,
        )
        _atomic_json(manifest_path, manifest)
        return InspectionResult(
            command="inspect",
            figure_path=figure_path,
            manifest_path=manifest_path,
            figure_sha256=digest,
            prediction_count=len(view.predictions),
            warning_count=len(warnings),
        )

    def dry_run_compare(self, request: CompareRequest) -> InspectionPlan:
        views = self._comparison_views(request)
        figure, manifest = _compare_paths(request, views[0].directory.path.parent)
        return _plan("compare", figure, manifest, views, request.item_id)

    def compare(self, request: CompareRequest) -> InspectionResult:
        views = self._comparison_views(request)
        figure_path, manifest_path = _compare_paths(
            request, views[0].directory.path.parent
        )
        _preflight((figure_path, manifest_path), request.overwrite)
        image, render_warnings, panels = _render_comparison(views, request.plot)
        warnings = tuple(
            sorted(
                set(
                    (*render_warnings, *(warning for view in views for warning in view.warnings))
                )
            )
        )
        _atomic_image(figure_path, image, request.figure_format)
        digest = _sha256_file(figure_path)
        manifest = _artifact_manifest(
            command="compare",
            figure_path=figure_path,
            figure_sha256=digest,
            figure_format=request.figure_format,
            plot=request.plot,
            views=views,
            selectors={
                "item_id": request.item_id,
                "sample_id": request.sample_id,
                "observer_id": request.observer_id,
                "task_text": request.task_text,
            },
            panels=panels,
            warnings=warnings,
        )
        _atomic_json(manifest_path, manifest)
        return InspectionResult(
            command="compare",
            figure_path=figure_path,
            manifest_path=manifest_path,
            figure_sha256=digest,
            prediction_count=sum(len(view.predictions) for view in views),
            warning_count=len(warnings),
        )

    def compare_instructions(
        self,
        run_directory: str | Path,
        instructions: Sequence[str],
        *,
        output_directory: str | Path,
        figure_format: FigureFormat = FigureFormat.PNG,
        overwrite: bool = False,
        plot: PlotParameters | None = None,
    ) -> InspectionResult:
        exact = _instructions(instructions)
        directory = Path(run_directory).expanduser().resolve(strict=False)
        views = tuple(
            self._prepare_view(
                directory,
                item_id=f"instruction-{index:04d}",
                sample_id=None,
                observer_id=None,
                task_text=text,
            )
            for index, text in enumerate(exact)
        )
        _validate_compatible_images(views)
        parameters = PlotParameters() if plot is None else plot
        output = Path(output_directory).expanduser().resolve(strict=False)
        figure_path = output / f"compare-instructions.{figure_format.value}"
        manifest_path = output / "compare-instructions.manifest.json"
        _preflight((figure_path, manifest_path), overwrite)
        image, render_warnings, panels = _render_comparison(views, parameters)
        warnings = tuple(
            sorted(
                set(
                    (*render_warnings, *(warning for view in views for warning in view.warnings))
                )
            )
        )
        _atomic_image(figure_path, image, figure_format)
        digest = _sha256_file(figure_path)
        manifest = _artifact_manifest(
            command="compare-instructions",
            figure_path=figure_path,
            figure_sha256=digest,
            figure_format=figure_format,
            plot=parameters,
            views=views,
            selectors={"instructions": list(exact)},
            panels=panels,
            warnings=warnings,
        )
        _atomic_json(manifest_path, manifest)
        return InspectionResult(
            command="compare-instructions",
            figure_path=figure_path,
            manifest_path=manifest_path,
            figure_sha256=digest,
            prediction_count=sum(len(view.predictions) for view in views),
            warning_count=len(warnings),
        )

    def _comparison_views(self, request: CompareRequest) -> tuple[_RunView, ...]:
        views = tuple(
            self._prepare_view(
                directory,
                item_id=request.item_id,
                sample_id=request.sample_id,
                observer_id=request.observer_id,
                task_text=request.task_text,
            )
            for directory in request.run_directories
        )
        _validate_compatible_images(views)
        return views

    def _prepare_view(
        self,
        run_directory: Path,
        *,
        item_id: str,
        sample_id: str | None,
        observer_id: str | None,
        task_text: str | None,
    ) -> _RunView:
        try:
            directory = RunDirectory.open(run_directory)
            manifest = directory.read_manifest()
            config = RunConfig.from_file(directory.path / RunDirectory.CONFIG_NAME)
            predictions = directory.read_predictions()
        except (RunResumeError, RunConfigurationError, OSError) as error:
            raise InspectionDataError(str(error)) from error
        status = manifest.get("status")
        valid_statuses = {
            RunStatus.COMPLETED.value,
            RunStatus.COMPLETED_WITH_ITEM_FAILURES.value,
        }
        if status not in valid_statuses:
            raise InspectionDataError(
                f"run '{directory.path}' is not complete; status is {status!r}"
            )
        selected, alias_used = _select_predictions(
            predictions,
            item_id=item_id,
            sample_id=sample_id,
            observer_id=observer_id,
            task_text=task_text,
        )
        request_map = {request.request_id: request for request in config.requests()}
        try:
            requests = tuple(request_map[prediction.request_id] for prediction in selected)
        except KeyError as error:
            raise InspectionDataError(
                f"prediction request '{error.args[0]}' is absent from stored run configuration"
            ) from error
        image_refs = {request.image_ref for request in requests}
        if len(image_refs) != 1:
            raise InspectionDataError(
                "selected prediction records do not reference exactly one source image"
            )
        image_path = _local_image_path(next(iter(image_refs)), directory.path)
        image_width, image_height = _image_dimensions(image_path)
        warnings: list[str] = []
        expected_dimensions = {
            (request.image_width, request.image_height) for request in requests
        }
        if len(expected_dimensions) != 1:
            raise InspectionDataError(
                "selected requests contain inconsistent configured image dimensions"
            )
        expected_width, expected_height = next(iter(expected_dimensions))
        if expected_width is not None and (
            expected_width != image_width or expected_height != image_height
        ):
            warnings.append(
                "stored request dimensions differ from the decoded image; "
                "plotting uses decoded image dimensions"
            )
        if alias_used:
            warnings.append(
                f"item selector '{item_id}' expanded to observer-qualified run item IDs"
            )
        if status == RunStatus.COMPLETED_WITH_ITEM_FAILURES.value:
            warnings.append("source run completed with item failures")
        ground_truth, ground_truth_warnings = _ground_truth(config, selected)
        warnings.extend(ground_truth_warnings)
        for prediction in selected:
            warnings.extend(prediction.warnings)
            for index, fixation in enumerate(prediction.fixations):
                x, y, converted = _fixation_xy(fixation, image_width, image_height)
                if converted:
                    warnings.append(
                        "normalized coordinates were converted with x*(width-1), y*(height-1)"
                    )
                if x < 0 or y < 0 or x > image_width - 1 or y > image_height - 1:
                    warnings.append(
                        f"sample '{prediction.sample_id}' fixation {index} is out "
                        "of bounds and is clipped only for display"
                    )
        metric_rows, metric_warnings = _metric_rows(directory.path, selected)
        warnings.extend(metric_warnings)
        density, density_warnings = _density(directory.path, metric_rows)
        warnings.extend(density_warnings)
        return _RunView(
            directory=directory,
            config=config,
            manifest=manifest,
            predictions=selected,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            image_sha256=_sha256_file(image_path),
            ground_truth=ground_truth,
            metric_rows=metric_rows,
            density=density,
            warnings=tuple(sorted(set(warnings))),
        )


def build_instruction_run_config(
    base: RunConfig,
    *,
    model_id: str,
    image_path: str | Path,
    instructions: Sequence[str],
    output_root: str | Path,
    run_id: str | None = None,
) -> RunConfig:
    if base.model_id != model_id:
        raise InspectionConfigurationError(
            f"--model '{model_id}' does not match base config model '{base.model_id}'"
        )
    exact = _instructions(instructions)
    image = Path(image_path).expanduser().resolve(strict=False)
    width, height = _image_dimensions(image)
    output = Path(output_root).expanduser().resolve(strict=False)
    digest_input = json.dumps(
        {
            "base_configuration_hash": base.configuration_hash,
            "image_sha256": _sha256_file(image),
            "instructions": list(exact),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    stable_run_id = run_id or (
        f"instructions-{model_id}-{hashlib.sha256(digest_input).hexdigest()[:12]}"
    )
    observer_id = base.observer_id
    observer_metadata: Mapping[str, object] = base.observer_metadata
    if observer_id is None:
        observer_items = {
            item.observer_id: item.observer_metadata
            for item in base.selected_items
            if item.observer_id is not None
        }
        if len(observer_items) == 1:
            observer_id, observer_metadata = next(iter(observer_items.items()))
    values = base.to_dict()
    values["run_id"] = stable_run_id
    values["dataset"] = {
        "id": "instruction_experiment",
        "version": "1",
        "split": "experiment",
        "root": None,
        "root_fingerprint": None,
        "items": [
            {
                "item_id": f"instruction-{index:04d}",
                "image_ref": str(image),
                "image_width": width,
                "image_height": height,
                "task_text": text,
                "target_description": None,
                "observer_id": observer_id,
                "observer_metadata": dict(observer_metadata),
            }
            for index, text in enumerate(exact)
        ],
        "item_subset": None,
    }
    request = dict(values["request"])  # type: ignore[arg-type]
    request["task_text"] = None
    request["target_description"] = None
    request["observer_id"] = None
    request["observer_metadata"] = {}
    values["request"] = request
    values["output_root"] = str(output)
    values["run_policy"] = RunPolicy.CREATE.value
    tags = list(values["tags"])  # type: ignore[arg-type]
    if "instruction-comparison" not in tags:
        tags.append("instruction-comparison")
    values["tags"] = tags
    values["notes"] = "Exact task-conditioning instructions are stored per run item."
    return RunConfig.from_dict(values, base_dir=Path.cwd())


def validate_instruction_capability(capabilities: CapabilitySet) -> None:
    if not capabilities.supports(Capability.INSTRUCTION_CONDITIONED):
        raise InspectionConfigurationError(
            "selected adapter does not declare instruction-conditioned capability"
        )


def _plan(
    command: str,
    figure: Path,
    manifest: Path,
    views: tuple[_RunView, ...],
    item_id: str,
) -> InspectionPlan:
    return InspectionPlan(
        command=command,
        figure_path=figure,
        manifest_path=manifest,
        run_ids=tuple(view.run_id for view in views),
        item_id=item_id,
        prediction_count=sum(len(view.predictions) for view in views),
        sample_ids=tuple(
            sorted({prediction.sample_id for view in views for prediction in view.predictions})
        ),
        observers=tuple(
            sorted(
                {
                    prediction.observer_id
                    for view in views
                    for prediction in view.predictions
                    if prediction.observer_id is not None
                }
            )
        ),
        tasks=tuple(
            sorted(
                {
                    prediction.task_text
                    for view in views
                    for prediction in view.predictions
                    if prediction.task_text is not None
                }
            )
        ),
        warnings=tuple(
            sorted({warning for view in views for warning in view.warnings})
        ),
    )


def _select_predictions(
    predictions: Sequence[PredictionRecord],
    *,
    item_id: str,
    sample_id: str | None,
    observer_id: str | None,
    task_text: str | None,
) -> tuple[tuple[PredictionRecord, ...], bool]:
    exact = tuple(prediction for prediction in predictions if prediction.item_id == item_id)
    alias_used = False
    candidates = exact
    if not candidates:
        candidates = tuple(
            prediction
            for prediction in predictions
            if prediction.item_id.startswith(f"{item_id}:")
            and prediction.observer_id is not None
            and prediction.item_id == f"{item_id}:{prediction.observer_id}"
        )
        alias_used = bool(candidates)
    if sample_id is not None:
        candidates = tuple(
            prediction for prediction in candidates if prediction.sample_id == sample_id
        )
    if observer_id is not None:
        candidates = tuple(
            prediction for prediction in candidates if prediction.observer_id == observer_id
        )
    if task_text is not None:
        candidates = tuple(
            prediction for prediction in candidates if prediction.task_text == task_text
        )
    if not candidates:
        filters = {
            "item_id": item_id,
            "sample_id": sample_id,
            "observer_id": observer_id,
            "task_text": task_text,
        }
        raise InspectionDataError(
            "no prediction record matches selectors "
            + json.dumps(filters, ensure_ascii=False, sort_keys=True)
        )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda value: (
                value.item_id,
                value.observer_id or "",
                value.task_text or "",
                value.sample_index,
                value.sample_id,
            ),
        )
    )
    return ordered, alias_used


def _ground_truth(
    config: RunConfig,
    predictions: tuple[PredictionRecord, ...],
) -> tuple[tuple[GroundTruthScanpath, ...], tuple[str, ...]]:
    if config.dataset_root is None:
        return (), ("ground truth is unavailable because the run has no dataset root",)
    try:
        dataset = _dataset_adapter(config)
    except Exception as error:
        return (), (f"ground truth adapter could not be opened: {error}",)
    item: DatasetItem | None = None
    errors: list[str] = []
    for candidate in _ground_truth_candidates(predictions):
        try:
            item = dataset.get_item(candidate, split=config.dataset_split)
            break
        except Exception as error:
            errors.append(str(error))
    if item is None:
        return (), ("ground truth item could not be loaded: " + "; ".join(errors),)
    observers = {prediction.observer_id for prediction in predictions if prediction.observer_id}
    tasks = {prediction.task_text for prediction in predictions if prediction.task_text is not None}
    scanpaths = item.scanpaths
    if observers:
        matched = tuple(path for path in scanpaths if path.observer_id in observers)
        if matched:
            scanpaths = matched
    if tasks:
        matched = tuple(
            path
            for path in scanpaths
            if (path.task_text if path.task_text is not None else item.task_text) in tasks
        )
        if matched:
            scanpaths = matched
    warnings = list(item.warnings)
    for path in scanpaths:
        warnings.extend(path.warnings)
    return tuple(sorted(scanpaths, key=lambda value: value.scanpath_id)), tuple(warnings)


def _dataset_adapter(config: RunConfig) -> DatasetAdapter:
    assert config.dataset_root is not None
    if config.dataset_id == "osie":
        return OSIEDatasetAdapter(
            config.dataset_root,
            dataset_version=config.dataset_version or "source-release-unversioned",
        )
    if config.dataset_id == "coco_search18":
        return COCOSearch18DatasetAdapter(
            config.dataset_root,
            dataset_version=config.dataset_version or "1.0-target-present",
        )
    return JsonDatasetAdapter(config.dataset_root)


def _ground_truth_candidates(
    predictions: tuple[PredictionRecord, ...],
) -> tuple[str, ...]:
    candidates: list[str] = []
    for prediction in predictions:
        candidates.append(prediction.item_id)
        if prediction.observer_id is not None and prediction.item_id.endswith(
            f":{prediction.observer_id}"
        ):
            candidates.append(prediction.item_id[: -(len(prediction.observer_id) + 1)])
    return tuple(dict.fromkeys(candidates))


def _metric_rows(
    run_directory: Path,
    predictions: tuple[PredictionRecord, ...],
) -> tuple[tuple[Mapping[str, str], ...], tuple[str, ...]]:
    metrics = run_directory / "metrics"
    item_ids = {prediction.item_id for prediction in predictions}
    request_ids = {prediction.request_id for prediction in predictions}
    sample_ids = {prediction.sample_id for prediction in predictions}
    rows: list[Mapping[str, str]] = []
    warnings: list[str] = []
    for name in ("per_item.csv", "per_sample.csv"):
        path = metrics / name
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("item_id") not in item_ids:
                        continue
                    if row.get("request_id") and row.get("request_id") not in request_ids:
                        continue
                    if name == "per_sample.csv" and row.get("sample_id") not in sample_ids:
                        continue
                    rows.append(dict(sorted(row.items())))
        except (OSError, csv.Error) as error:
            warnings.append(f"evaluation rows could not be read from {name}: {error}")
    return (
        tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.get("metric_name", ""),
                    row.get("sample_id", ""),
                    row.get("observer_id", ""),
                ),
            )
        ),
        tuple(warnings),
    )


def _density(
    run_directory: Path,
    rows: tuple[Mapping[str, str], ...],
) -> tuple[Mapping[str, object] | None, tuple[str, ...]]:
    references = sorted(
        {
            row.get("artifact_reference", "")
            for row in rows
            if row.get("metric_name") == "fixation_density_kde"
            and row.get("artifact_reference")
        }
    )
    if not references:
        return None, ()
    reference = references[0]
    path = Path(reference)
    if not path.is_absolute():
        path = run_directory / "metrics" / path
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, (f"KDE artifact could not be read: {error}",)
    if not isinstance(value, Mapping) or value.get("metric_name") != "fixation_density_kde":
        return None, ("KDE artifact has an incompatible schema",)
    return value, ()


def _validate_compatible_images(views: tuple[_RunView, ...]) -> None:
    datasets = {
        (view.predictions[0].dataset_id, view.predictions[0].dataset_split)
        for view in views
    }
    if len(datasets) != 1:
        raise InspectionDataError(
            "selected runs have incompatible dataset identity or split"
        )
    identities = {
        (view.image_width, view.image_height, view.image_sha256) for view in views
    }
    if len(identities) != 1:
        details = ", ".join(
            f"{view.run_id}={view.image_width}x{view.image_height},sha256:{view.image_sha256[:12]}"
            for view in views
        )
        raise InspectionDataError(
            "selected runs do not reference the same image bytes and dimensions: "
            + details
        )


def _render_inspection(
    view: _RunView, plot: PlotParameters
) -> tuple[Any, tuple[str, ...], tuple[Mapping[str, object], ...]]:
    Image, ImageDraw, ImageFont = _pillow()
    source = _open_image(view.image_path)
    observers = {
        prediction.observer_id
        for prediction in view.predictions
        if prediction.observer_id is not None
    }
    small = (
        view.model_id == "scandiff" and len(view.predictions) > 1
    ) or (
        view.model_id == "individualscanpath" and len(observers) > 1
    )
    shown = min(len(view.predictions), plot.max_small_multiples) if small else 0
    columns = 3
    rows = math.ceil(shown / columns) if shown else 0
    small_height = rows * 300
    timeline_height = 230 if view.model_id == "tpp_gaze" else 0
    height = (
        plot.title_height_px
        + plot.panel_height_px
        + small_height
        + timeline_height
        + plot.footer_height_px
        + plot.margin_px * 4
    )
    canvas = Image.new("RGB", (plot.canvas_width_px, height), plot.background_color)
    draw = ImageDraw.Draw(canvas)
    fonts = _fonts(ImageFont)
    title = (
        f"{_model_label(view.model_id)} prediction | "
        f"{view.predictions[0].item_id}"
    )
    draw.text((plot.margin_px, plot.margin_px), title, fill="#111111", font=fonts["title"])
    subtitle = _selector_label(view.predictions)
    draw.text(
        (plot.margin_px, plot.margin_px + 42),
        subtitle,
        fill="#333333",
        font=fonts["body"],
    )
    top = plot.margin_px + plot.title_height_px
    box = (
        plot.margin_px,
        top,
        plot.canvas_width_px - plot.margin_px,
        top + plot.panel_height_px,
    )
    render_warnings = list(
        _draw_scanpath_panel(
            canvas,
            draw,
            source,
            box,
            view,
            view.predictions,
            plot,
            fonts,
            density=view.density if view.model_id == "scandiff" else None,
        )
    )
    panels: list[Mapping[str, object]] = [
        {
            "kind": "overlay",
            "run_id": view.run_id,
            "sample_ids": [prediction.sample_id for prediction in view.predictions],
            "ground_truth_scanpath_ids": [path.scanpath_id for path in view.ground_truth],
            "kde_rendered": view.density is not None and view.model_id == "scandiff",
        }
    ]
    current = box[3] + plot.margin_px
    if shown:
        tile_width = (plot.canvas_width_px - plot.margin_px * (columns + 1)) // columns
        for index, prediction in enumerate(view.predictions[:shown]):
            row, column = divmod(index, columns)
            left = plot.margin_px + column * (tile_width + plot.margin_px)
            tile = (left, current + row * 300, left + tile_width, current + row * 300 + 270)
            render_warnings.extend(
                _draw_scanpath_panel(
                    canvas,
                    draw,
                    source,
                    tile,
                    view,
                    (prediction,),
                    plot,
                    fonts,
                    density=None,
                    title=(
                        f"observer {prediction.observer_id} | "
                        f"sample {prediction.sample_index + 1} | "
                        f"seed {prediction.seed}"
                        if prediction.observer_id is not None
                        else (
                            f"sample {prediction.sample_index + 1} | "
                            f"seed {prediction.seed}"
                        )
                    ),
                )
            )
            panels.append(
                {
                    "kind": (
                        "observer_small_multiple"
                        if view.model_id == "individualscanpath"
                        else "sample_small_multiple"
                    ),
                    "run_id": view.run_id,
                    "sample_ids": [prediction.sample_id],
                }
            )
        if shown < len(view.predictions):
            render_warnings.append(
                f"small multiples show {shown} of {len(view.predictions)} samples; "
                "the overlay includes every sample"
            )
        current += small_height
    if timeline_height:
        timeline_box = (
            plot.margin_px,
            current,
            plot.canvas_width_px - plot.margin_px,
            current + timeline_height,
        )
        _draw_timeline(draw, timeline_box, view, fonts, plot)
        panels.append(
            {
                "kind": "continuous_time_timeline",
                "run_id": view.run_id,
                "sample_ids": [prediction.sample_id for prediction in view.predictions],
                "unit": "seconds",
            }
        )
        current += timeline_height + plot.margin_px
    _draw_footer(draw, (plot.margin_px, current), view, fonts, plot.canvas_width_px)
    return canvas, tuple(render_warnings), tuple(panels)


def _render_comparison(
    views: tuple[_RunView, ...], plot: PlotParameters
) -> tuple[Any, tuple[str, ...], tuple[Mapping[str, object], ...]]:
    Image, ImageDraw, ImageFont = _pillow()
    columns = 2 if len(views) > 1 else 1
    rows = math.ceil(len(views) / columns)
    panel_width = (plot.canvas_width_px - plot.margin_px * (columns + 1)) // columns
    panel_height = 700
    height = plot.title_height_px + rows * (panel_height + plot.margin_px) + plot.footer_height_px
    canvas = Image.new("RGB", (plot.canvas_width_px, height), plot.background_color)
    draw = ImageDraw.Draw(canvas)
    fonts = _fonts(ImageFont)
    title = f"Model prediction comparison | {views[0].predictions[0].item_id}"
    draw.text((plot.margin_px, plot.margin_px), title, fill="#111111", font=fonts["title"])
    draw.text(
        (plot.margin_px, plot.margin_px + 42),
        f"{views[0].image_width}x{views[0].image_height} pixels | "
        f"{len(views)} model outputs",
        fill="#333333",
        font=fonts["body"],
    )
    warnings: list[str] = []
    panels: list[Mapping[str, object]] = []
    top = plot.title_height_px
    for index, view in enumerate(views):
        row, column = divmod(index, columns)
        left = plot.margin_px + column * (panel_width + plot.margin_px)
        box = (
            left,
            top + row * (panel_height + plot.margin_px),
            left + panel_width,
            top + row * (panel_height + plot.margin_px) + panel_height,
        )
        source = _open_image(view.image_path)
        plot_box = (box[0], box[1], box[2], box[3] - 160)
        warnings.extend(
            _draw_scanpath_panel(
                canvas,
                draw,
                source,
                plot_box,
                view,
                view.predictions,
                plot,
                fonts,
                density=view.density if view.model_id == "scandiff" else None,
                title=_model_label(view.model_id),
            )
        )
        _draw_panel_details(
            draw,
            (box[0], box[3] - 152),
            view,
            fonts,
            max(36, panel_width // 9),
        )
        panels.append(
            {
                "kind": "selected_run",
                "run_id": view.run_id,
                "model_id": view.model_id,
                "sample_ids": [prediction.sample_id for prediction in view.predictions],
                "observer_ids": sorted(
                    {
                        prediction.observer_id
                        for prediction in view.predictions
                        if prediction.observer_id is not None
                    }
                ),
                "task_texts": sorted(
                    {
                        prediction.task_text
                        for prediction in view.predictions
                        if prediction.task_text is not None
                    }
                ),
                "metric_row_count": len(view.metric_rows),
            }
        )
    footer_y = top + rows * (panel_height + plot.margin_px)
    draw.text(
        (plot.margin_px, footer_y),
        "Coordinates: canonical pixel centers; normalized values use x*(width-1), y*(height-1).",
        fill="#222222",
        font=fonts["body"],
    )
    draw.text(
        (plot.margin_px, footer_y + 30),
        "All selected stochastic samples are overlaid.",
        fill="#222222",
        font=fonts["body"],
    )
    return canvas, tuple(warnings), tuple(panels)


def _draw_scanpath_panel(
    canvas: Any,
    draw: Any,
    source: Any,
    box: tuple[int, int, int, int],
    view: _RunView,
    predictions: tuple[PredictionRecord, ...],
    plot: PlotParameters,
    fonts: Mapping[str, Any],
    *,
    density: Mapping[str, object] | None,
    title: str | None = None,
) -> tuple[str, ...]:
    left, top, right, bottom = box
    title_space = 34 if title is not None else 0
    image_box = (left, top + title_space, right, bottom)
    if title is not None:
        draw.text((left, top + 4), title, fill="#111111", font=fonts["body"])
    placed, transform = _place_image(canvas, source, image_box)
    draw.rectangle(placed, outline="#555555", width=1)
    if density is not None:
        _draw_density(canvas, density, placed)
    warnings: list[str] = []
    for path in view.ground_truth:
        points, current = _display_points(
            path.fixations, view.image_width, view.image_height, transform
        )
        warnings.extend(current)
        if len(points) > 1:
            draw.line(points, fill=plot.ground_truth_color, width=max(2, plot.line_width_px - 1))
        for index, point in enumerate(points):
            radius = max(3, plot.marker_radius_px - 2)
            _circle(draw, point, radius, "#FFFFFF", plot.ground_truth_color, 2)
            draw.text(
                (point[0] + radius, point[1] - radius),
                f"G{index + 1}",
                fill=plot.ground_truth_color,
                font=fonts["small"],
            )
    for sample_index, prediction in enumerate(predictions):
        points, current = _display_points(
            prediction.fixations, view.image_width, view.image_height, transform
        )
        warnings.extend(current)
        color = plot.predicted_colors[sample_index % len(plot.predicted_colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=plot.line_width_px)
        durations = [fixation.duration_s for fixation in prediction.fixations]
        maximum_duration = max((value for value in durations if value is not None), default=None)
        temporal = view.model_id == "tpp_gaze"
        for index, point in enumerate(points):
            radius = plot.marker_radius_px
            duration = durations[index]
            if duration is not None and maximum_duration and maximum_duration > 0:
                radius = round(radius * (0.75 + duration / maximum_duration))
            fill = _time_color(index, len(points)) if temporal else color
            _circle(draw, point, radius, fill, "#FFFFFF", 2)
            draw.text(
                (point[0] + radius + 1, point[1] - radius),
                str(index + 1),
                fill="#111111",
                font=fonts["small"],
            )
    return tuple(warnings)


def _place_image(
    canvas: Any,
    source: Any,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float]]:
    Image, _, _ = _pillow()
    left, top, right, bottom = box
    available_width = max(1, right - left)
    available_height = max(1, bottom - top)
    scale = min(available_width / source.width, available_height / source.height)
    width = max(1, round(source.width * scale))
    height = max(1, round(source.height * scale))
    x = left + (available_width - width) // 2
    y = top + (available_height - height) // 2
    resized = source.resize((width, height), Image.Resampling.LANCZOS)
    canvas.paste(resized, (x, y))
    return (x, y, x + width, y + height), (x, y, width / source.width, height / source.height)


def _draw_density(
    canvas: Any,
    density: Mapping[str, object],
    placed: tuple[int, int, int, int],
) -> None:
    Image, _, _ = _pillow()
    raw = density.get("density")
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], list):
        return
    height = len(raw)
    width = len(raw[0])
    values = [float(value) for row in raw for value in row]
    maximum = max(values, default=0.0)
    if maximum <= 0:
        return
    pixels = []
    for value in values:
        ratio = min(max(value / maximum, 0.0), 1.0)
        pixels.append(
            (
                round(255 * ratio),
                round(90 * (1 - ratio)),
                round(255 * (1 - ratio)),
                round(170 * ratio),
            )
        )
    heat = Image.new("RGBA", (width, height))
    heat.putdata(pixels)
    target_width = placed[2] - placed[0]
    target_height = placed[3] - placed[1]
    heat = heat.resize((target_width, target_height), Image.Resampling.BILINEAR)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(heat, (placed[0], placed[1]), heat)
    combined = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    canvas.paste(combined)


def _draw_timeline(
    draw: Any,
    box: tuple[int, int, int, int],
    view: _RunView,
    fonts: Mapping[str, Any],
    plot: PlotParameters,
) -> None:
    left, top, right, bottom = box
    draw.text((left, top), "Continuous-time timeline (seconds)", fill="#111111", font=fonts["body"])
    series = []
    for prediction in view.predictions:
        times = _times(prediction)
        series.append((prediction, times))
    maximum = max((max(times, default=0.0) for _, times in series), default=1.0)
    maximum = max(maximum, 1e-9)
    axis_left = left + 190
    axis_right = right - 30
    row_height = max(30, (bottom - top - 50) // max(1, len(series)))
    for row, (prediction, times) in enumerate(series):
        y = top + 45 + row * row_height
        draw.text(
            (left, y - 9),
            f"sample {prediction.sample_index + 1}",
            fill="#222222",
            font=fonts["small"],
        )
        draw.line((axis_left, y, axis_right, y), fill="#777777", width=1)
        for index, value in enumerate(times):
            x = axis_left + (axis_right - axis_left) * value / maximum
            radius = max(4, plot.marker_radius_px - 2)
            _circle(draw, (x, y), radius, _time_color(index, len(times)), "#FFFFFF", 1)
    draw.text((axis_left, bottom - 24), "0", fill="#333333", font=fonts["small"])
    draw.text(
        (axis_right - 80, bottom - 24),
        f"{maximum:.3f} s",
        fill="#333333",
        font=fonts["small"],
    )


def _draw_footer(
    draw: Any,
    origin: tuple[int, int],
    view: _RunView,
    fonts: Mapping[str, Any],
    width: int,
) -> None:
    x, y = origin
    lines = [
        "Coordinates: canonical pixel centers; normalized values use x*(width-1), y*(height-1).",
        "Legend: GT = ground truth; colored paths = predictions; numbers = fixation order.",
    ]
    stopping = sorted(
        {
            prediction.stopping_reason.value
            for prediction in view.predictions
            if prediction.stopping_reason is not None
        }
    )
    if stopping:
        lines.append("Stopping reason(s): " + ", ".join(stopping))
    if view.model_id == "tpp_gaze":
        lines.append(
            "Temporal unit: seconds; configured horizon: "
            f"{view.config.time_horizon_s!r} s"
        )
    tasks = sorted(
        {
            prediction.task_text
            for prediction in view.predictions
            if prediction.task_text is not None
        }
    )
    for task in tasks:
        lines.append("Exact task: " + task)
    explanations = [
        prediction.explanation_text
        for prediction in view.predictions
        if prediction.explanation_text is not None
    ]
    for explanation in explanations:
        lines.append("Explanation: " + explanation)
    metric_summaries = _metric_summary(view.metric_rows)
    if metric_summaries:
        lines.append("Evaluation: " + "; ".join(metric_summaries))
    warning_text = " | ".join(view.warnings)
    if warning_text:
        lines.append("Warnings: " + warning_text)
    line_y = y
    for line in lines:
        for wrapped in _wrap(line, max(42, (width - x * 2) // 9)):
            draw.text((x, line_y), wrapped, fill="#222222", font=fonts["small"])
            line_y += 20


def _draw_panel_details(
    draw: Any,
    origin: tuple[int, int],
    view: _RunView,
    fonts: Mapping[str, Any],
    wrap_width: int,
) -> None:
    x, y = origin
    tasks = sorted(
        {
            prediction.task_text
            for prediction in view.predictions
            if prediction.task_text is not None
        }
    )
    observers = sorted(
        {
            prediction.observer_id
            for prediction in view.predictions
            if prediction.observer_id is not None
        }
    )
    stopping = sorted(
        {
            prediction.stopping_reason.value
            for prediction in view.predictions
            if prediction.stopping_reason is not None
        }
    )
    lines = [
        f"item: {view.predictions[0].item_id}",
        "samples: " + _sample_summary(view.predictions),
        "observer: " + (", ".join(observers) if observers else "anonymous"),
        "exact task: " + (" | ".join(tasks) if tasks else "none"),
    ]
    explanations = [
        prediction.explanation_text
        for prediction in view.predictions
        if prediction.explanation_text is not None
    ]
    if explanations:
        lines.append("explanation: " + " | ".join(explanations))
    lines.append("stopping: " + (", ".join(stopping) if stopping else "not recorded"))
    if view.warnings:
        lines.append("warnings: " + " | ".join(view.warnings))
    current = y
    for line in lines:
        for wrapped in _wrap(line, wrap_width):
            draw.text((x, current), wrapped, fill="#222222", font=fonts["small"])
            current += 18
            if current > y + 146:
                return


def _times(prediction: PredictionRecord) -> tuple[float, ...]:
    timestamps = tuple(fixation.timestamp_s for fixation in prediction.fixations)
    if timestamps and all(value is not None for value in timestamps):
        return tuple(float(value) for value in timestamps if value is not None)
    durations = tuple(fixation.duration_s for fixation in prediction.fixations)
    if durations and all(value is not None for value in durations):
        total = 0.0
        values = []
        for duration in durations:
            total += float(duration)
            values.append(total)
        return tuple(values)
    return tuple(float(index) for index in range(len(prediction.fixations)))


def _display_points(
    fixations: Sequence[FixationEvent],
    width: int,
    height: int,
    transform: tuple[float, float, float, float],
) -> tuple[tuple[tuple[float, float], ...], tuple[str, ...]]:
    offset_x, offset_y, scale_x, scale_y = transform
    points = []
    warnings = []
    for fixation in fixations:
        x, y, _ = _fixation_xy(fixation, width, height)
        clipped_x = min(max(x, 0.0), width - 1)
        clipped_y = min(max(y, 0.0), height - 1)
        if clipped_x != x or clipped_y != y:
            warnings.append("out-of-bounds coordinates were clipped only for display")
        points.append((offset_x + clipped_x * scale_x, offset_y + clipped_y * scale_y))
    return tuple(points), tuple(warnings)


def _fixation_xy(
    fixation: FixationEvent, width: int, height: int
) -> tuple[float, float, bool]:
    if fixation.x_px is not None and fixation.y_px is not None:
        return fixation.x_px, fixation.y_px, False
    if fixation.x_norm is not None and fixation.y_norm is not None:
        return fixation.x_norm * (width - 1), fixation.y_norm * (height - 1), True
    raise InspectionDataError("fixation has no complete coordinate pair")


def _metric_summary(rows: tuple[Mapping[str, str], ...]) -> tuple[str, ...]:
    result = []
    names = {
        "endpoint_pairwise_distance_normalized",
        "endpoint_dispersion_normalized",
        "fixation_count",
        "timestamp_span_s",
        "total_fixation_duration_s",
    }
    for row in rows:
        name = row.get("metric_name", "")
        if name not in names or row.get("status") != "applicable":
            continue
        value = row.get("mean") or row.get("value")
        if value:
            result.append(f"{name}={value} {row.get('unit', '')}".strip())
    return tuple(dict.fromkeys(result))


def _artifact_manifest(
    *,
    command: str,
    figure_path: Path,
    figure_sha256: str,
    figure_format: FigureFormat,
    plot: PlotParameters,
    views: tuple[_RunView, ...],
    selectors: Mapping[str, object],
    panels: tuple[Mapping[str, object], ...],
    warnings: tuple[str, ...],
) -> dict[str, object]:
    Image, _, _ = _pillow()
    return {
        "schema_version": 1,
        "command": command,
        "inference_rerun": command == "compare-instructions",
        "selectors": dict(selectors),
        "coordinate_convention": {
            "space": "canonical_pixel_centers",
            "pixel_origin": "top_left_zero_based",
            "normalized_to_pixel": "x_norm*(image_width-1), y_norm*(image_height-1)",
            "out_of_bounds": "preserved_in_records; clipped_only_for_display_with_warning",
        },
        "figure": {
            "filename": figure_path.name,
            "format": figure_format.value,
            "sha256": figure_sha256,
            "renderer": {
                "backend": "Pillow",
                "version": getattr(Image, "__version__", None),
                "headless": True,
            },
            "plot_parameters": plot.to_dict(),
            "panels": [dict(panel) for panel in panels],
        },
        "image_identity": {
            "sha256": views[0].image_sha256,
            "width_px": views[0].image_width,
            "height_px": views[0].image_height,
        },
        "runs": [_view_manifest(view) for view in views],
        "warnings": list(warnings),
        "deterministic_order": "run input order, then item/observer/task/sample_index/sample_id",
    }


def _view_manifest(view: _RunView) -> dict[str, object]:
    model = view.manifest.get("model")
    capabilities = model.get("capabilities") if isinstance(model, Mapping) else None
    return {
        "run_id": view.run_id,
        "run_directory": str(view.directory.path),
        "status": view.manifest.get("status"),
        "configuration_hash": view.manifest.get("configuration_hash"),
        "model": {
            "id": view.model_id,
            "variant": view.predictions[0].model.model_variant,
            "adapter_version": view.predictions[0].model.adapter_version,
            "upstream_version": view.predictions[0].model.upstream_version,
            "checkpoint_id": view.predictions[0].model.checkpoint_id,
            "capabilities": capabilities,
        },
        "dataset": {
            "id": view.predictions[0].dataset_id,
            "split": view.predictions[0].dataset_split,
        },
        "image_reference": str(view.image_path),
        "predictions": [
            {
                "item_id": prediction.item_id,
                "request_id": prediction.request_id,
                "sample_id": prediction.sample_id,
                "sample_index": prediction.sample_index,
                "seed": prediction.seed,
                "observer_id": prediction.observer_id,
                "task_text": prediction.task_text,
                "target_description": prediction.target_description,
                "fixation_count": len(prediction.fixations),
                "stopping_reason": (
                    None
                    if prediction.stopping_reason is None
                    else prediction.stopping_reason.value
                ),
                "explanation_text": prediction.explanation_text,
                "warnings": list(prediction.warnings),
                "native_artifacts": [
                    artifact.to_dict() for artifact in prediction.native_artifacts
                ],
            }
            for prediction in view.predictions
        ],
        "ground_truth_scanpaths": [
            {
                "scanpath_id": path.scanpath_id,
                "observer_id": path.observer_id,
                "task_text": path.task_text,
                "fixation_count": len(path.fixations),
            }
            for path in view.ground_truth
        ],
        "evaluation_rows": [dict(row) for row in view.metric_rows],
        "warnings": list(view.warnings),
    }


def _inspect_paths(request: InspectRequest, run_directory: Path) -> tuple[Path, Path]:
    output = request.output_directory or run_directory / "figures" / "inspect"
    stem = f"inspect-{_safe_filename(request.item_id)}"
    return output / f"{stem}.{request.figure_format.value}", output / f"{stem}.manifest.json"


def _compare_paths(
    request: CompareRequest, default_root: Path
) -> tuple[Path, Path]:
    output = request.output_directory or default_root / "comparisons"
    stem = f"compare-{_safe_filename(request.item_id)}"
    return output / f"{stem}.{request.figure_format.value}", output / f"{stem}.manifest.json"


def _selector_label(predictions: tuple[PredictionRecord, ...]) -> str:
    observers = ", ".join(
        sorted(
            {
                prediction.observer_id
                for prediction in predictions
                if prediction.observer_id is not None
            }
        )
    ) or "anonymous"
    tasks = " | ".join(
        sorted(
            {
                prediction.task_text
                for prediction in predictions
                if prediction.task_text is not None
            }
        )
    ) or "none"
    return (
        f"{_sample_summary(predictions)} | observer(s): {observers} | "
        f"exact task(s): {tasks}"
    )


def _model_label(model_id: str) -> str:
    return _MODEL_LABELS.get(model_id, model_id)


def _sample_summary(predictions: tuple[PredictionRecord, ...]) -> str:
    seeds = sorted({prediction.seed for prediction in predictions})
    if not seeds:
        return "0 stochastic samples"
    seed_text = str(seeds[0])
    if len(seeds) > 1:
        seed_text = f"{seeds[0]}-{seeds[-1]}"
    return f"{len(predictions)} stochastic samples | seeds {seed_text}"


def _local_image_path(reference: str, run_directory: Path) -> Path:
    if reference.startswith("file://"):
        parsed = urlparse(reference)
        if parsed.netloc not in {"", "localhost"}:
            raise InspectionDataError("source image file URI must be local")
        path = Path(unquote(parsed.path))
    elif "://" in reference:
        raise InspectionDataError(
            "inspection is offline and cannot fetch a non-local source image"
        )
    else:
        path = Path(reference)
        if not path.is_absolute():
            path = run_directory / path
    path = path.expanduser().resolve(strict=False)
    if not path.is_file():
        raise InspectionDataError(f"source image is missing: {path}")
    return path


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with _open_image(path) as image:
            width, height = image.size
    except InspectionOutputError:
        raise
    except Exception as error:
        raise InspectionDataError(f"cannot decode source image '{path}': {error}") from error
    if width < 2 or height < 2:
        raise InspectionDataError("source image dimensions must be at least 2x2")
    return int(width), int(height)


def _open_image(path: Path) -> Any:
    Image, _, _ = _pillow()
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except Exception as error:
        raise InspectionDataError(f"cannot decode source image '{path}': {error}") from error


def _pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise InspectionOutputError(
            "static figure generation requires Pillow; install the visualization extra"
        ) from error
    return Image, ImageDraw, ImageFont


def _fonts(ImageFont: Any) -> Mapping[str, Any]:
    return {
        "title": ImageFont.load_default(size=28),
        "body": ImageFont.load_default(size=18),
        "small": ImageFont.load_default(size=14),
    }


def _circle(
    draw: Any,
    point: tuple[float, float],
    radius: int,
    fill: str,
    outline: str,
    width: int,
) -> None:
    x, y = point
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=outline,
        width=width,
    )


def _time_color(index: int, count: int) -> str:
    ratio = 0.0 if count <= 1 else index / (count - 1)
    start = (0, 114, 178)
    end = (213, 94, 0)
    rgb = tuple(round(first + (second - first) * ratio) for first, second in zip(start, end))
    return "#" + "".join(f"{value:02X}" for value in rgb)


def _wrap(text: str, width: int) -> tuple[str, ...]:
    if len(text) <= width:
        return (text,)
    lines = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            while len(word) > width:
                lines.append(word[:width])
                word = word[width:]
            current = word
    if current:
        lines.append(current)
    return tuple(lines)


def _instructions(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or len(values) < 2:
        raise InspectionConfigurationError("at least two --instruction values are required")
    result = []
    for index, value in enumerate(values):
        if type(value) is not str or not value:
            raise InspectionConfigurationError(
                f"instructions[{index}] must be a non-empty exact string"
            )
        result.append(value)
    if len(result) != len(set(result)):
        raise InspectionConfigurationError("instruction texts must be distinct")
    return tuple(result)


def _preflight(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise InspectionOutputError(
            f"output already exists: {existing[0]}; use explicit overwrite"
        )
    parents = {path.parent for path in paths}
    for parent in parents:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise InspectionOutputError(
                f"cannot create output directory '{parent}': {error}"
            ) from error


def _atomic_image(path: Path, image: Any, figure_format: FigureFormat) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=f".{figure_format.value}", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_image = image.convert("RGB")
        save_image.save(
            temporary,
            format="PNG" if figure_format is FigureFormat.PNG else "PDF",
            optimize=False,
        )
        os.replace(temporary, path)
    except Exception as error:
        raise InspectionOutputError(f"cannot write figure '{path}': {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise InspectionOutputError(f"cannot write manifest '{path}': {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise InspectionDataError(f"cannot hash file '{path}': {error}") from error
    return digest.hexdigest()


def _safe_filename(value: str) -> str:
    readable = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    )
    readable = readable.strip(".-")[:80]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{readable or 'item'}-{digest}"

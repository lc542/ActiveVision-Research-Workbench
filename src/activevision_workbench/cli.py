
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from activevision_workbench.adapters.dummy import (
    DUMMY_MODEL_ID,
    create_dummy_registry,
)
from activevision_workbench.errors import (
    ContractValidationError,
    EvaluationConfigurationError,
    EvaluationDataError,
    EvaluationOutputError,
    InspectionConfigurationError,
    InspectionDataError,
    InspectionOutputError,
    RegistryError,
    RunExistsError,
    RunResumeError,
    SlurmError,
    WorkbenchError,
)
from activevision_workbench.registry import AdapterRegistry
from activevision_workbench.run_artifacts import RunStatus
from activevision_workbench.run_config import RunConfig
from activevision_workbench.run_engine import RunEngine


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="activevision")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", description="Execute or inspect a local inference run."
    )
    run.add_argument("--config", required=True, help="Path to a run YAML/JSON file.")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print run/resume decisions without writing.",
    )
    mode = run.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume", action="store_true", help="Resume the configured run ID."
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing run directory.",
    )
    run.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry failed incomplete items during resume.",
    )
    evaluate = commands.add_parser(
        "evaluate", description="Evaluate a completed local inference run."
    )
    evaluate.add_argument(
        "--run", required=True, help="Path to the completed run directory."
    )
    evaluate.add_argument(
        "--protocol", required=True, help="Path to an evaluation YAML/JSON file."
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="Show metric eligibility without writing evaluation outputs.",
    )
    evaluation_mode = evaluate.add_mutually_exclusive_group()
    evaluation_mode.add_argument(
        "--resume", action="store_true", help="Resume matching evaluation output."
    )
    evaluation_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing evaluation files explicitly.",
    )
    inspect = commands.add_parser(
        "inspect", description="Render static artifacts from one completed run."
    )
    inspect.add_argument("--run", required=True, help="Completed run directory.")
    inspect.add_argument("--item-id", required=True, help="Exact run or source item ID.")
    inspect.add_argument("--sample-id", help="Select one preserved sample ID.")
    inspect.add_argument("--observer-id", help="Select one exact observer ID.")
    inspect.add_argument(
        "--task-condition", help="Select one exact task/instruction string."
    )
    inspect.add_argument(
        "--format", choices=("png", "pdf"), default="png", help="Figure format."
    )
    inspect.add_argument("--output-dir", help="Static artifact output directory.")
    inspect.add_argument(
        "--dry-run", action="store_true", help="Validate selection without writing."
    )
    inspect.add_argument(
        "--overwrite", action="store_true", help="Replace exact existing artifacts."
    )
    compare = commands.add_parser(
        "compare", description="Compare explicitly selected completed runs."
    )
    compare.add_argument("--runs", required=True, nargs="+", help="Two or more runs.")
    compare.add_argument("--item-id", required=True, help="Exact shared item ID.")
    compare.add_argument("--sample-id", help="Select one exact sample ID when shared.")
    compare.add_argument("--observer-id", help="Select one exact observer ID.")
    compare.add_argument(
        "--task-condition", help="Select one exact task/instruction string."
    )
    compare.add_argument(
        "--format", choices=("png", "pdf"), default="png", help="Figure format."
    )
    compare.add_argument("--output-dir", help="Static artifact output directory.")
    compare.add_argument(
        "--dry-run", action="store_true", help="Validate comparison without writing."
    )
    compare.add_argument(
        "--overwrite", action="store_true", help="Replace exact existing artifacts."
    )
    instructions = commands.add_parser(
        "compare-instructions",
        description="Run and compare exact instructions for a supported adapter.",
    )
    instructions.add_argument("--config", required=True, help="Base model run config.")
    instructions.add_argument("--model", required=True, help="Configured model ID.")
    instructions.add_argument("--image", required=True, help="Local source image.")
    instructions.add_argument(
        "--instruction",
        required=True,
        action="append",
        help="Exact instruction; provide at least twice.",
    )
    instructions.add_argument(
        "--output-dir", required=True, help="Experiment runs and static artifacts root."
    )
    instructions.add_argument("--run-id", help="Optional explicit experiment run ID.")
    instructions.add_argument(
        "--format", choices=("png", "pdf"), default="png", help="Figure format."
    )
    instructions.add_argument(
        "--dry-run", action="store_true", help="Validate requests without inference."
    )
    instructions.add_argument(
        "--overwrite", action="store_true", help="Replace an existing experiment run."
    )
    submit = commands.add_parser(
        "submit", description="Plan or submit a trusted internal Slurm run."
    )
    submit.add_argument("--config", required=True, help="Run YAML/JSON configuration.")
    submit.add_argument("--slurm", required=True, help="Slurm YAML/JSON configuration.")
    submit_mode = submit.add_mutually_exclusive_group()
    submit_mode.add_argument(
        "--dry-run", action="store_true", help="Print the complete plan without writing or submitting."
    )
    submit_mode.add_argument(
        "--materialize-only",
        action="store_true",
        help="Write the plan and batch scripts without calling sbatch.",
    )
    submit.add_argument(
        "--resume", action="store_true", help="Submit failed or incomplete shards within retry limits."
    )
    submit.add_argument(
        "--retry-running",
        action="store_true",
        help="Explicitly treat stale running statuses as retryable; first verify the old job ended.",
    )
    shard = commands.add_parser(
        "slurm-run-shard", help="Execute one materialized shard (internal)."
    )
    shard.add_argument("--plan", required=True)
    shard.add_argument("--shard-index", required=True, type=int)
    merge = commands.add_parser(
        "slurm-merge", help="Finalize materialized shard outputs (internal)."
    )
    merge.add_argument("--plan", required=True)
    merge.add_argument("--replace-matching-plan", action="store_true")
    retry = commands.add_parser(
        "slurm-retry-plan", help="Inspect retry-eligible shard state."
    )
    retry.add_argument("--plan", required=True)
    retry.add_argument(
        "--include-running",
        action="store_true",
        help="Treat stale running statuses as eligible after checking the old job.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:

    arguments = build_parser().parse_args(argv)
    if arguments.command == "evaluate":
        return _evaluate(arguments)
    if arguments.command == "inspect":
        return _inspect(arguments)
    if arguments.command == "compare":
        return _compare(arguments)
    if arguments.command == "compare-instructions":
        return _compare_instructions(arguments)
    if arguments.command == "submit":
        return _submit(arguments)
    if arguments.command == "slurm-run-shard":
        return _slurm_run_shard(arguments)
    if arguments.command == "slurm-merge":
        return _slurm_merge(arguments)
    if arguments.command == "slurm-retry-plan":
        return _slurm_retry_plan(arguments)
    try:
        config = RunConfig.from_file(arguments.config)
        registry = _registry_for(config)
    except (WorkbenchError, OSError) as error:
        print(f"activevision: {error}", file=sys.stderr)
        return 2

    engine = RunEngine(registry)
    try:
        if arguments.dry_run:
            summary = engine.dry_run(
                config,
                resume=arguments.resume,
                overwrite=arguments.overwrite,
                retry_failures=arguments.retry_failures,
            )
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
            print(f"Run directory: {summary.run_directory}")
            return 0
        result = engine.run(
            config,
            resume=arguments.resume,
            overwrite=arguments.overwrite,
            retry_failures=arguments.retry_failures,
        )
        print(f"Run directory: {result.run_directory}")
        if result.status is RunStatus.COMPLETED:
            return 0
        if result.status is RunStatus.COMPLETED_WITH_ITEM_FAILURES:
            return 3
        if result.status is RunStatus.INTERRUPTED:
            return 130
        return 1
    except (
        ContractValidationError,
        RegistryError,
        RunExistsError,
        RunResumeError,
    ) as error:
        print(f"activevision: {error}", file=sys.stderr)
        return 2
    except (WorkbenchError, OSError) as error:
        print(f"activevision: run failed: {error}", file=sys.stderr)
        return 1


def _evaluate(arguments: argparse.Namespace) -> int:
    from activevision_workbench.evaluation import (
        EvaluationEngine,
        EvaluationProtocol,
        ExistingOutputPolicy,
    )

    try:
        protocol = EvaluationProtocol.from_file(arguments.protocol)
        output_policy = None
        if arguments.resume:
            output_policy = ExistingOutputPolicy.RESUME
        elif arguments.overwrite:
            output_policy = ExistingOutputPolicy.OVERWRITE
        protocol = protocol.with_overrides(
            run_directory=arguments.run,
            existing_output=output_policy,
        )
        engine = EvaluationEngine()
        if arguments.dry_run:
            plan = engine.dry_run(protocol)
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            print(f"Evaluation directory: {plan.output_directory}")
            return 0
        result = engine.evaluate(protocol)
        print(f"Evaluation directory: {result.output_directory}")
        return 0 if result.status == "completed" else 3
    except (
        EvaluationConfigurationError,
        EvaluationDataError,
        EvaluationOutputError,
    ) as error:
        print(f"activevision: {error}", file=sys.stderr)
        return 2
    except (WorkbenchError, OSError) as error:
        print(f"activevision: evaluation failed: {error}", file=sys.stderr)
        return 1


def _inspect(arguments: argparse.Namespace) -> int:
    from activevision_workbench.inspection import (
        FigureFormat,
        InspectRequest,
        InspectionEngine,
    )

    try:
        request = InspectRequest(
            run_directory=arguments.run,
            item_id=arguments.item_id,
            sample_id=arguments.sample_id,
            observer_id=arguments.observer_id,
            task_text=arguments.task_condition,
            figure_format=FigureFormat(arguments.format),
            output_directory=arguments.output_dir,
            overwrite=arguments.overwrite,
        )
        engine = InspectionEngine()
        if arguments.dry_run:
            print(json.dumps(engine.dry_run_inspect(request).to_dict(), indent=2, sort_keys=True))
            return 0
        result = engine.inspect(request)
        print(f"Figure: {result.figure_path}")
        print(f"Manifest: {result.manifest_path}")
        return 0
    except (InspectionConfigurationError, InspectionDataError, InspectionOutputError) as error:
        print(f"activevision: {error}", file=sys.stderr)
        return 2


def _compare(arguments: argparse.Namespace) -> int:
    from activevision_workbench.inspection import (
        CompareRequest,
        FigureFormat,
        InspectionEngine,
    )

    try:
        request = CompareRequest(
            run_directories=tuple(arguments.runs),
            item_id=arguments.item_id,
            sample_id=arguments.sample_id,
            observer_id=arguments.observer_id,
            task_text=arguments.task_condition,
            figure_format=FigureFormat(arguments.format),
            output_directory=arguments.output_dir,
            overwrite=arguments.overwrite,
        )
        engine = InspectionEngine()
        if arguments.dry_run:
            print(json.dumps(engine.dry_run_compare(request).to_dict(), indent=2, sort_keys=True))
            return 0
        result = engine.compare(request)
        print(f"Figure: {result.figure_path}")
        print(f"Manifest: {result.manifest_path}")
        return 0
    except (InspectionConfigurationError, InspectionDataError, InspectionOutputError) as error:
        print(f"activevision: {error}", file=sys.stderr)
        return 2


def _compare_instructions(arguments: argparse.Namespace) -> int:
    from pathlib import Path

    from activevision_workbench.inspection import (
        FigureFormat,
        InspectionEngine,
        build_instruction_run_config,
        validate_instruction_capability,
    )

    try:
        base = RunConfig.from_file(arguments.config)
        output = Path(arguments.output_dir).expanduser().resolve(strict=False)
        config = build_instruction_run_config(
            base,
            model_id=arguments.model,
            image_path=arguments.image,
            instructions=arguments.instruction,
            output_root=output / "runs",
            run_id=arguments.run_id,
        )
        registry = _registry_for(config)
        validate_instruction_capability(
            registry.capabilities(config.model_id, config.model_variant)
        )
        run_engine = RunEngine(registry)
        if arguments.dry_run:
            run_plan = run_engine.dry_run(config, overwrite=arguments.overwrite)
            value = run_plan.to_dict()
            value["instructions"] = list(arguments.instruction)
            value["image"] = str(Path(arguments.image).expanduser().resolve(strict=False))
            value["comparison_directory"] = str(output / "figures")
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        run_result = run_engine.run(config, overwrite=arguments.overwrite)
        if run_result.status is not RunStatus.COMPLETED:
            print(f"Run directory: {run_result.run_directory}")
            return 3
        result = InspectionEngine().compare_instructions(
            run_result.run_directory,
            arguments.instruction,
            output_directory=output / "figures",
            figure_format=FigureFormat(arguments.format),
            overwrite=arguments.overwrite,
        )
        print(f"Run directory: {run_result.run_directory}")
        print(f"Figure: {result.figure_path}")
        print(f"Manifest: {result.manifest_path}")
        return 0
    except (
        InspectionConfigurationError,
        InspectionDataError,
        InspectionOutputError,
        WorkbenchError,
        OSError,
    ) as error:
        print(f"activevision: instruction comparison failed: {error}", file=sys.stderr)
        return 2


def _registry_for(config: RunConfig) -> AdapterRegistry:
    if config.model_id == DUMMY_MODEL_ID:
        return create_dummy_registry()
    if config.model_id == "gazexplain":
        from activevision_workbench.adapters.gazexplain import (
            GazeXplainAdapterConfig,
            create_gazexplain_registry,
        )

        return create_gazexplain_registry(
            GazeXplainAdapterConfig.from_run_config(config)
        )
    if config.model_id == "individualscanpath":
        from activevision_workbench.adapters.individual_scanpath import (
            IndividualScanpathAdapterConfig,
            create_individual_scanpath_registry,
        )

        return create_individual_scanpath_registry(
            IndividualScanpathAdapterConfig.from_run_config(config)
        )
    if config.model_id == "scandiff":
        from activevision_workbench.adapters.scandiff import (
            ScanDiffAdapterConfig,
            create_scandiff_registry,
        )

        return create_scandiff_registry(ScanDiffAdapterConfig.from_run_config(config))
    if config.model_id == "tpp_gaze":
        from activevision_workbench.adapters.tpp_gaze import (
            TPPGazeAdapterConfig,
            create_tpp_gaze_registry,
        )

        return create_tpp_gaze_registry(TPPGazeAdapterConfig.from_run_config(config))
    return AdapterRegistry()


def _submit(arguments: argparse.Namespace) -> int:
    from dataclasses import replace

    from activevision_workbench.slurm import (
        SlurmConfig,
        SlurmExecutor,
        load_plan,
        retry_plan,
    )

    try:
        if arguments.retry_running and not arguments.resume:
            raise SlurmError("--retry-running requires --resume")
        if arguments.materialize_only and arguments.resume:
            raise SlurmError("--materialize-only cannot be combined with --resume")
        config = RunConfig.from_file(arguments.config)
        slurm = SlurmConfig.from_file(arguments.slurm)
        registry = _registry_for(config)
        registry.describe(config.model_id, config.model_variant)
        executor = SlurmExecutor()
        if arguments.dry_run:
            plan = executor.dry_run(config, slurm)
            summary = plan.summary()
            if arguments.resume:
                if not plan.plan_directory.exists():
                    raise SlurmError(
                        f"resume plan does not exist: {plan.plan_directory}"
                    )
                stored = load_plan(plan.plan_directory)
                retry = retry_plan(
                    plan.plan_directory,
                    include_running=arguments.retry_running,
                )
                plan = replace(
                    stored,
                    selected_shards=retry.eligible_shards,
                    retry=True,
                )
                summary = plan.summary()
                summary["retry_plan"] = retry.to_dict()
            print(json.dumps(summary, indent=2, sort_keys=True))
            print(f"Plan directory: {plan.plan_directory}")
            return 0
        if arguments.materialize_only:
            plan = executor.materialize(config, slurm)
            print(f"Plan directory: {plan.plan_directory}")
            print(f"Array script: {plan.plan_directory / 'array.sbatch'}")
            print(f"Merge script: {plan.plan_directory / 'merge.sbatch'}")
            return 0
        result = executor.submit(
            config,
            slurm,
            resume=arguments.resume,
            retry_running=arguments.retry_running,
        )
        print(f"Plan directory: {result.plan_directory}")
        if result.array_job_id is None:
            print("No eligible shards remain within the retry policy.")
            return 0
        print(f"Array job ID: {result.array_job_id}")
        print(f"Merge job ID: {result.merge_job_id}")
        return 0
    except (SlurmError, WorkbenchError, OSError) as error:
        print(f"activevision: Slurm submission failed: {error}", file=sys.stderr)
        return 2


def _slurm_run_shard(arguments: argparse.Namespace) -> int:
    from activevision_workbench.slurm import (
        install_term_handler,
        load_plan,
        restore_term_handler,
        run_shard,
    )

    previous = None
    try:
        plan = load_plan(arguments.plan)
        previous = install_term_handler()
        result = run_shard(
            arguments.plan,
            arguments.shard_index,
            _registry_for,
        )
        if result is None or result.status is RunStatus.COMPLETED:
            return 0
        if result.status is RunStatus.COMPLETED_WITH_ITEM_FAILURES:
            return 3
        if result.status is RunStatus.INTERRUPTED:
            return 130
        return 1
    except (SlurmError, WorkbenchError, OSError) as error:
        print(f"activevision: Slurm shard failed: {error}", file=sys.stderr)
        return 1
    finally:
        if previous is not None:
            restore_term_handler(previous)


def _slurm_merge(arguments: argparse.Namespace) -> int:
    from activevision_workbench.slurm import load_plan, merge_shards

    try:
        plan = load_plan(arguments.plan)
        registry = _registry_for(plan.run_config)
        destination = merge_shards(
            arguments.plan,
            registry,
            replace_matching_plan=arguments.replace_matching_plan,
        )
        print(f"Run directory: {destination}")
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        return 0 if manifest.get("status") == RunStatus.COMPLETED.value else 3
    except (SlurmError, WorkbenchError, OSError) as error:
        print(f"activevision: Slurm merge failed: {error}", file=sys.stderr)
        return 1


def _slurm_retry_plan(arguments: argparse.Namespace) -> int:
    from activevision_workbench.slurm import retry_plan

    try:
        print(
            json.dumps(
                retry_plan(
                    arguments.plan,
                    include_running=arguments.include_running,
                ).to_dict(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (SlurmError, OSError) as error:
        print(f"activevision: retry planning failed: {error}", file=sys.stderr)
        return 2

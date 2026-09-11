# ActiveVision Research Workbench

This project is the work for the Google Summer of Code 2026, with the organization INCF.
The complete project summary, benchmark results, and figures are available in
the [Google Summer of Code 2026 report](https://gist.github.com/lc542/798b5114dad87928212dc54f21384576).

ActiveVision Research Workbench provides reproducible execution, evaluation,
inspection, and comparison for four gaze and scanpath models:

- ScanDiff
- TPP-Gaze
- IndividualScanpath
- GazeXplain

The models share versioned request, prediction, run, evaluation, and inspection
artifacts while retaining their task, observer, temporal, stochastic, and text
semantics. Model dependencies run in separate Apptainer or Python environments.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
activevision run --config examples/dummy_tiny_dataset_run.yaml
activevision evaluate \
  --run runs/dummy-tiny-dataset-v1 \
  --protocol examples/evaluation/dummy_tiny_dataset.yaml
activevision inspect \
  --run runs/dummy-tiny-dataset-v1 \
  --item-id scene-a
```

The dummy path is CPU-only and uses the committed synthetic dataset fixture.
Use a different `run_id`, explicit resume, or explicit overwrite when repeating
the example.

The base installation accepts the checked-in JSON-compatible YAML files. For
general block-style YAML, install the optional parser:

```bash
python -m pip install -e '.[yaml]'
```

## Workflow and artifacts

```text
dataset adapter -> inference request -> model adapter -> prediction JSONL
                -> run manifest -> evaluation -> inspect / compare
```

Each run directory contains the resolved configuration, provenance, prediction
JSONL, status and error records, and a run manifest. Evaluation adds metric
tables and summaries; inspection and comparison add a figure plus a
machine-readable figure manifest. Repeating a run requires an explicit
lifecycle choice:

```bash
activevision run --config path/to/config.json --dry-run
activevision run --config path/to/config.json --resume
activevision run --config path/to/config.json --resume --retry-failures
```

Use `--overwrite` only when intentionally replacing existing artifacts.

## Configuration

Run, evaluation, inspection, and Slurm files use strict schema version 1.
Unknown fields and unsupported capabilities fail during validation. Paths are
resolved relative to the configuration file, so configurations can remain
portable. A run configuration records:

- model identity, adapter options, upstream revision, and checkpoint identity;
- dataset identity, split, root, selected items, tasks, and observers;
- sample count, seed policy, fixation/time limits, and device;
- output location, run policy, tags, and environment identity.

`num_samples=N` always produces `N` separate prediction records. Every record
retains its stable sample ID, sample index, and actual seed.

## Datasets

The workbench supports normalized JSON fixtures, OSIE, and COCO-Search18.
Adapters read external data in place and preserve source coordinates, task
text, observer identity, timing, target boxes, split identity, and provenance.
Pixel/normalized conversion and clipping are explicit operations.

Real datasets are not copied into this repository. Set each configuration's
dataset root to a local project or scratch location visible from the execution
environment.

## Apptainer environments

### Core workbench image

Build the workbench image from the repository root:

```bash
container/build_apptainer.sh
```

On Compute Canada, the script loads `apptainer/1.4.5` automatically when
Apptainer is not already available. To rebuild an existing image, use:

```bash
container/build_apptainer.sh --force
```

Run the command-line interface or the included test suite from the image:

```bash
apptainer run container/activevision-workbench-py311.sif --help

apptainer exec container/activevision-workbench-py311.sif \
  python -m unittest discover -s /opt/activevision-workbench/tests -v
```

Datasets remain outside the image. Bind the required directory as read-only
when starting Apptainer. This example expects the datasets directory beside
the workbench repository:

```bash
osie_root="$(realpath ../datasets/activevision/osie/data)"
apptainer exec --bind "${osie_root}:/data/osie:ro" \
  container/activevision-workbench-py311.sif \
  python -c 'from activevision_workbench.datasets import OSIEDatasetAdapter; \
a=OSIEDatasetAdapter("/data/osie"); print(a.metadata.to_dict())'
```

### Model workers

Each model has its own image, worker launcher, and example configuration:

| Model | Build command | Worker launcher | Example configuration |
| --- | --- | --- | --- |
| ScanDiff | `container/build_scandiff_apptainer.sh` | `container/run_scandiff_worker.sh` | [`examples/scandiff/free_viewing.json`](examples/scandiff/free_viewing.json) |
| TPP-Gaze | `container/build_tppgaze_apptainer.sh` | `container/run_tppgaze_worker.sh` | [`examples/tpp_gaze/multiple_simulations.json`](examples/tpp_gaze/multiple_simulations.json) |
| IndividualScanpath | `container/build_individual_scanpath_apptainer.sh` | `container/run_individual_scanpath_worker.sh` | [`examples/individual_scanpath/two_observers.json`](examples/individual_scanpath/two_observers.json) |
| GazeXplain | `container/build_gazexplain_apptainer.sh` | `container/run_gazexplain_worker.sh` | [`examples/gazexplain/two_tasks.json`](examples/gazexplain/two_tasks.json) |

Build the environment for the selected model, then edit its example
configuration so the upstream repository, checkpoint, dataset, and output
paths point to your local files. For example:

```bash
container/build_scandiff_apptainer.sh
activevision run --config examples/scandiff/free_viewing.json
```

The worker launchers enable NVIDIA GPU access automatically. Generated `.sif`
images are not added to Git.

## Evaluation and inspection

```bash
activevision evaluate \
  --run runs/dummy-tiny-dataset-v1 \
  --protocol examples/evaluation/dummy_tiny_dataset.yaml \
  --dry-run

activevision compare \
  --runs runs/model-a runs/model-b \
  --item-id shared-item \
  --output-dir comparisons/shared-item
```

Evaluation preserves per-sample results and supports fixation count, normalized
scanpath length, timestamps, fixation duration, duration MAE, target-fixation
probability AUC, native-map AUC-Judd and NSS, scanpath-density AUC-Judd and NSS,
endpoint diversity/dispersion, and fixation KDE.
Eligibility is reported explicitly as `applicable`, `not_applicable`,
`unavailable`, `invalid`, or `error`. Specialized protocols are available in
[`examples/evaluation/`](examples/evaluation/).

Inspection produces deterministic PNG/PDF figures for scanpath overlays, KDE,
continuous-time timelines, observers, exact task instructions, and cross-model
comparisons. Runnable examples are in
[`examples/inspection/`](examples/inspection/).

## Compute Canada and Slurm

Run the following commands from the repository root with the workbench Python
environment active.

### Submit one experiment

Select a run configuration and its matching Slurm configuration:

```bash
activevision submit \
  --config examples/scandiff/free_viewing.json \
  --slurm examples/slurm/scandiff.json
```

This submits a GPU array followed by a CPU merge job. Use `--dry-run` to review
the plan without writing files or submitting jobs:

```bash
activevision submit \
  --config examples/scandiff/free_viewing.json \
  --slurm examples/slurm/scandiff.json \
  --dry-run
```

Use `--materialize-only` to write the plan and submission scripts without
submitting them:

```bash
activevision submit \
  --config examples/scandiff/free_viewing.json \
  --slurm examples/slurm/scandiff.json \
  --materialize-only
```

The command prints the plan directory containing:

```text
<output-root>/.slurm-work/<run-and-plan-id>/array.sbatch
<output-root>/.slurm-work/<run-and-plan-id>/merge.sbatch
```

Monitor submitted jobs with:

```bash
squeue --me
```

### Prepare the six final benchmark experiments

Create the run, Slurm, and evaluation configurations with:

```bash
PYTHONPATH=src python scripts/prepare_final_benchmark.py \
  --output-root path/to/final-v1 \
  --osie-root path/to/osie/data \
  --coco-root path/to/coco_search18/extracted \
  --scandiff-root path/to/ScanDiff \
  --tpp-gaze-root path/to/tppgaze \
  --individual-scanpath-root path/to/IndividualScanpath \
  --gazexplain-root path/to/GazeXplain \
  --account your-compute-canada-account
```

Submit all six generated full-run configurations:

```bash
benchmark_root=path/to/final-v1
for run_config in "${benchmark_root}"/configs/full/*.run.json; do
  config_stem="${run_config%.run.json}"
  activevision submit \
    --config "${run_config}" \
    --slurm "${config_stem}.slurm.json"
done
```

After all array and merge jobs finish, submit the CPU evaluation array:

```bash
sbatch \
  --account=your-account \
  --export=ALL,AVRW_BENCHMARK_ROOT="${benchmark_root}",AVRW_REPOSITORY_ROOT="${PWD}" \
  slurm/evaluate_final.sbatch
```

Evaluation outputs are written to `<benchmark-root>/evaluations/`.

### Resource and shard settings

The JSON files in `examples/slurm/` contain the GPU, CPU, memory, time, array,
and concurrency settings for each model. Increase the shard count to create
more short jobs, and use `max_parallel_tasks` to limit how many run at once.

Use `balanced` assignment to spread requests evenly across shards. For
observer-conditioned data, `balanced_by_image` keeps requests for the same
image together and reduces repeated image preprocessing.

The `.sbatch` files in [`slurm/`](slurm/) are starting templates for the four
models, merging, and final evaluation. Generated submission scripts contain
the settings from the selected JSON configuration.

The final Rorqual benchmark completed all six compatible model and dataset
pairs. It contains 1,912 requests and 19,120 separately preserved predictions
with ten samples per request.

## Dataset and model organization

The repository keeps workbench code, runnable configurations, environment
definitions, and small test fixtures together:

```text
ActiveVision-Research-Workbench/
├── src/activevision_workbench/
│   ├── datasets/                 # OSIE, COCO-Search18, and JSON adapters
│   └── adapters/                 # one adapter package per gaze model
├── examples/
│   ├── scandiff/                 # model run configurations
│   ├── tpp_gaze/
│   ├── individual_scanpath/
│   ├── gazexplain/
│   ├── evaluation/               # metric protocols
│   └── slurm/                    # scheduler profiles
├── container/                    # Apptainer definitions and worker launchers
├── slurm/                        # path-neutral submission templates
├── scripts/                      # benchmark preparation and release checks
├── tests/fixtures/               # small committed data and golden artifacts
└── runs/                         # generated local runs
```

Large datasets, upstream repositories, and checkpoints stay beside the
workbench rather than inside it. The example configurations follow this layout:

```text
<workspace>/
├── ActiveVision-Research-Workbench/
├── ScanDiff/
│   └── checkpoints/
├── tppgaze/
│   ├── data/config.yaml
│   └── checkpoints/
├── IndividualScanpath/
│   └── checkpoints/
├── GazeXplain/
│   └── checkpoints/
└── datasets/activevision/
    ├── osie/data/
    └── coco_search18/raw/extracted/
```

Each model keeps its upstream code and checkpoint files in its own directory.
Run configurations under `examples/<model>/` connect those assets to a dataset
adapter and choose an output directory. Edit the relative roots in the selected
configuration when using a different storage layout.

## Validation

```bash
python scripts/check_release.py --repository .
bash scripts/check_clean_environment.sh
```

The first command performs fast source, example, fixture, link, path, and shell
audits. The second creates a fresh virtual environment, installs the core,
runs the dummy workflow, and executes the complete CPU suite.

## License

Workbench-owned code is distributed under the [MIT License](LICENSE). Dataset,
upstream model, and checkpoint licenses remain governed by their original
providers.

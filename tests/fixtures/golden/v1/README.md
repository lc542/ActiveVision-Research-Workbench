# Golden artifact fixtures, schema version 1

These small files fix the public shape and representative semantics of AVRW's
persisted request, prediction, run, and evaluation artifacts. They use portable
paths and synthetic values.

| Fixture | Contract represented |
| --- | --- |
| `request.json` | Canonical deterministic inference request |
| `deterministic_prediction.json` | Canonical deterministic prediction |
| `stochastic_predictions.jsonl` | Two separately preserved stochastic samples |
| `temporal_prediction.json` | Continuous and native time fields |
| `observer_conditioned_prediction.json` | Observer identity and model mapping |
| `instruction_text_prediction.json` | Exact task text and explanation output |
| `run_manifest.json` | Run, provenance, seed, environment, and Slurm identity |
| `evaluation_summary.json` | Metric status and aggregation summary |

The version-directory name matches every top-level `schema_version`. Readers
accept version 1 strictly; schema evolution and migration policy are described
in `docs/research_workbench/reproducibility.md`.

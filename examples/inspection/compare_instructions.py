from __future__ import annotations

import argparse

from activevision_workbench.cli import main as activevision_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    command = [
        "compare-instructions",
        "--config",
        arguments.config,
        "--model",
        arguments.model,
        "--image",
        arguments.image,
        "--output-dir",
        arguments.output_dir,
    ]
    for instruction in arguments.instruction:
        command.extend(("--instruction", instruction))
    if arguments.dry_run:
        command.append("--dry-run")
    raise SystemExit(activevision_main(command))


if __name__ == "__main__":
    main()

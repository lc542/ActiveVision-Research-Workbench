from __future__ import annotations

import argparse

from activevision_workbench.inspection import InspectRequest, InspectionEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    result = InspectionEngine().inspect(
        InspectRequest(
            run_directory=arguments.run,
            item_id=arguments.item_id,
            sample_id=arguments.sample_id,
            output_directory=arguments.output_dir,
        )
    )
    print(result.figure_path)
    print(result.manifest_path)


if __name__ == "__main__":
    main()

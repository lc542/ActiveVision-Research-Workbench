from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image


EXPECTED_PATHS = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "tests/fixtures/golden/v1/request.json",
    "tests/fixtures/golden/v1/deterministic_prediction.json",
    "tests/fixtures/golden/v1/stochastic_predictions.jsonl",
    "tests/fixtures/golden/v1/temporal_prediction.json",
    "tests/fixtures/golden/v1/observer_conditioned_prediction.json",
    "tests/fixtures/golden/v1/instruction_text_prediction.json",
    "tests/fixtures/golden/v1/run_manifest.json",
    "tests/fixtures/golden/v1/evaluation_summary.json",
    "tests/fixtures/datasets/tiny_gaze/images/wide.png",
    "tests/fixtures/datasets/tiny_gaze/images/tall.png",
    "scripts/check_release.py",
    "scripts/check_clean_environment.sh",
    "scripts/generate_fixture_images.py",
)

PRIVATE_PATTERNS = (
    re.compile(r"/home/"),
    re.compile(r"/Users/"),
    re.compile(r"/lustre(?:\d+)?/"),
    re.compile(r"lchen\d+", re.IGNORECASE),
    re.compile(r"(?:def|rrg)-[a-z][a-z0-9_-]*", re.IGNORECASE),
)

TEXT_SUFFIXES = {".def", ".json", ".md", ".py", ".sbatch", ".sh", ".yaml"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="check-release")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="ActiveVision Research Workbench repository root",
    )
    return parser.parse_args()


def _files(root: Path, directory: str) -> tuple[Path, ...]:
    base = root / directory
    if not base.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.suffix in TEXT_SUFFIXES
    )


def _check_required(root: Path) -> list[str]:
    return [
        f"missing required release file: {path}"
        for path in EXPECTED_PATHS
        if not (root / path).is_file()
    ]


def _check_json_examples(root: Path) -> list[str]:
    errors = []
    paths = tuple(sorted((root / "examples").rglob("*.json"))) + tuple(
        sorted((root / "examples").glob("*.yaml"))
    )
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid JSON-compatible example {path.relative_to(root)}: {error}")
            continue
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            errors.append(f"example lacks schema_version 1: {path.relative_to(root)}")
    return errors


def _check_golden_json(root: Path) -> list[str]:
    errors = []
    golden = root / "tests" / "fixtures" / "golden" / "v1"
    for path in sorted(golden.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid golden JSON {path.relative_to(root)}: {error}")
            continue
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            errors.append(f"golden fixture lacks schema_version 1: {path.relative_to(root)}")
    jsonl = golden / "stochastic_predictions.jsonl"
    try:
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid golden JSONL {jsonl.relative_to(root)}: {error}")
    else:
        if len(values) < 2 or any(value.get("schema_version") != 1 for value in values):
            errors.append(
                "golden stochastic JSONL must contain at least two "
                "schema-version-1 records"
            )
    return errors


def _check_fixture_images(root: Path) -> list[str]:
    errors = []
    image_root = root / "tests" / "fixtures" / "datasets" / "tiny_gaze" / "images"
    expected = {"wide.png": (100, 50), "tall.png": (50, 100)}
    for name, dimensions in expected.items():
        path = image_root / name
        try:
            with Image.open(path) as image:
                image.load()
                actual = image.size
        except (OSError, ValueError) as error:
            errors.append(f"fixture image is not decodable {path.relative_to(root)}: {error}")
            continue
        if actual != dimensions:
            errors.append(
                f"fixture image has size {actual}, expected {dimensions}: "
                f"{path.relative_to(root)}"
            )
    return errors


def _check_markdown_links(root: Path) -> list[str]:
    errors = []
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    markdown = tuple(sorted(root.glob("*.md")))
    markdown += tuple(sorted((root / "docs").rglob("*.md")))
    markdown += tuple(sorted((root / "report").rglob("*.md")))
    markdown += tuple(sorted((root / "slurm").glob("*.md")))
    markdown += tuple(sorted((root / "container").glob("*.md")))
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            target = target.strip().strip("<>")
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).resolve(strict=False).exists():
                errors.append(
                    f"broken Markdown link in {path.relative_to(root)}: {target}"
                )
    return errors


def _check_private_paths(root: Path) -> list[str]:
    errors = []
    files = ()
    for directory in (
        "src",
        "examples",
        "slurm",
        "container",
        "scripts",
        "docs/research_workbench",
    ):
        files += _files(root, directory)
    for path in files:
        if path == root / "scripts" / "check_release.py":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                errors.append(
                    f"private path pattern {pattern.pattern!r} in {path.relative_to(root)}"
                )
    return errors


def _check_shell(root: Path) -> list[str]:
    errors = []
    paths = tuple(sorted((root / "container").glob("*.sh")))
    paths += tuple(sorted((root / "slurm").glob("*.sbatch")))
    paths += (root / "scripts" / "check_clean_environment.sh",)
    for path in paths:
        if not path.is_file():
            continue
        result = subprocess.run(
            ("bash", "-n", str(path)),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            errors.append(
                f"shell syntax failed for {path.relative_to(root)}: {result.stderr.strip()}"
            )
    return errors


def _check_tracked_images(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "ls-files", "*.sif"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if result.returncode == 0 and result.stdout.strip():
        return [
            f"generated Apptainer image is tracked: {line}"
            for line in result.stdout.splitlines()
        ]
    return []


def main() -> int:
    root = _arguments().repository.expanduser().resolve(strict=False)
    checks = (
        ("required release files", _check_required),
        ("JSON-compatible examples", _check_json_examples),
        ("golden fixtures", _check_golden_json),
        ("fixture images", _check_fixture_images),
        ("Markdown links", _check_markdown_links),
        ("private paths", _check_private_paths),
        ("shell syntax", _check_shell),
        ("tracked Apptainer images", _check_tracked_images),
    )
    errors = []
    for label, check in checks:
        current = check(root)
        if current:
            print(f"FAIL {label}")
            errors.extend(current)
        else:
            print(f"PASS {label}")
    if errors:
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUFFIX_PATTERN = re.compile(r"(\d+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rename images in a scene folder to img_xxx based on the numeric suffix "
            "at the end of each original filename."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Scene folder containing images to rename.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned renames without changing any files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing img_xxx names in the target folder.",
    )
    return parser


def extract_numeric_suffix(path: Path) -> str | None:
    match = SUFFIX_PATTERN.search(path.stem)
    return None if match is None else match.group(1)


def format_target_name(suffix: str, extension: str) -> str:
    width = max(3, len(suffix))
    return f"img_{int(suffix):0{width}d}{extension.lower()}"


def collect_renames(folder: Path) -> list[tuple[Path, Path]]:
    planned: list[tuple[Path, Path]] = []
    seen_targets: dict[str, Path] = {}

    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        suffix = extract_numeric_suffix(path)
        if suffix is None:
            continue

        target_name = format_target_name(suffix, path.suffix)
        target_path = path.with_name(target_name)

        if target_name in seen_targets and seen_targets[target_name] != path:
            other = seen_targets[target_name]
            raise ValueError(
                f"Duplicate target name detected: {path.name} and {other.name} both map to {target_name}"
            )

        seen_targets[target_name] = path
        planned.append((path, target_path))

    return planned


def validate_targets(folder: Path, planned: list[tuple[Path, Path]], force: bool) -> None:
    planned_targets = {target.name for _, target in planned}

    for source, target in planned:
        if source.name == target.name:
            continue

        if target.exists() and target.name not in planned_targets and not force:
            raise FileExistsError(
                f"Target already exists and is not part of this rename plan: {target.name}. "
                "Use --force if you want to replace existing img_xxx files."
            )

    if not planned:
        raise ValueError(
            f"No renamable images found in {folder}. Expected filenames ending with digits, "
            "for example 20260411_140944_001.jpg"
        )


def apply_renames(planned: list[tuple[Path, Path]], dry_run: bool) -> None:
    rename_pairs: list[tuple[Path, Path]] = []

    for source, target in planned:
        print(f"{source.name} -> {target.name}")
        if dry_run or source.name == target.name:
            continue

        temp_path = source.with_name(f".__rename_tmp__{uuid.uuid4().hex}{source.suffix.lower()}")
        source.rename(temp_path)
        rename_pairs.append((temp_path, target))

    if dry_run:
        return

    for temp_path, target in rename_pairs:
        if target.exists():
            target.unlink()
        temp_path.rename(target)


def main() -> int:
    args = build_parser().parse_args()
    folder = args.folder.resolve()

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    planned = collect_renames(folder)
    validate_targets(folder, planned, force=args.force)
    apply_renames(planned, dry_run=args.dry_run)

    changed_count = sum(1 for source, target in planned if source.name != target.name)
    print(f"\nProcessed {len(planned)} image(s); {changed_count} rename(s) planned.")
    if args.dry_run:
        print("Dry run only: no files were changed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_utils.panorama_dataset import list_all_image_files, list_scene_dirs, ordered_scene_files


VALIDATE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate dataset scenes and meta.json files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Root folder containing scene_* directories.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Open each image with Pillow and run Image.verify() to detect truncated files.",
    )
    return parser


def validate_scene(scene_dir: Path) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    meta_path = scene_dir / "meta.json"

    if not meta_path.exists():
        return [f"{scene_dir.name}: missing meta.json"], warnings, {}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ordered_files, reference_files, _, _ = ordered_scene_files(scene_dir)
    all_images = list_all_image_files(scene_dir)

    if meta.get("scene_id") != scene_dir.name:
        errors.append(f"{scene_dir.name}: scene_id mismatch ({meta.get('scene_id')})")

    ordered_names = [path.name for path in ordered_files]
    reference_names = [path.name for path in reference_files]
    all_names = {path.name for path in all_images}

    if meta.get("ordered_files") != ordered_names:
        errors.append(f"{scene_dir.name}: ordered_files mismatch")

    if meta.get("reference_files", []) != reference_names:
        errors.append(f"{scene_dir.name}: reference_files mismatch")

    if any(name not in all_names for name in meta.get("ordered_files", [])):
        errors.append(f"{scene_dir.name}: meta ordered_files contains missing image(s)")

    if any(name not in all_names for name in meta.get("reference_files", [])):
        errors.append(f"{scene_dir.name}: meta reference_files contains missing image(s)")

    if set(meta.get("ordered_files", [])) & set(meta.get("reference_files", [])):
        errors.append(f"{scene_dir.name}: ordered_files overlaps reference_files")

    if meta.get("num_images") != len(ordered_files):
        errors.append(f"{scene_dir.name}: num_images mismatch")

    image_stats = meta.get("image_stats", [])
    pair_audit = meta.get("pair_audit", [])

    if len(image_stats) != len(ordered_files):
        errors.append(f"{scene_dir.name}: image_stats length mismatch")

    if [row.get("file") for row in image_stats] != ordered_names:
        errors.append(f"{scene_dir.name}: image_stats file order mismatch")

    expected_pair_count = max(0, len(ordered_files) - 1)
    if len(pair_audit) != expected_pair_count:
        errors.append(f"{scene_dir.name}: pair_audit length mismatch")

    expected_pairs = list(zip(ordered_names[:-1], ordered_names[1:]))
    actual_pairs = [(row.get("image_a"), row.get("image_b")) for row in pair_audit]
    if actual_pairs != expected_pairs:
        errors.append(f"{scene_dir.name}: pair_audit image order mismatch")

    audit_summary = meta.get("audit_summary", {})
    pair_counter = Counter(row.get("pair_label") for row in pair_audit)
    expected_label_counts = {
        "strong": int(pair_counter.get("strong", 0)),
        "ok": int(pair_counter.get("ok", 0)),
        "weak": int(pair_counter.get("weak", 0)),
        "fail": int(pair_counter.get("fail", 0)),
    }
    if audit_summary.get("pair_label_counts") != expected_label_counts:
        errors.append(f"{scene_dir.name}: pair_label_counts mismatch")

    stability_check = audit_summary.get("stability_check")
    if stability_check is not None:
        if not isinstance(stability_check, dict):
            errors.append(f"{scene_dir.name}: stability_check must be an object or null")
        else:
            runs = stability_check.get("runs")
            status_counts = stability_check.get("status_counts", {})
            ok_runs = stability_check.get("ok_runs")
            ok_rate = stability_check.get("ok_rate")
            dominant_status = stability_check.get("dominant_status")
            dominant_rate = stability_check.get("dominant_rate")
            shape_counts = stability_check.get("ok_panorama_shape_counts", [])
            dominant_shape = stability_check.get("dominant_ok_panorama_shape")
            dominant_shape_rate = stability_check.get("dominant_ok_panorama_shape_rate")
            shape_bucket_size = stability_check.get("ok_panorama_shape_bucket_size")
            shape_bucket_counts = stability_check.get("ok_panorama_shape_bucket_counts", [])
            dominant_shape_bucket = stability_check.get("dominant_ok_panorama_shape_bucket")
            dominant_shape_bucket_rate = stability_check.get("dominant_ok_panorama_shape_bucket_rate")
            output_consistent = stability_check.get("is_output_consistent")
            error_counts = stability_check.get("error_counts", {})

            if not isinstance(runs, int) or runs <= 0:
                errors.append(f"{scene_dir.name}: stability_check.runs must be a positive integer")
            if not isinstance(status_counts, dict) or not status_counts:
                errors.append(f"{scene_dir.name}: stability_check.status_counts must be a non-empty object")
            else:
                total_runs = sum(int(value) for value in status_counts.values())
                if isinstance(runs, int) and total_runs != runs:
                    errors.append(f"{scene_dir.name}: stability_check.status_counts does not sum to runs")
                if ok_runs != int(status_counts.get("OK", 0)):
                    errors.append(f"{scene_dir.name}: stability_check.ok_runs mismatch")
                if isinstance(runs, int) and runs > 0:
                    expected_ok_rate = round(float(int(status_counts.get('OK', 0)) / runs), 3)
                    if ok_rate != expected_ok_rate:
                        errors.append(f"{scene_dir.name}: stability_check.ok_rate mismatch")
                if dominant_status not in status_counts:
                    errors.append(f"{scene_dir.name}: stability_check.dominant_status missing from status_counts")
                elif isinstance(runs, int) and runs > 0:
                    expected_dominant_rate = round(float(int(status_counts[dominant_status]) / runs), 3)
                    if dominant_rate != expected_dominant_rate:
                        errors.append(f"{scene_dir.name}: stability_check.dominant_rate mismatch")
            if not isinstance(shape_counts, list):
                errors.append(f"{scene_dir.name}: stability_check.ok_panorama_shape_counts must be a list")
            else:
                counted_ok_runs = sum(int(row.get("count", 0)) for row in shape_counts if isinstance(row, dict))
                if counted_ok_runs > ok_runs:
                    errors.append(f"{scene_dir.name}: stability_check.ok_panorama_shape_counts exceeds ok_runs")
            has_bucket_fields = (
                "ok_panorama_shape_bucket_size" in stability_check
                or "ok_panorama_shape_bucket_counts" in stability_check
                or "dominant_ok_panorama_shape_bucket" in stability_check
                or "dominant_ok_panorama_shape_bucket_rate" in stability_check
            )
            if has_bucket_fields:
                if not isinstance(shape_bucket_counts, list):
                    errors.append(f"{scene_dir.name}: stability_check.ok_panorama_shape_bucket_counts must be a list")
                else:
                    counted_bucket_ok_runs = sum(int(row.get("count", 0)) for row in shape_bucket_counts if isinstance(row, dict))
                    if counted_bucket_ok_runs != ok_runs:
                        errors.append(f"{scene_dir.name}: stability_check.ok_panorama_shape_bucket_counts does not sum to ok_runs")
                if ok_runs == 0:
                    if (
                        dominant_shape is not None
                        or dominant_shape_rate is not None
                        or dominant_shape_bucket is not None
                        or dominant_shape_bucket_rate is not None
                        or output_consistent is not None
                    ):
                        errors.append(f"{scene_dir.name}: stability_check output fields must be null when ok_runs is 0")
                else:
                    if not isinstance(dominant_shape, dict) or "width" not in dominant_shape or "height" not in dominant_shape:
                        errors.append(f"{scene_dir.name}: stability_check.dominant_ok_panorama_shape is invalid")
                    if not isinstance(dominant_shape_rate, (int, float)):
                        errors.append(f"{scene_dir.name}: stability_check.dominant_ok_panorama_shape_rate is invalid")
                    if has_bucket_fields:
                        if not isinstance(shape_bucket_size, int) or shape_bucket_size <= 0:
                            errors.append(f"{scene_dir.name}: stability_check.ok_panorama_shape_bucket_size is invalid")
                        if not isinstance(dominant_shape_bucket, dict) or "width" not in dominant_shape_bucket or "height" not in dominant_shape_bucket:
                            errors.append(f"{scene_dir.name}: stability_check.dominant_ok_panorama_shape_bucket is invalid")
                        if not isinstance(dominant_shape_bucket_rate, (int, float)):
                            errors.append(f"{scene_dir.name}: stability_check.dominant_ok_panorama_shape_bucket_rate is invalid")
                    if not isinstance(output_consistent, bool):
                        errors.append(f"{scene_dir.name}: stability_check.is_output_consistent must be boolean when ok_runs > 0")
            if not isinstance(error_counts, dict):
                errors.append(f"{scene_dir.name}: stability_check.error_counts must be an object")

    category = meta.get("category")
    stitcher_status = audit_summary.get("stitcher_status")
    if category == "success" and stitcher_status != "OK":
        warnings.append(f"{scene_dir.name}: category=success but stitcher_status={stitcher_status}")
    if category == "failure" and stitcher_status == "OK":
        warnings.append(f"{scene_dir.name}: category=failure but stitcher_status=OK")
    if isinstance(stability_check, dict):
        ok_rate = stability_check.get("ok_rate")
        if category == "success" and isinstance(ok_rate, (int, float)) and ok_rate < 0.8:
            warnings.append(f"{scene_dir.name}: category=success but stability ok_rate={ok_rate}")
        if category == "failure" and isinstance(ok_rate, (int, float)) and ok_rate > 0.2:
            warnings.append(f"{scene_dir.name}: category=failure but stability ok_rate={ok_rate}")

    return errors, warnings, meta


def verify_images(root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALIDATE_IMAGE_EXTS:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # pragma: no cover - defensive
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {exc}")
    return failures


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    all_errors: list[str] = []
    all_warnings: list[str] = []
    category_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    total_images = 0
    total_reference_images = 0

    for scene_dir in list_scene_dirs(root):
        errors, warnings, meta = validate_scene(scene_dir)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        if meta:
            category_counter.update([str(meta.get("category", "unknown"))])
            status_counter.update([str(meta.get("audit_summary", {}).get("stitcher_status", "unknown"))])
            total_images += int(meta.get("num_images", 0))
            total_reference_images += len(meta.get("reference_files", []))

    image_failures: list[str] = []
    if args.verify_images:
        image_failures = verify_images(root)
        all_errors.extend(image_failures)

    print(f"Scenes checked: {sum(1 for _ in list_scene_dirs(root))}")
    print(f"Ordered images: {total_images}")
    print(f"Reference images: {total_reference_images}")
    print("Category counts:", dict(category_counter))
    print("Stitcher status counts:", dict(status_counter))

    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"- {warning}")

    if all_errors:
        print("\nErrors:")
        for error in all_errors:
            print(f"- {error}")
        return 1

    print("\nDataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

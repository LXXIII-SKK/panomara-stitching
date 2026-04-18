from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_utils.panorama_dataset import list_all_image_files, list_scene_dirs, ordered_scene_files

IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import cv2
    from project_utils.preprocessing import (
        BRIGHTNESS_RECOMMENDATIONS,
        CLAHE_RECOMMENDATIONS,
        DROP_RECOMMENDATIONS,
        SHARPEN_RECOMMENDATIONS,
        PreprocessConfig,
        load_audit_image_recommendations,
        load_bgr,
        preprocess_color_image,
        preprocess_feature_image,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    cv2 = None
    BRIGHTNESS_RECOMMENDATIONS = set()
    CLAHE_RECOMMENDATIONS = set()
    DROP_RECOMMENDATIONS = set()
    SHARPEN_RECOMMENDATIONS = set()
    PreprocessConfig = None
    load_audit_image_recommendations = None
    load_bgr = None
    preprocess_color_image = None
    preprocess_feature_image = None
    IMPORT_ERROR = exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply preprocessing to selected panorama dataset images or the full database.",
        epilog=(
            "Examples:\n"
            "  python scripts/apply_preprocessing.py\n"
            "  python scripts/apply_preprocessing.py --scene scene_04 --scene scene_21\n"
            "  python scripts/apply_preprocessing.py --image scene_04/img_01.jpg --image scene_30/img_03.jpg\n"
            "  python scripts/apply_preprocessing.py --skip-scene scene_14 --skip-image scene_29/img_08.jpg\n"
            "  python scripts/apply_preprocessing.py --ordered-only --profile baseline --output-kind both\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Root folder containing scene directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed",
        help="Destination root where processed outputs will be written.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Process every image inside this scene. Repeat the flag for multiple scenes.",
    )
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        help="Process one specific image. Use a path relative to --root, like scene_04/img_01.jpg.",
    )
    parser.add_argument(
        "--skip-scene",
        action="append",
        dest="skip_scenes",
        help="Skip an entire scene while processing a larger selection.",
    )
    parser.add_argument(
        "--skip-image",
        action="append",
        dest="skip_images",
        help="Skip one specific image. Use a path relative to --root.",
    )
    parser.add_argument(
        "--ordered-only",
        action="store_true",
        help="Only process ordered stitch inputs from each scene, not excluded reference files.",
    )
    parser.add_argument(
        "--profile",
        choices=["baseline", "enhanced", "audit_auto"],
        default="audit_auto",
        help="Preprocessing profile. 'audit_auto' follows audit recommendations when available.",
    )
    parser.add_argument(
        "--output-kind",
        choices=["gray", "color", "both"],
        default="gray",
        help="Whether to save feature-ready grayscale outputs, color-enhanced outputs, or both.",
    )
    parser.add_argument(
        "--audit-image-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "audit" / "image_metrics.csv",
        help="CSV used by the audit_auto profile.",
    )
    parser.add_argument("--max-width", type=int, default=1600, help="Resize images so width does not exceed this value.")
    parser.add_argument("--gaussian-kernel", type=int, default=3, help="Gaussian blur kernel size.")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clipLimit parameter.")
    parser.add_argument("--clahe-tile-grid", type=int, default=8, help="CLAHE tile grid size.")
    parser.add_argument("--denoise", action="store_true", help="Force non-local means denoising.")
    parser.add_argument("--unsharp", action="store_true", help="Force an unsharp-mask finishing step.")
    parser.add_argument(
        "--skip-drop-recommended",
        action="store_true",
        help="Skip images whose audit recommends retaking or dropping them instead of preprocessing.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve targets and print what would happen without writing files.")
    return parser


def resolve_image_arg(root: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    options = []
    if candidate.is_absolute():
        options.append(candidate)
    else:
        options.append(root / candidate)
        options.append(PROJECT_ROOT / candidate)

    for option in options:
        resolved = option.resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"Could not resolve image path: {raw_value}")


def collect_scene_images(root: Path, scene_name: str, ordered_only: bool) -> list[Path]:
    scene_dir = (root / scene_name).resolve()
    if not scene_dir.exists() or not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene not found: {scene_name}")
    if ordered_only:
        ordered, _, _, _ = ordered_scene_files(scene_dir)
        return ordered
    return list_all_image_files(scene_dir)


def collect_target_images(args) -> list[Path]:
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    skip_scene_names = set(args.skip_scenes or [])
    skip_image_paths = {resolve_image_arg(root, value).resolve() for value in (args.skip_images or [])}

    targets: set[Path] = set()

    explicit_scenes = list(args.scenes or [])
    explicit_images = list(args.images or [])
    process_everything = not explicit_scenes and not explicit_images

    if process_everything:
        for scene_dir in list_scene_dirs(root):
            if scene_dir.name in skip_scene_names:
                continue
            scene_files = collect_scene_images(root, scene_dir.name, args.ordered_only)
            targets.update(scene_files)
    else:
        for scene_name in explicit_scenes:
            if scene_name in skip_scene_names:
                continue
            scene_files = collect_scene_images(root, scene_name, args.ordered_only)
            targets.update(scene_files)
        for image_value in explicit_images:
            targets.add(resolve_image_arg(root, image_value).resolve())

    final_targets = []
    for path in sorted(targets):
        if path in skip_image_paths:
            continue
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Image is outside the dataset root: {path}") from exc
        if path.parent.name in skip_scene_names:
            continue
        final_targets.append(path)

    return final_targets


def build_config(args, recommendations: set[str]) -> PreprocessConfig:
    enable_brightness = args.profile in {"enhanced", "audit_auto"}
    enable_clahe = True
    enable_unsharp = args.unsharp

    if args.profile == "baseline":
        enable_brightness = False
    elif args.profile == "audit_auto":
        enable_brightness = bool(recommendations & (BRIGHTNESS_RECOMMENDATIONS | CLAHE_RECOMMENDATIONS))
        enable_unsharp = enable_unsharp or bool(recommendations & SHARPEN_RECOMMENDATIONS)

    return PreprocessConfig(
        max_width=args.max_width,
        gaussian_kernel=args.gaussian_kernel,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid,
        enable_clahe=enable_clahe,
        enable_brightness_normalization=enable_brightness,
        enable_denoise=args.denoise,
        enable_unsharp=enable_unsharp,
    )


def output_paths(output_root: Path, input_path: Path, dataset_root: Path, output_kind: str) -> dict[str, Path]:
    relative = input_path.resolve().relative_to(dataset_root.resolve())
    stem = relative.with_suffix("")
    paths: dict[str, Path] = {}
    if output_kind in {"gray", "both"}:
        paths["gray"] = (output_root / "feature_gray" / stem).with_suffix(".png")
    if output_kind in {"color", "both"}:
        paths["color"] = (output_root / "color_enhanced" / stem).with_suffix(".png")
    return paths


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    ensure_parent(manifest_path)
    fieldnames = [
        "scene_id",
        "input_path",
        "recommendations",
        "applied_steps_gray",
        "applied_steps_color",
        "output_gray",
        "output_color",
        "gamma_gray",
        "gamma_color",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    if IMPORT_ERROR is not None:
        raise SystemExit(
            "OpenCV preprocessing dependencies are not installed in this Python environment. "
            "Install requirements.txt first, then rerun the command."
        )

    dataset_root = args.root.resolve()
    output_root = args.output_root.resolve()

    target_images = collect_target_images(args)
    if not target_images:
        print("No images matched the requested selection.")
        return 0

    audit_recommendations = {}
    if args.profile == "audit_auto":
        audit_recommendations = load_audit_image_recommendations(args.audit_image_metrics.resolve())
        if not audit_recommendations:
            print("Audit CSV not found or empty; audit_auto will fall back to the baseline CLAHE pipeline.")

    manifest_rows: list[dict[str, str]] = []
    processed_count = 0
    skipped_count = 0

    for image_path in target_images:
        recommendations = audit_recommendations.get(image_path.resolve(), set())
        if args.skip_drop_recommended and recommendations & DROP_RECOMMENDATIONS:
            skipped_count += 1
            print(f"SKIP {image_path.relative_to(dataset_root)}  audit={sorted(recommendations)}")
            continue

        config = build_config(args, recommendations)
        out_paths = output_paths(output_root, image_path, dataset_root, args.output_kind)
        existing_block = [path for path in out_paths.values() if path.exists()]
        if existing_block and not args.overwrite:
            skipped_count += 1
            print(f"SKIP {image_path.relative_to(dataset_root)}  existing outputs present (use --overwrite)")
            continue

        image_bgr = load_bgr(image_path)
        gray_result = None
        color_result = None

        if args.output_kind in {"gray", "both"}:
            gray_result = preprocess_feature_image(image_bgr, config=config)
        if args.output_kind in {"color", "both"}:
            color_result = preprocess_color_image(image_bgr, config=config)

        if args.dry_run:
            processed_count += 1
            print(
                f"DRY  {image_path.relative_to(dataset_root)}  "
                f"recs={sorted(recommendations)}  "
                f"gray_steps={gray_result['applied_steps'] if gray_result else []}  "
                f"color_steps={color_result['applied_steps'] if color_result else []}"
            )
            continue

        if gray_result is not None:
            gray_path = out_paths["gray"]
            ensure_parent(gray_path)
            cv2.imwrite(str(gray_path), gray_result["final"])

        if color_result is not None:
            color_path = out_paths["color"]
            ensure_parent(color_path)
            cv2.imwrite(str(color_path), color_result["final"])

        manifest_rows.append(
            {
                "scene_id": image_path.parent.name,
                "input_path": str(image_path),
                "recommendations": ", ".join(sorted(recommendations)),
                "applied_steps_gray": ", ".join(gray_result["applied_steps"]) if gray_result else "",
                "applied_steps_color": ", ".join(color_result["applied_steps"]) if color_result else "",
                "output_gray": str(out_paths["gray"]) if "gray" in out_paths else "",
                "output_color": str(out_paths["color"]) if "color" in out_paths else "",
                "gamma_gray": f"{gray_result['gamma']:.3f}" if gray_result else "",
                "gamma_color": f"{color_result['gamma']:.3f}" if color_result else "",
            }
        )
        processed_count += 1
        print(f"OK   {image_path.relative_to(dataset_root)}")

    if not args.dry_run and manifest_rows:
        write_manifest(output_root / "preprocess_manifest.csv", manifest_rows)

    print(
        f"Done. matched={len(target_images)} processed={processed_count} skipped={skipped_count} "
        f"output_root={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

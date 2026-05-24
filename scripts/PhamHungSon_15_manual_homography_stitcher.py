from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_utils.panorama_dataset import list_scene_dirs
from scripts.PhamHungSon_15_portable_panorama_pipeline import (
    PanoramaConfig,
    apply_profile,
    load_pair_method_map,
    normalize_method,
    normalize_motion_model,
    stitch_scene_folder,
)


SPLIT_ROOT = PROJECT_ROOT / "data" / "split"
FEATURE_CACHE_ROOT = PROJECT_ROOT / "data" / "feature_extract"
BATCH_METRICS = PROJECT_ROOT / "outputs" / "batch_feature_matching" / "pair_metrics.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "manual_homography_stitcher"
DEFAULT_SHOWCASE_SPLITS = ["test", "failure_analysis"]
SPLIT_ALIASES = {
    "all": "all",
    "development": "development",
    "dev": "development",
    "test": "test",
    "failure": "failure_analysis",
    "failure_analysis": "failure_analysis",
}


def scene_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return int(digits) if digits else 10**9, path.name


def normalize_split(value: str) -> str:
    key = value.strip().lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(f"Unknown split: {value}")
    return SPLIT_ALIASES[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the project manual stitcher on one scene, one split, or the full split database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scope = parser.add_argument_group("Selection")
    scope.add_argument("--split", action="append", help="development, test, failure_analysis, or all. Repeatable.")
    scope.add_argument("--scene", action="append", dest="scenes", help="Scene id to run, e.g. scene_15. Repeatable.")
    scope.add_argument("--scene-folder", action="append", type=Path, help="Custom scene folder outside data/split. Repeatable.")
    scope.add_argument("--limit-scenes", type=int, default=0, help="Optional maximum number of resolved scenes.")

    output = parser.add_argument_group("Output")
    output.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--dry-run", action="store_true")
    output.add_argument("--save-visualizations", action="store_true", help="Save keypoint/match/inlier/warp previews for every pair and descriptor.")
    output.add_argument("--save-score-table", action="store_true", help="Save descriptor score CSV/PNG tables for every pair.")
    output.add_argument("--save-debug", action="store_true", help="Shortcut that saves score tables and visualizations.")
    output.add_argument("--diagnostic-method", action="append", dest="diagnostics_methods", help="Descriptor to visualize: ORB, AKAZE, HARRIS_HOG, SIFT, or all. Repeatable.")
    output.add_argument("--visualization-max-matches", type=int, default=80)
    output.add_argument("--visualization-max-keypoints", type=int, default=1000)
    output.add_argument("--visualization-max-width", type=int, default=1800)
    output.add_argument("--visualization-jpeg-quality", type=int, default=85)

    method = parser.add_argument_group("Stitching")
    method.add_argument("--engine", choices=["manual", "opencv"], default="manual")
    method.add_argument("--profile", choices=["fast", "balanced", "quality"], default="balanced")
    method.add_argument("--method", default="auto", help="auto, ORB, AKAZE, HARRIS_HOG, or SIFT.")
    method.add_argument("--candidate-method", action="append", dest="candidate_methods", help="Candidate method for auto mode. Repeatable.")
    method.add_argument("--feature-source", choices=["auto", "cache", "compute"], default="cache")
    method.add_argument("--use-batch-metrics", action="store_true", default=True)
    method.add_argument("--no-batch-metrics", action="store_false", dest="use_batch_metrics")
    method.add_argument("--batch-metrics", type=Path, default=BATCH_METRICS)
    method.add_argument("--feature-cache-root", type=Path, default=FEATURE_CACHE_ROOT)

    mobile = parser.add_argument_group("Portable/mobile tuning")
    mobile.add_argument("--work-width", type=int, default=1280)
    mobile.add_argument("--max-features", type=int, default=3000)
    mobile.add_argument("--ratio-test", type=float, default=0.75)
    mobile.add_argument("--ransac-threshold", type=float, default=4.0)
    mobile.add_argument("--min-good-matches", type=int, default=12)
    mobile.add_argument("--min-inliers", type=int, default=16)
    mobile.add_argument("--min-inlier-ratio", type=float, default=0.18)
    mobile.add_argument("--blend-mode", choices=["average", "feather", "overwrite"], default="average")
    mobile.add_argument("--anchor", default="middle")
    mobile.add_argument(
        "--manual-motion-model",
        choices=["translation", "similarity", "affine", "homography"],
        default="affine",
        help="Geometric model for the manual stitcher. Affine is the safer default; homography preserves the original projective chain for comparison.",
    )
    mobile.add_argument("--allow-partial", action="store_true")
    mobile.add_argument("--no-crop", action="store_true")
    mobile.add_argument("--max-canvas-megapixels", type=float, default=24.0)
    mobile.add_argument("--max-canvas-side", type=int, default=12000)
    mobile.add_argument("--preprocess", choices=["none", "gray", "clahe"], default="clahe")
    mobile.add_argument("--enable-gamma", action="store_true")
    mobile.add_argument("--harris-max-corners", type=int, default=1500)
    mobile.add_argument("--harris-quality", type=float, default=0.01)
    mobile.add_argument("--harris-min-distance", type=float, default=8.0)
    mobile.add_argument("--hog-patch-size", type=int, default=32)
    mobile.add_argument("--hog-cells", type=int, default=4)
    mobile.add_argument("--hog-bins", type=int, default=8)
    mobile.add_argument("--stitcher-mode", choices=["PANORAMA", "SCANS"], default="PANORAMA")
    mobile.add_argument("--reverse-order", action="store_true")
    mobile.add_argument("--max-images", type=int, default=0)
    mobile.add_argument("--skip-every", type=int, default=1)
    return parser


def available_splits() -> list[str]:
    return [path.name for path in sorted(SPLIT_ROOT.iterdir()) if path.is_dir()]


def resolve_split_list(raw_splits: list[str] | None) -> list[str]:
    if not raw_splits:
        return DEFAULT_SHOWCASE_SPLITS[:]
    resolved: list[str] = []
    for raw_split in raw_splits:
        split = normalize_split(raw_split)
        if split == "all":
            resolved.extend(available_splits())
        else:
            resolved.append(split)
    unique: list[str] = []
    for split in resolved:
        if split not in unique:
            unique.append(split)
    return unique


def resolve_scene_records(args: argparse.Namespace) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    requested_scenes = set(args.scenes or [])
    split_names = resolve_split_list(args.split) if args.split else []
    if not split_names and not args.scene_folder:
        split_names = DEFAULT_SHOWCASE_SPLITS[:]

    for split_name in split_names:
        split_dir = SPLIT_ROOT / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")
        scene_dirs = sorted(list_scene_dirs(split_dir), key=scene_sort_key)
        if requested_scenes:
            scene_dirs = [scene_dir for scene_dir in scene_dirs if scene_dir.name in requested_scenes]
        records.extend((split_name, scene_dir) for scene_dir in scene_dirs)

    for scene_folder in args.scene_folder or []:
        if not scene_folder.is_dir():
            raise FileNotFoundError(f"Custom scene folder not found: {scene_folder}")
        records.append(("custom", scene_folder.resolve()))

    if args.limit_scenes and args.limit_scenes > 0:
        records = records[: args.limit_scenes]
    return records


def config_for_scene(args: argparse.Namespace, split_name: str, scene_dir: Path) -> PanoramaConfig:
    method = normalize_method(args.method)
    candidate_methods = [normalize_method(value) for value in (args.candidate_methods or [])]
    config = PanoramaConfig(
        engine=args.engine,
        profile=args.profile,
        method=method,
        candidate_methods=candidate_methods,
        work_width=args.work_width,
        max_features=args.max_features,
        ratio_test=args.ratio_test,
        ransac_threshold=args.ransac_threshold,
        min_good_matches=args.min_good_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        blend_mode=args.blend_mode,
        anchor=args.anchor,
        manual_motion_model=normalize_motion_model(args.manual_motion_model),
        crop=not args.no_crop,
        allow_partial=args.allow_partial,
        max_canvas_megapixels=args.max_canvas_megapixels,
        max_canvas_side=args.max_canvas_side,
        preprocess=args.preprocess,
        enable_gamma=args.enable_gamma,
        stitcher_mode=args.stitcher_mode,
        image_order="meta",
        reverse_order=args.reverse_order,
        max_images=args.max_images,
        skip_every=args.skip_every,
        feature_cache_root=str(args.feature_cache_root),
        split_name="" if split_name == "custom" else split_name,
        scene_id=scene_dir.name,
        prefer_cache=args.feature_source in {"auto", "cache"} and split_name != "custom",
        save_debug=args.save_debug,
        save_pair_visualizations=args.save_visualizations,
        save_score_table=args.save_score_table,
        visualization_dir=str(args.output_root / "diagnostics" / split_name / scene_dir.name),
        diagnostics_methods=[value for value in (args.diagnostics_methods or [])],
        visualization_max_matches=args.visualization_max_matches,
        visualization_max_keypoints=args.visualization_max_keypoints,
        visualization_max_width=args.visualization_max_width,
        visualization_jpeg_quality=args.visualization_jpeg_quality,
        harris_max_corners=args.harris_max_corners,
        harris_quality=args.harris_quality,
        harris_min_distance=args.harris_min_distance,
        hog_patch_size=args.hog_patch_size,
        hog_cells=args.hog_cells,
        hog_bins=args.hog_bins,
    )
    config = apply_profile(config)
    if args.use_batch_metrics and split_name != "custom" and config.method == "auto":
        config.pair_method_map = load_pair_method_map(
            args.batch_metrics,
            split_name,
            scene_dir.name,
            config.candidate_methods,
        )
    return config


def output_paths(args: argparse.Namespace, split_name: str, scene_dir: Path) -> tuple[Path, Path]:
    stem = f"{scene_dir.name}_manual_homography"
    panorama_path = args.output_root / "panoramas" / split_name / f"{stem}.jpg"
    log_path = args.output_root / "logs" / split_name / f"{stem}.json"
    return panorama_path, log_path


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "scene_id",
        "status",
        "engine",
        "profile",
        "method",
        "manual_motion_model",
        "image_count",
        "panorama_path",
        "log_path",
        "runtime_sec",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = build_parser().parse_args()
    records = resolve_scene_records(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Resolved scene count: {len(records)}")
    manifest_rows: list[dict[str, Any]] = []
    for index, (split_name, scene_dir) in enumerate(records, start=1):
        panorama_path, log_path = output_paths(args, split_name, scene_dir)
        config = config_for_scene(args, split_name, scene_dir)
        print(f"[{index}/{len(records)}] {split_name}/{scene_dir.name} -> {panorama_path}")

        if args.dry_run:
            manifest_rows.append(
                {
                    "split": split_name,
                    "scene_id": scene_dir.name,
                    "status": "dry_run",
                    "engine": config.engine,
                    "profile": config.profile,
                    "method": config.method,
                    "manual_motion_model": config.manual_motion_model,
                    "panorama_path": str(panorama_path),
                    "log_path": str(log_path),
                }
            )
            continue

        if panorama_path.exists() and log_path.exists() and not args.overwrite:
            manifest_rows.append(
                {
                    "split": split_name,
                    "scene_id": scene_dir.name,
                    "status": "existing",
                    "engine": config.engine,
                    "profile": config.profile,
                    "method": config.method,
                    "manual_motion_model": config.manual_motion_model,
                    "panorama_path": str(panorama_path),
                    "log_path": str(log_path),
                }
            )
            continue

        start = time.perf_counter()
        try:
            payload = stitch_scene_folder(scene_dir, panorama_path, config, log_path)
            manifest_rows.append(
                {
                    "split": split_name,
                    "scene_id": scene_dir.name,
                    "status": payload.get("status", "ok"),
                    "engine": payload.get("engine", config.engine),
                    "profile": config.profile,
                    "method": config.method,
                    "manual_motion_model": config.manual_motion_model,
                    "image_count": payload.get("image_count", ""),
                    "panorama_path": str(panorama_path),
                    "log_path": str(log_path),
                    "runtime_sec": payload.get("runtime_sec", time.perf_counter() - start),
                    "error": "",
                }
            )
        except Exception as exc:
            error_payload = {
                "status": "failed",
                "split": split_name,
                "scene_id": scene_dir.name,
                "scene_dir": str(scene_dir),
                "panorama_path": str(panorama_path),
                "error": str(exc),
                "config": config.__dict__,
                "runtime_sec": time.perf_counter() - start,
            }
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
            manifest_rows.append(
                {
                    "split": split_name,
                    "scene_id": scene_dir.name,
                    "status": "failed",
                    "engine": config.engine,
                    "profile": config.profile,
                    "method": config.method,
                    "manual_motion_model": config.manual_motion_model,
                    "panorama_path": str(panorama_path),
                    "log_path": str(log_path),
                    "runtime_sec": error_payload["runtime_sec"],
                    "error": str(exc),
                }
            )
            print(f"  failed: {exc}")

    manifest_path = args.output_root / "manual_homography_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"Manifest: {manifest_path}")
    counts: dict[str, int] = {}
    for row in manifest_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("Status counts:", counts)


if __name__ == "__main__":
    main()

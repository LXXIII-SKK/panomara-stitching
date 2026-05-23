from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL_ROOT = PROJECT_ROOT / "outputs" / "manual_homography_stitcher"
DEFAULT_OPENCV_SUMMARY = PROJECT_ROOT / "outputs" / "openCV" / "logs" / "batch_summary_panorama.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_MANUAL_ROOT / "comparison"
DEFAULT_SHOWCASE_SPLITS = ["test", "failure_analysis"]
SPLIT_ALIASES = {
    "dev": "development",
    "development": "development",
    "test": "test",
    "failure": "failure_analysis",
    "failure_analysis": "failure_analysis",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the project manual geometry stitcher with OpenCV Stitcher outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manual-root", type=Path, default=DEFAULT_MANUAL_ROOT)
    parser.add_argument("--opencv-summary", type=Path, default=DEFAULT_OPENCV_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split",
        action="append",
        help="Optional split filter. Repeatable. Defaults to test and failure_analysis; use --split all to include development.",
    )
    parser.add_argument("--scene", action="append", help="Optional scene_id filter. Repeatable.")
    parser.add_argument("--side-by-side-limit", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def path_from_text(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path


def normalize_requested_splits(raw_splits: list[str] | None) -> set[str] | None:
    if not raw_splits:
        return set(DEFAULT_SHOWCASE_SPLITS)
    normalized: set[str] = set()
    for raw_split in raw_splits:
        token = raw_split.strip().lower()
        if token == "all":
            return None
        normalized.add(SPLIT_ALIASES.get(token, token))
    return normalized


def image_info(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "exists": False,
            "width": "",
            "height": "",
            "area_px": "",
            "file_size_mb": "",
        }
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "exists": False,
            "width": "",
            "height": "",
            "area_px": "",
            "file_size_mb": "",
        }
    height, width = image.shape[:2]
    return {
        "exists": True,
        "width": width,
        "height": height,
        "area_px": width * height,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
    }


def read_opencv_rows(summary_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            split_name = (raw.get("split") or "").strip()
            scene_id = (raw.get("scene_id") or "").strip()
            if not split_name or not scene_id:
                continue
            log_path = path_from_text(raw.get("log_path"))
            log_payload = read_json(log_path) if log_path else {}
            panorama_path = path_from_text(raw.get("panorama_path") or log_payload.get("panorama_path"))
            info = image_info(panorama_path)
            num_images = parse_int(raw.get("num_images") or log_payload.get("num_images"))
            rows[(split_name, scene_id)] = {
                "split": split_name,
                "scene_id": scene_id,
                "opencv_status_code": raw.get("status_code", ""),
                "opencv_status_name": raw.get("status_name", ""),
                "opencv_meta_stitcher_status": raw.get("meta_stitcher_status", ""),
                "opencv_num_images": num_images if num_images is not None else "",
                "opencv_category": raw.get("meta_category", ""),
                "opencv_difficulty": raw.get("meta_difficulty", ""),
                "opencv_stability_label": raw.get("meta_stability_label", ""),
                "opencv_output_consistent": raw.get("meta_output_consistent", ""),
                "opencv_panorama_path": str(panorama_path) if panorama_path else "",
                "opencv_log_path": str(log_path) if log_path else "",
                "opencv_exists": info["exists"],
                "opencv_width": info["width"],
                "opencv_height": info["height"],
                "opencv_area_px": info["area_px"],
                "opencv_file_size_mb": info["file_size_mb"],
                "opencv_runtime_sec": log_payload.get("runtime_sec", ""),
            }
    return rows


def read_manual_rows(manual_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for log_path in sorted((manual_root / "logs").glob("**/*_manual_homography.json")):
        payload = read_json(log_path)
        split_name = str(payload.get("split_name") or payload.get("split") or log_path.parent.name)
        scene_id = str(payload.get("scene_id") or log_path.stem.replace("_manual_homography", ""))
        panorama_path = path_from_text(payload.get("output_path") or payload.get("panorama_path"))
        info = image_info(panorama_path)
        pairs = payload.get("pairs") if isinstance(payload.get("pairs"), list) else []
        successful_pairs = sum(1 for pair in pairs if pair.get("status") == "success")
        failed_pairs = sum(1 for pair in pairs if pair.get("status") != "success")
        inliers = [parse_float(pair.get("inliers")) for pair in pairs]
        inlier_ratios = [parse_float(pair.get("inlier_ratio")) for pair in pairs]
        reproj_errors = [parse_float(pair.get("reprojection_error_mean")) for pair in pairs]
        inliers = [value for value in inliers if value is not None]
        inlier_ratios = [value for value in inlier_ratios if value is not None]
        reproj_errors = [value for value in reproj_errors if value is not None]
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        rows[(split_name, scene_id)] = {
            "split": split_name,
            "scene_id": scene_id,
            "manual_status": payload.get("status", ""),
            "manual_engine": payload.get("engine", "manual"),
            "manual_profile": config.get("profile", ""),
            "manual_method": config.get("method", ""),
            "manual_motion_model": config.get("manual_motion_model", ""),
            "manual_candidate_methods": ",".join(config.get("candidate_methods", [])),
            "manual_blend_mode": config.get("blend_mode", ""),
            "manual_allow_partial": config.get("allow_partial", ""),
            "manual_image_count": payload.get("image_count", ""),
            "manual_used_image_offset": payload.get("used_image_offset", ""),
            "manual_successful_pairs": successful_pairs,
            "manual_failed_pairs": failed_pairs,
            "manual_min_inliers": min(inliers) if inliers else "",
            "manual_mean_inlier_ratio": round(sum(inlier_ratios) / len(inlier_ratios), 4) if inlier_ratios else "",
            "manual_mean_reprojection_error": round(sum(reproj_errors) / len(reproj_errors), 4) if reproj_errors else "",
            "manual_panorama_path": str(panorama_path) if panorama_path else "",
            "manual_log_path": str(log_path),
            "manual_exists": info["exists"],
            "manual_width": info["width"],
            "manual_height": info["height"],
            "manual_area_px": info["area_px"],
            "manual_file_size_mb": info["file_size_mb"],
            "manual_runtime_sec": payload.get("runtime_sec", ""),
            "manual_error": payload.get("error", ""),
        }
    return rows


def is_truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def status_label(row: dict[str, Any]) -> str:
    manual_ok = row.get("manual_status") == "ok" and is_truthy(row.get("manual_exists"))
    opencv_ok = row.get("opencv_status_name") == "OK" and is_truthy(row.get("opencv_exists"))
    manual_count = parse_int(row.get("manual_image_count"))
    opencv_count = parse_int(row.get("opencv_num_images"))
    manual_offset = parse_int(row.get("manual_used_image_offset")) or 0
    manual_partial = bool(
        manual_ok
        and (
            (manual_count is not None and opencv_count is not None and manual_count < opencv_count)
            or manual_offset > 0
        )
    )
    if manual_ok and opencv_ok and manual_partial:
        return "both_ok_manual_partial"
    if manual_ok and opencv_ok:
        return "both_ok"
    if manual_ok and not opencv_ok and manual_partial:
        return "manual_partial_only"
    if manual_ok and not opencv_ok:
        return "manual_only"
    if opencv_ok and not manual_ok:
        return "opencv_only_or_manual_missing"
    return "both_failed_or_missing"


def compare_rows(
    manual_rows: dict[tuple[str, str], dict[str, Any]],
    opencv_rows: dict[tuple[str, str], dict[str, Any]],
    splits: set[str] | None,
    scenes: set[str] | None,
) -> list[dict[str, Any]]:
    keys = sorted(set(manual_rows) | set(opencv_rows), key=lambda item: (item[0], int(item[1].split("_")[-1]) if item[1].split("_")[-1].isdigit() else item[1]))
    rows: list[dict[str, Any]] = []
    for split_name, scene_id in keys:
        if splits and split_name not in splits:
            continue
        if scenes and scene_id not in scenes:
            continue
        row: dict[str, Any] = {
            "split": split_name,
            "scene_id": scene_id,
        }
        row.update(opencv_rows.get((split_name, scene_id), {}))
        row.update(manual_rows.get((split_name, scene_id), {}))
        row.setdefault("opencv_status_name", "")
        row.setdefault("manual_status", "")
        row["comparison_status"] = status_label(row)
        manual_runtime = parse_float(row.get("manual_runtime_sec"))
        opencv_runtime = parse_float(row.get("opencv_runtime_sec"))
        row["runtime_delta_manual_minus_opencv"] = (
            round(manual_runtime - opencv_runtime, 4)
            if manual_runtime is not None and opencv_runtime is not None
            else ""
        )
        manual_area = parse_float(row.get("manual_area_px"))
        opencv_area = parse_float(row.get("opencv_area_px"))
        row["area_ratio_manual_over_opencv"] = (
            round(manual_area / opencv_area, 4)
            if manual_area is not None and opencv_area not in (None, 0)
            else ""
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "split",
        "scene_id",
        "comparison_status",
        "opencv_status_name",
        "manual_status",
        "opencv_num_images",
        "manual_image_count",
        "manual_used_image_offset",
        "opencv_category",
        "opencv_difficulty",
        "opencv_stability_label",
        "opencv_output_consistent",
        "manual_profile",
        "manual_method",
        "manual_motion_model",
        "manual_candidate_methods",
        "manual_blend_mode",
        "manual_allow_partial",
        "manual_successful_pairs",
        "manual_failed_pairs",
        "manual_min_inliers",
        "manual_mean_inlier_ratio",
        "manual_mean_reprojection_error",
        "opencv_width",
        "opencv_height",
        "opencv_area_px",
        "manual_width",
        "manual_height",
        "manual_area_px",
        "area_ratio_manual_over_opencv",
        "opencv_file_size_mb",
        "manual_file_size_mb",
        "opencv_runtime_sec",
        "manual_runtime_sec",
        "runtime_delta_manual_minus_opencv",
        "opencv_panorama_path",
        "manual_panorama_path",
        "opencv_log_path",
        "manual_log_path",
        "manual_error",
    ]
    all_fields = preferred[:]
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in all_fields})


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    status_counts = Counter(row["comparison_status"] for row in rows)
    split_counts = Counter((row["split"], row["comparison_status"]) for row in rows)
    lines = [
        "# Manual Geometry Stitcher vs OpenCV Stitcher",
        "",
        f"Compared scenes: {len(rows)}",
        "",
        "## Comparison Status Counts",
        "",
    ]
    for status, count in status_counts.most_common():
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Split Breakdown", ""])
    for (split_name, status), count in sorted(split_counts.items()):
        lines.append(f"- {split_name} / {status}: {count}")
    lines.extend(
        [
            "",
            "## Label Meaning",
            "",
            "- both_ok: both methods produced a panorama using the full available scene.",
            "- both_ok_manual_partial: both methods produced an output, but the manual output used fewer images or started after the first image.",
            "- manual_partial_only: OpenCV failed, while the manual pipeline produced a partial output from a valid contiguous sub-chain.",
            "- manual_only: only the manual pipeline produced an output; this still needs visual inspection because the manual method lacks global camera adjustment.",
            "- opencv_only_or_manual_missing: OpenCV produced an output but the manual run is missing or failed.",
            "- both_failed_or_missing: neither method produced a usable panorama output.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(output_dir: Path, rows: list[dict[str, Any]], side_by_side_limit: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plot_dir.glob("manual_vs_opencv*.png"):
        old_plot.unlink()
    for old_plot in plot_dir.glob("manual_runtime_by_scene.png"):
        old_plot.unlink()

    status_counts = Counter(row["comparison_status"] for row in rows)
    if status_counts:
        labels = list(status_counts.keys())
        values = [status_counts[label] for label in labels]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4))
        ax.bar(labels, values, color="#4c78a8")
        ax.set_ylabel("Scene count")
        ax.set_title("Manual Geometry vs OpenCV Stitcher")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(plot_dir / "manual_vs_opencv_status_counts.png", dpi=160)
        plt.close(fig)

    area_rows = [
        row
        for row in rows
        if parse_float(row.get("manual_area_px")) is not None and parse_float(row.get("opencv_area_px")) is not None
    ]
    if area_rows:
        labels = [f"{row['split']}/{row['scene_id']}" for row in area_rows]
        manual_values = [parse_float(row.get("manual_area_px")) / 1_000_000 for row in area_rows]
        opencv_values = [parse_float(row.get("opencv_area_px")) / 1_000_000 for row in area_rows]
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.5))
        ax.bar([i - 0.18 for i in x], opencv_values, width=0.36, label="OpenCV")
        ax.bar([i + 0.18 for i in x], manual_values, width=0.36, label="Manual")
        ax.set_ylabel("Panorama area (MP)")
        ax.set_title("Output Canvas Size")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "manual_vs_opencv_area.png", dpi=160)
        plt.close(fig)

    runtime_rows = [row for row in rows if parse_float(row.get("manual_runtime_sec")) is not None]
    if runtime_rows:
        labels = [f"{row['split']}/{row['scene_id']}" for row in runtime_rows]
        manual_values = [parse_float(row.get("manual_runtime_sec")) for row in runtime_rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4))
        ax.bar(labels, manual_values, color="#f58518")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("Manual Stitcher Runtime")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(plot_dir / "manual_runtime_by_scene.png", dpi=160)
        plt.close(fig)

    side_candidates = [
        row
        for row in rows
        if is_truthy(row.get("manual_exists")) or is_truthy(row.get("opencv_exists"))
    ]
    side_candidates = sorted(
        side_candidates,
        key=lambda row: (
            0 if row.get("comparison_status") != "both_ok" else 1,
            row.get("split", ""),
            row.get("scene_id", ""),
        ),
    )
    side_rows = side_candidates[: max(0, side_by_side_limit)]
    for row in side_rows:
        opencv_path = path_from_text(row.get("opencv_panorama_path"))
        manual_path = path_from_text(row.get("manual_panorama_path"))
        opencv_image = cv2.imread(str(opencv_path), cv2.IMREAD_COLOR) if opencv_path and opencv_path.exists() else None
        manual_image = cv2.imread(str(manual_path), cv2.IMREAD_COLOR) if manual_path and manual_path.exists() else None
        if opencv_image is None and manual_image is None:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(11, 7))
        if opencv_image is not None:
            axes[0].imshow(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"OpenCV Stitcher: {row['split']}/{row['scene_id']}")
        else:
            axes[0].text(
                0.5,
                0.5,
                f"OpenCV Stitcher\n{row.get('opencv_status_name', 'missing output')}",
                ha="center",
                va="center",
                fontsize=16,
            )
            axes[0].set_title(f"OpenCV Stitcher: {row['split']}/{row['scene_id']}")
        axes[0].axis("off")
        if manual_image is not None:
            axes[1].imshow(cv2.cvtColor(manual_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"Manual geometry: {row['comparison_status']}")
        else:
            axes[1].text(
                0.5,
                0.5,
                f"Manual geometry\n{row.get('manual_status', 'missing output')}",
                ha="center",
                va="center",
                fontsize=16,
            )
            axes[1].set_title(f"Manual geometry: {row['comparison_status']}")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(plot_dir / f"manual_vs_opencv_{row['split']}_{row['scene_id']}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    manual_rows = read_manual_rows(args.manual_root)
    opencv_rows = read_opencv_rows(args.opencv_summary)
    rows = compare_rows(
        manual_rows,
        opencv_rows,
        normalize_requested_splits(args.split),
        set(args.scene or []) or None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = args.output_dir / "manual_vs_opencv_comparison.csv"
    summary_md = args.output_dir / "manual_vs_opencv_summary.md"
    write_csv(comparison_csv, rows)
    write_summary(summary_md, rows)
    if not args.no_plots:
        plot_outputs(args.output_dir, rows, args.side_by_side_limit)

    counts = Counter(row["comparison_status"] for row in rows)
    print(f"Comparison rows: {len(rows)}")
    print(f"CSV: {comparison_csv}")
    print(f"Summary: {summary_md}")
    print("Status counts:", dict(counts))


if __name__ == "__main__":
    main()

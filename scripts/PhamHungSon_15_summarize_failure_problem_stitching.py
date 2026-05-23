from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data" / "split"
DEFAULT_COMPARISON_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "manual_homography_stitcher"
    / "comparison"
    / "manual_vs_opencv_comparison.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "failure_problem_analysis"

PROBLEM_FLAG_FIELDS = [
    "has_moving_objects",
    "has_repeated_patterns",
    "has_low_texture",
    "has_parallax",
    "has_exposure_change",
    "has_motion_blur",
    "has_insufficient_overlap",
]

PROBLEM_GLOSSARY = {
    "insufficient_overlap": "Adjacent images do not share enough reliable content.",
    "insufficient_global_connectivity": "The local pair chain does not form a reliable full-scene graph.",
    "global_stitch_failure": "Pair-level matches exist, but the global stitcher rejects the camera chain.",
    "low_texture": "Large smooth or repetitive regions provide too few stable keypoints.",
    "parallax": "Foreground and background shift differently because the camera was translated or rotated off-center.",
    "repeated_patterns": "Similar-looking structures can create ambiguous correspondences.",
    "sideways_scan": "The capture behaves like a translated scan rather than a pure rotational panorama.",
    "exposure_change": "Brightness/color changes between frames can reveal seams or reduce matcher consistency.",
    "wide_sweep": "A long sweep accumulates drift and makes global optimization harder.",
    "output_variation": "Repeated stitcher runs can produce noticeably different panorama footprints.",
    "stitch_instability": "The scene is sensitive to bundle adjustment or seam-placement decisions.",
    "moving_objects": "Objects move between frames and can appear duplicated or ghosted.",
    "motion_blur": "Blur weakens keypoints and makes seams/ghosting more visible.",
}

LIKELY_GLITCH_BY_PROBLEM = {
    "insufficient_overlap": "broken image chain, ERR_NEED_MORE_IMGS, missing regions, or forced narrow manual output",
    "insufficient_global_connectivity": "local pairs may connect, but the whole scene can still fail globally",
    "global_stitch_failure": "good adjacent matches can still be rejected during global camera estimation",
    "low_texture": "few stable anchors, unstable homography, stretched canvas, or weak confidence",
    "parallax": "ghosting, bent structures, doubled edges, and partial stitching because one homography cannot explain all depths",
    "repeated_patterns": "wrong correspondences or believable but incorrect alignment",
    "sideways_scan": "translation-like capture causes drift and perspective stretching in a manual transform chain",
    "exposure_change": "visible brightness seams and inconsistent panorama footprint",
    "wide_sweep": "accumulated drift and large-canvas output variation",
    "output_variation": "different runs can produce different crops or output sizes",
    "stitch_instability": "sensitive panorama footprint and non-repeatable global optimization",
    "moving_objects": "ghosting or duplicated moving cars/people",
    "motion_blur": "lower keypoint quality, blurred seams, and unstable matching",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join failure-analysis scene problem tags with manual/OpenCV stitching results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="failure_analysis")
    parser.add_argument("--scene", action="append", help="Optional scene_id filter. Repeatable.")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_comparison_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("split", ""), row.get("scene_id", "")): row
            for row in csv.DictReader(handle)
        }


def clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def problem_flags(meta: dict[str, Any]) -> list[str]:
    return [field.removeprefix("has_") for field in PROBLEM_FLAG_FIELDS if meta.get(field) is True]


def merged_problem_tags(meta: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for tag in clean_list(meta.get("issues")) + problem_flags(meta):
        if tag not in ordered:
            ordered.append(tag)
    return ordered


def shape_text(row: dict[str, str], prefix: str) -> str:
    width = row.get(f"{prefix}_width", "")
    height = row.get(f"{prefix}_height", "")
    if not width or not height:
        return ""
    return f"{width}x{height}"


def manual_coverage(row: dict[str, str]) -> str:
    used = row.get("manual_image_count", "")
    total = row.get("opencv_num_images", "")
    if used and total:
        return f"{used}/{total}"
    return used


def observed_result(row: dict[str, str]) -> str:
    comparison_status = row.get("comparison_status", "")
    opencv_status = row.get("opencv_status_name", "")
    manual_status = row.get("manual_status", "")
    coverage = manual_coverage(row)
    if comparison_status == "both_ok":
        return "OpenCV and manual both produced panoramas; compare seams, canvas size, drift, and ghosting visually."
    if comparison_status == "manual_only":
        return f"OpenCV returned {opencv_status}; manual produced a full-chain output ({coverage})."
    if comparison_status == "manual_partial_only":
        return f"OpenCV returned {opencv_status}; manual saved only a contiguous partial chain ({coverage})."
    if manual_status:
        return f"OpenCV status={opencv_status}; manual status={manual_status}."
    return "No stitched comparison result was found for this scene."


def likely_glitch(problem_tags: list[str], row: dict[str, str]) -> str:
    notes = [LIKELY_GLITCH_BY_PROBLEM[tag] for tag in problem_tags if tag in LIKELY_GLITCH_BY_PROBLEM]
    if row.get("comparison_status") == "manual_partial_only":
        notes.append("partial output means at least one adjacent transition was rejected")
    if row.get("comparison_status") == "both_ok" and row.get("opencv_stability_label") == "success_with_output_variation":
        notes.append("successful stitch still needs visual inspection because output footprint varies")
    unique: list[str] = []
    for note in notes:
        if note not in unique:
            unique.append(note)
    return "; ".join(unique)


def build_rows(
    split_root: Path,
    comparison_rows: dict[tuple[str, str], dict[str, str]],
    split_name: str,
    scene_filter: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_dir = split_root / split_name
    for scene_dir in sorted(path for path in split_dir.glob("scene_*") if path.is_dir()):
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        meta = read_json(scene_dir / "meta.json")
        audit = meta.get("audit_summary", {}) or {}
        stability = audit.get("stability_check", {}) or {}
        comparison = comparison_rows.get((split_name, scene_dir.name), {})
        problems = merged_problem_tags(meta)
        row = {
            "split": split_name,
            "scene_id": scene_dir.name,
            "category": meta.get("category", ""),
            "difficulty": meta.get("difficulty", ""),
            "issues": ", ".join(problems),
            "problem_flags": ", ".join(problem_flags(meta)),
            "opencv_meta_stitcher_status": audit.get("stitcher_status", ""),
            "opencv_meta_stability_label": stability.get("stability_label", ""),
            "opencv_meta_ok_rate": stability.get("ok_rate", ""),
            "opencv_meta_output_consistent": stability.get("is_output_consistent", ""),
            "comparison_status": comparison.get("comparison_status", ""),
            "opencv_status_name": comparison.get("opencv_status_name", ""),
            "manual_status": comparison.get("manual_status", ""),
            "manual_image_coverage": manual_coverage(comparison),
            "manual_successful_pairs": comparison.get("manual_successful_pairs", ""),
            "manual_failed_pairs": comparison.get("manual_failed_pairs", ""),
            "manual_mean_inlier_ratio": comparison.get("manual_mean_inlier_ratio", ""),
            "manual_mean_reprojection_error": comparison.get("manual_mean_reprojection_error", ""),
            "manual_panorama_shape": shape_text(comparison, "manual"),
            "opencv_panorama_shape": shape_text(comparison, "opencv"),
            "observed_stitching_result": observed_result(comparison),
            "likely_glitch_or_error": likely_glitch(problems, comparison),
            "notes": meta.get("notes", ""),
        }
        for field in PROBLEM_FLAG_FIELDS:
            row[field] = bool(meta.get(field))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "scene_id",
        "category",
        "difficulty",
        "issues",
        "problem_flags",
        *PROBLEM_FLAG_FIELDS,
        "opencv_meta_stitcher_status",
        "opencv_meta_stability_label",
        "opencv_meta_ok_rate",
        "opencv_meta_output_consistent",
        "comparison_status",
        "opencv_status_name",
        "manual_status",
        "manual_image_coverage",
        "manual_successful_pairs",
        "manual_failed_pairs",
        "manual_mean_inlier_ratio",
        "manual_mean_reprojection_error",
        "manual_panorama_shape",
        "opencv_panorama_shape",
        "observed_stitching_result",
        "likely_glitch_or_error",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_problem_counts(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for problem in [value.strip() for value in row["issues"].split(",") if value.strip()]:
            counts[problem]["scene_count"] += 1
            counts[problem][row.get("comparison_status") or "missing_comparison"] += 1
            opencv_status = row.get("opencv_status_name") or row.get("opencv_meta_stitcher_status") or "missing"
            counts[problem][f"opencv_{opencv_status}"] += 1

    comparison_statuses = sorted(
        {
            key
            for counter in counts.values()
            for key in counter
            if key not in {"scene_count"} and not key.startswith("opencv_")
        }
    )
    opencv_statuses = sorted(
        {
            key
            for counter in counts.values()
            for key in counter
            if key.startswith("opencv_")
        }
    )
    fieldnames = ["problem", "scene_count", *comparison_statuses, *opencv_statuses]
    count_rows: list[dict[str, Any]] = []
    for problem in sorted(counts):
        counter = counts[problem]
        count_rows.append(
            {
                "problem": problem,
                **{field: counter.get(field, 0) for field in fieldnames if field != "problem"},
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(count_rows)
    return count_rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Scene | Category | Problems | OpenCV | Manual | Observed result | Likely artifact/error |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {scene_id} | {category} | {issues} | {opencv_status_name} | {manual_image_coverage} ({manual_status}) | {observed_stitching_result} | {likely_glitch_or_error} |".format(
                **{
                    key: str(value).replace("|", "/")
                    for key, value in row.items()
                }
            )
        )
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict[str, Any]], count_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    problem_counts = Counter()
    for row in rows:
        problem_counts.update(value.strip() for value in row["issues"].split(",") if value.strip())

    lines = [
        "# Failure Problem Stitching Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Scene problems are read from `{args.split_root / args.split / 'scene_xx' / 'meta.json'}` fields: `issues` and `has_*` flags.",
        f"Stitching results are joined from `{args.comparison_csv}`.",
        "",
        "These problem labels are human/project metadata, not labels returned by OpenCV. OpenCV contributes the stitcher status, stability check, and panorama output.",
        "",
        "## Problem Coverage",
        "",
        "| Problem | Scenes | Meaning |",
        "|---|---:|---|",
    ]
    for problem, count in sorted(problem_counts.items()):
        lines.append(f"| {problem} | {count} | {PROBLEM_GLOSSARY.get(problem, '')} |")

    lines.extend(["", "## Scene Outcomes", "", markdown_table(rows), ""])

    if count_rows:
        lines.extend(
            [
                "## Counts by Problem",
                "",
                "The count table is also saved as CSV. One scene can contribute to multiple problem tags.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_plot(path: Path, count_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        save_plot_with_pillow(path, count_rows)
        return

    if not count_rows:
        return

    problems = [row["problem"] for row in count_rows]
    statuses = ["both_ok", "manual_only", "manual_partial_only", "missing_comparison"]
    colors = {
        "both_ok": "#2ca02c",
        "manual_only": "#ff7f0e",
        "manual_partial_only": "#d62728",
        "missing_comparison": "#7f7f7f",
    }

    fig_height = max(4.5, 0.42 * len(problems) + 1.4)
    figure, axis = plt.subplots(figsize=(11, fig_height))
    left = [0] * len(problems)
    y_positions = list(range(len(problems)))

    for status in statuses:
        values = [int(row.get(status, 0) or 0) for row in count_rows]
        if not any(values):
            continue
        axis.barh(y_positions, values, left=left, label=status, color=colors[status])
        left = [current + value for current, value in zip(left, values)]

    axis.set_yticks(y_positions)
    axis.set_yticklabels(problems)
    axis.invert_yaxis()
    axis.set_xlabel("Scene count (overlapping problem tags)")
    axis.set_title("Failure-analysis problem tags vs stitching outcome")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_plot_with_pillow(path: Path, count_rows: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    if not count_rows:
        return

    statuses = ["both_ok", "manual_only", "manual_partial_only", "missing_comparison"]
    colors = {
        "both_ok": (44, 160, 44),
        "manual_only": (255, 127, 14),
        "manual_partial_only": (214, 39, 40),
        "missing_comparison": (127, 127, 127),
    }
    font = ImageFont.load_default()
    row_height = 34
    label_width = 260
    plot_width = 520
    legend_height = 58
    width = label_width + plot_width + 60
    height = 64 + row_height * len(count_rows) + legend_height
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((20, 18), "Failure-analysis problem tags vs stitching outcome", fill=(20, 20, 20), font=font)
    y = 54
    max_count = max(int(row.get("scene_count", 0) or 0) for row in count_rows) or 1
    for row in count_rows:
        problem = str(row["problem"])
        draw.text((20, y + 7), problem, fill=(20, 20, 20), font=font)
        x = label_width
        for status in statuses:
            value = int(row.get(status, 0) or 0)
            if value <= 0:
                continue
            bar_width = int((value / max_count) * plot_width)
            draw.rectangle([x, y + 6, x + bar_width, y + 24], fill=colors[status])
            draw.text((x + 4, y + 8), str(value), fill="white", font=font)
            x += bar_width
        draw.line([label_width, y + 29, label_width + plot_width, y + 29], fill=(225, 225, 225))
        y += row_height

    legend_y = y + 12
    legend_x = 20
    for status in statuses:
        draw.rectangle([legend_x, legend_y, legend_x + 14, legend_y + 14], fill=colors[status])
        draw.text((legend_x + 20, legend_y + 2), status, fill=(20, 20, 20), font=font)
        legend_x += 170

    image.save(path)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    scene_filter = set(args.scene) if args.scene else None

    comparison_rows = read_comparison_rows(args.comparison_csv)
    rows = build_rows(args.split_root, comparison_rows, args.split, scene_filter)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "failure_problem_stitching_summary.csv"
    json_path = args.output_dir / "failure_problem_stitching_summary.json"
    md_path = args.output_dir / "failure_problem_stitching_summary.md"
    count_csv_path = args.output_dir / "failure_problem_status_counts.csv"
    plot_path = args.output_dir / "failure_problem_status_counts.png"

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    count_rows = write_problem_counts(count_csv_path, rows)
    write_markdown(md_path, rows, count_rows, args)
    if not args.no_plots:
        save_plot(plot_path, count_rows)

    print(f"Scenes summarized: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Problem counts: {count_csv_path}")
    if plot_path.exists():
        print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

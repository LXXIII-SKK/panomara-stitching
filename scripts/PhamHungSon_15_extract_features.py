from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_PREPROCESSING_ROOT = DATA_ROOT / "preprocessing"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "feature_extract"
DEFAULT_SHOWCASE_SPLITS = ["test", "failure_analysis"]

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
KEYPOINT_COLUMNS = np.array(
    ["x", "y", "size", "angle", "response", "octave", "class_id"],
    dtype=object,
)

SPLIT_ALIASES = {
    "dev": "development",
    "development": "development",
    "test": "test",
    "failure": "failure_analysis",
    "failure_analysis": "failure_analysis",
}

DESCRIPTOR_ALIASES = {
    "SIFT": "SIFT",
    "ORB": "ORB",
    "AKAZE": "AKAZE",
    "HARRIS_HOG": "HARRIS_HOG",
    "HARRIS+HOG": "HARRIS_HOG",
    "HARRIS_HOG": "HARRIS_HOG",
    "ALL": "ALL",
}

DESCRIPTOR_DISPLAY_NAMES = {
    "SIFT": "SIFT",
    "ORB": "ORB",
    "HARRIS_HOG": "Harris + HOG",
    "AKAZE": "AKAZE",
}

ALL_DESCRIPTORS = ["ORB", "AKAZE", "HARRIS_HOG", "SIFT"]
EXPLICIT_DESCRIPTORS = ALL_DESCRIPTORS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract local features/descriptors from preprocessed panorama images and save per-scene results."
        ),
        epilog=(
            "Examples:\n"
            "  conda run -n image_recognition python scripts/PhamHungSon_15_extract_features.py --split failure_analysis --scene scene_32 --descriptor SIFT\n"
            "  conda run -n image_recognition python scripts/PhamHungSon_15_extract_features.py --split test --scene scene_01 --descriptor SIFT\n"
            "  conda run -n image_recognition python scripts/PhamHungSon_15_extract_features.py --split test --descriptor ORB --descriptor AKAZE\n"
            "  conda run -n image_recognition python scripts/PhamHungSon_15_extract_features.py --descriptor all --overwrite\n\n"
            "Default input layout:\n"
            "  data/preprocessing/<split>/feature_gray/<scene>/*.png\n\n"
            "Default output layout:\n"
            "  data/feature_extract/<split>/<scene>/<descriptor>/\n\n"
            "Scene names like scene_3 are accepted as input aliases, but the output preserves the actual "
            "preprocessing folder name, for example scene_03."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preprocessing-root",
        type=Path,
        default=DEFAULT_PREPROCESSING_ROOT,
        help="Root containing preprocessed split folders. Defaults to data/preprocessing.",
    )
    parser.add_argument(
        "--input-kind",
        default="feature_gray",
        help="Preprocessing subfolder to read inside each split. Defaults to feature_gray.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root where feature extraction results are written. Defaults to data/feature_extract.",
    )
    parser.add_argument(
        "--split",
        action="append",
        help="Split to process. Repeat for multiple splits. If omitted, test and failure_analysis are used for showcase runs. Use --split all to include development too.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Scene to process, such as scene_03 or scene_3. Repeat for multiple scenes. If omitted, all scenes are used.",
    )
    parser.add_argument(
        "--descriptor",
        action="append",
        default=None,
        help=(
            "Descriptor/pipeline to run: ORB, AKAZE, Harris_HOG, SIFT, or all. "
            "'all' runs the four-method experiment set used in the report. "
            "Repeat for multiple descriptors."
        ),
    )
    parser.add_argument("--max-features", type=int, default=4000, help="Maximum features for methods that support limiting.")
    parser.add_argument("--orb-fast-threshold", type=int, default=10, help="FAST threshold used by ORB.")
    parser.add_argument("--harris-max-corners", type=int, default=1500, help="Maximum Harris corners for Harris pipelines.")
    parser.add_argument("--harris-quality", type=float, default=0.01, help="Harris qualityLevel parameter.")
    parser.add_argument("--harris-min-distance", type=float, default=8.0, help="Harris minimum corner distance.")
    parser.add_argument("--hog-patch-size", type=int, default=32, help="Patch size used by the simple HOG descriptor.")
    parser.add_argument("--hog-cells", type=int, default=4, help="Number of HOG cells per side.")
    parser.add_argument("--hog-bins", type=int, default=8, help="Number of HOG orientation bins.")
    parser.add_argument("--draw-keypoints-limit", type=int, default=1000, help="Maximum keypoints drawn in visualizations.")
    parser.add_argument("--no-visualizations", action="store_true", help="Do not save keypoint visualization images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files and metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved work items without writing files.")
    parser.add_argument("--limit-images", type=int, default=None, help="Optional maximum number of images per scene.")
    return parser


def normalize_descriptor(raw_value: str) -> str:
    token = re.sub(r"[\s\-]+", "_", raw_value.strip().upper())
    token = token.replace("_+_", "+").replace("+", "+")
    token = token.replace("HARRIS_HOG", "HARRIS_HOG")
    normalized = DESCRIPTOR_ALIASES.get(token)
    if normalized is None:
        raise ValueError(
            f"Unknown descriptor '{raw_value}'. Choose one of: {', '.join(EXPLICIT_DESCRIPTORS)} or all."
        )
    return normalized


def normalize_descriptors(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return ["SIFT"]

    descriptors: list[str] = []
    for raw_value in raw_values:
        canonical = normalize_descriptor(raw_value)
        if canonical == "ALL":
            descriptors.extend(ALL_DESCRIPTORS)
        else:
            descriptors.append(canonical)

    unique: list[str] = []
    for descriptor in descriptors:
        if descriptor not in unique:
            unique.append(descriptor)
    return unique


def normalize_split(raw_value: str) -> str:
    token = raw_value.strip().lower()
    return SPLIT_ALIASES.get(token, token)


def scene_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def image_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def list_image_files(scene_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in scene_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS
        ],
        key=image_sort_key,
    )


def available_splits(preprocessing_root: Path, input_kind: str) -> list[str]:
    if not preprocessing_root.exists():
        return []
    split_names = []
    for split_dir in sorted(preprocessing_root.iterdir(), key=lambda path: path.name):
        if split_dir.is_dir() and (split_dir / input_kind).is_dir():
            split_names.append(split_dir.name)
    return split_names


def resolve_splits(preprocessing_root: Path, input_kind: str, requested_splits: list[str] | None) -> list[str]:
    if not requested_splits:
        split_names = available_splits(preprocessing_root, input_kind)
        if not split_names:
            raise FileNotFoundError(
                f"No preprocessing splits found under {preprocessing_root} with input kind '{input_kind}'."
            )
        showcase = [split_name for split_name in DEFAULT_SHOWCASE_SPLITS if split_name in split_names]
        return showcase or split_names

    split_names = []
    for raw_split in requested_splits:
        if raw_split.strip().lower() == "all":
            split_names.extend(available_splits(preprocessing_root, input_kind))
        else:
            split_names.append(normalize_split(raw_split))

    unique = []
    for split_name in split_names:
        if split_name not in unique:
            split_input_dir = preprocessing_root / split_name / input_kind
            if not split_input_dir.is_dir():
                raise FileNotFoundError(
                    f"Preprocessed input folder not found: {split_input_dir}. "
                    "Run preprocessing for this split first."
                )
            unique.append(split_name)
    return unique


def scene_aliases(scene_name: str) -> list[str]:
    cleaned = scene_name.strip()
    aliases = [cleaned]
    match = re.fullmatch(r"scene_(\d+)", cleaned, flags=re.IGNORECASE)
    if match:
        number = int(match.group(1))
        aliases.append(f"scene_{number:02d}")
        aliases.append(f"scene_{number}")
    return list(dict.fromkeys(aliases))


def resolve_scene_dirs(split_input_dir: Path, requested_scenes: list[str] | None) -> list[Path]:
    if not requested_scenes:
        return sorted([path for path in split_input_dir.iterdir() if path.is_dir()], key=scene_sort_key)

    scene_dirs = []
    for raw_scene in requested_scenes:
        candidates = [split_input_dir / alias for alias in scene_aliases(raw_scene)]
        for candidate in candidates:
            if candidate.is_dir():
                scene_dirs.append(candidate)
                break
        else:
            tried = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(f"Scene '{raw_scene}' was not found. Tried: {tried}")
    return list(dict.fromkeys(scene_dirs))


def harris_keypoints(
    gray: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    keypoint_size: float = 31.0,
) -> list[cv2.KeyPoint]:
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=3,
        useHarrisDetector=True,
        k=0.04,
    )
    if corners is None:
        return []
    keypoints = []
    for x, y in corners.reshape(-1, 2):
        keypoints.append(cv2.KeyPoint(float(x), float(y), keypoint_size))
    return keypoints


def compute_hog_descriptors(
    gray: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    patch_size: int,
    cells_per_side: int,
    bins: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    if not keypoints:
        return [], None

    patch_size = max(8, int(patch_size))
    if patch_size % cells_per_side != 0:
        raise ValueError("--hog-patch-size must be divisible by --hog-cells.")

    half = patch_size // 2
    pad = half + 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT101)
    gx = cv2.Sobel(padded, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(padded, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)

    cell_size = patch_size // cells_per_side
    bin_width = (2.0 * np.pi) / bins
    kept_keypoints: list[cv2.KeyPoint] = []
    descriptors: list[np.ndarray] = []

    for keypoint in keypoints:
        center_x = int(round(keypoint.pt[0])) + pad
        center_y = int(round(keypoint.pt[1])) + pad
        y0, y1 = center_y - half, center_y + half
        x0, x1 = center_x - half, center_x + half
        patch_mag = magnitude[y0:y1, x0:x1]
        patch_ang = angle[y0:y1, x0:x1]
        if patch_mag.shape != (patch_size, patch_size):
            continue

        feature_parts = []
        for cy in range(cells_per_side):
            for cx in range(cells_per_side):
                cell_mag = patch_mag[
                    cy * cell_size : (cy + 1) * cell_size,
                    cx * cell_size : (cx + 1) * cell_size,
                ]
                cell_ang = patch_ang[
                    cy * cell_size : (cy + 1) * cell_size,
                    cx * cell_size : (cx + 1) * cell_size,
                ]
                cell_bins = np.floor(cell_ang / bin_width).astype(np.int32) % bins
                hist = np.zeros(bins, dtype=np.float32)
                for bin_index in range(bins):
                    hist[bin_index] = float(cell_mag[cell_bins == bin_index].sum())
                feature_parts.append(hist)

        descriptor = np.concatenate(feature_parts).astype(np.float32)
        descriptor /= np.linalg.norm(descriptor) + 1e-7
        descriptor = np.clip(descriptor, 0.0, 0.2)
        descriptor /= np.linalg.norm(descriptor) + 1e-7
        descriptors.append(descriptor)
        kept_keypoints.append(keypoint)

    if not descriptors:
        return [], None
    return kept_keypoints, np.vstack(descriptors).astype(np.float32)


def sort_and_limit_keypoints(
    keypoints: list[cv2.KeyPoint],
    descriptors: np.ndarray | None,
    max_features: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    if max_features <= 0 or len(keypoints) <= max_features:
        return keypoints, descriptors

    order = sorted(range(len(keypoints)), key=lambda index: keypoints[index].response, reverse=True)[:max_features]
    limited_keypoints = [keypoints[index] for index in order]
    if descriptors is None:
        return limited_keypoints, None
    return limited_keypoints, descriptors[np.array(order)]


def extract_features(gray: np.ndarray, descriptor: str, args: argparse.Namespace) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    max_features = max(0, int(args.max_features))

    if descriptor == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is unavailable in this OpenCV build.")
        detector = cv2.SIFT_create(nfeatures=max_features)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return list(keypoints or []), descriptors

    if descriptor == "ORB":
        detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=args.orb_fast_threshold)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return list(keypoints or []), descriptors

    if descriptor == "AKAZE":
        detector = cv2.AKAZE_create()
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return sort_and_limit_keypoints(list(keypoints or []), descriptors, max_features)

    if descriptor == "HARRIS_HOG":
        corner_limit = min(args.harris_max_corners, max_features) if max_features > 0 else args.harris_max_corners
        keypoints = harris_keypoints(gray, corner_limit, args.harris_quality, args.harris_min_distance)
        return compute_hog_descriptors(
            gray,
            keypoints,
            patch_size=args.hog_patch_size,
            cells_per_side=args.hog_cells,
            bins=args.hog_bins,
        )

    raise ValueError(f"Unknown descriptor: {descriptor}")


def keypoints_to_array(keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    rows = np.zeros((len(keypoints), len(KEYPOINT_COLUMNS)), dtype=np.float32)
    for index, keypoint in enumerate(keypoints):
        rows[index] = [
            float(keypoint.pt[0]),
            float(keypoint.pt[1]),
            float(keypoint.size),
            float(keypoint.angle),
            float(keypoint.response),
            float(keypoint.octave),
            float(keypoint.class_id),
        ]
    return rows


def descriptor_to_array(descriptors: np.ndarray | None) -> np.ndarray:
    if descriptors is None:
        return np.empty((0, 0), dtype=np.float32)
    if descriptors.ndim == 1:
        return descriptors.reshape(-1, 1)
    return descriptors


def relative_to_project(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_keypoint_visualization(
    gray: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    output_path: Path,
    draw_limit: int,
) -> None:
    draw_keypoints = keypoints
    if draw_limit > 0 and len(keypoints) > draw_limit:
        draw_keypoints = sorted(keypoints, key=lambda keypoint: keypoint.response, reverse=True)[:draw_limit]

    visualization = cv2.drawKeypoints(
        gray,
        draw_keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(str(output_path), visualization)


def output_paths(
    output_root: Path,
    split_name: str,
    scene_name: str,
    descriptor: str,
    image_path: Path,
) -> dict[str, Path]:
    descriptor_dir = output_root / split_name / scene_name / descriptor
    return {
        "descriptor_dir": descriptor_dir,
        "features_dir": descriptor_dir / "features",
        "metadata_dir": descriptor_dir / "metadata",
        "visualizations_dir": descriptor_dir / "visualizations",
        "feature_file": descriptor_dir / "features" / f"{image_path.stem}.npz",
        "metadata_file": descriptor_dir / "metadata" / f"{image_path.stem}.json",
        "visualization_file": descriptor_dir / "visualizations" / f"{image_path.stem}_keypoints.jpg",
    }


def process_image(
    image_path: Path,
    split_name: str,
    scene_name: str,
    descriptor: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    paths = output_paths(args.output_root, split_name, scene_name, descriptor, image_path)
    for key in ["features_dir", "metadata_dir", "visualizations_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    if (
        not args.overwrite
        and paths["feature_file"].exists()
        and paths["metadata_file"].exists()
    ):
        existing = json.loads(paths["metadata_file"].read_text(encoding="utf-8"))
        existing["status"] = "existing"
        return existing

    start_time = time.perf_counter()
    metadata: dict[str, Any] = {
        "split": split_name,
        "scene": scene_name,
        "descriptor": descriptor,
        "descriptor_display_name": DESCRIPTOR_DISPLAY_NAMES[descriptor],
        "source_image": relative_to_project(image_path),
        "image_name": image_path.name,
        "status": "ok",
        "error": "",
        "outputs": {
            "feature_file": relative_to_project(paths["feature_file"]),
            "metadata_file": relative_to_project(paths["metadata_file"]),
            "visualization_file": "" if args.no_visualizations else relative_to_project(paths["visualization_file"]),
        },
        "parameters": {
            "max_features": args.max_features,
            "orb_fast_threshold": args.orb_fast_threshold,
            "harris_max_corners": args.harris_max_corners,
            "harris_quality": args.harris_quality,
            "harris_min_distance": args.harris_min_distance,
            "hog_patch_size": args.hog_patch_size,
            "hog_cells": args.hog_cells,
            "hog_bins": args.hog_bins,
        },
    }

    try:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        keypoints, descriptors = extract_features(gray, descriptor, args)
        descriptor_array = descriptor_to_array(descriptors)
        keypoint_array = keypoints_to_array(keypoints)

        np.savez_compressed(
            paths["feature_file"],
            keypoints=keypoint_array,
            descriptors=descriptor_array,
            keypoint_columns=KEYPOINT_COLUMNS,
            image_shape=np.array(gray.shape, dtype=np.int32),
            descriptor=np.array(descriptor),
            source_image=np.array(relative_to_project(image_path)),
        )

        if not args.no_visualizations:
            save_keypoint_visualization(gray, keypoints, paths["visualization_file"], args.draw_keypoints_limit)

        metadata.update(
            {
                "image_shape": {"height": int(gray.shape[0]), "width": int(gray.shape[1])},
                "keypoint_count": int(len(keypoints)),
                "descriptor_shape": [int(value) for value in descriptor_array.shape],
                "descriptor_dtype": str(descriptor_array.dtype),
                "runtime_sec": float(time.perf_counter() - start_time),
            }
        )

    except Exception as exc:
        status = "unavailable" if "unavailable" in str(exc).lower() else "failed"
        if status == "unavailable":
            metadata["outputs"]["feature_file"] = ""
            metadata["outputs"]["visualization_file"] = ""
        metadata.update(
            {
                "status": status,
                "error": str(exc),
                "image_shape": {},
                "keypoint_count": 0,
                "descriptor_shape": [0, 0],
                "descriptor_dtype": "",
                "runtime_sec": float(time.perf_counter() - start_time),
            }
        )

    write_json(paths["metadata_file"], metadata)
    return metadata


def metadata_to_row(metadata: dict[str, Any]) -> dict[str, Any]:
    outputs = metadata.get("outputs", {})
    image_shape = metadata.get("image_shape", {})
    descriptor_shape = metadata.get("descriptor_shape", [0, 0])
    descriptor_rows = descriptor_shape[0] if len(descriptor_shape) > 0 else 0
    descriptor_cols = descriptor_shape[1] if len(descriptor_shape) > 1 else 0

    return {
        "split": metadata.get("split", ""),
        "scene": metadata.get("scene", ""),
        "descriptor": metadata.get("descriptor", ""),
        "image": metadata.get("image_name", ""),
        "source_image": metadata.get("source_image", ""),
        "status": metadata.get("status", ""),
        "keypoints": metadata.get("keypoint_count", 0),
        "descriptor_rows": descriptor_rows,
        "descriptor_cols": descriptor_cols,
        "descriptor_dtype": metadata.get("descriptor_dtype", ""),
        "image_height": image_shape.get("height", ""),
        "image_width": image_shape.get("width", ""),
        "runtime_sec": f"{metadata.get('runtime_sec', 0.0):.6f}",
        "feature_file": outputs.get("feature_file", ""),
        "visualization_file": outputs.get("visualization_file", ""),
        "metadata_file": outputs.get("metadata_file", ""),
        "error": metadata.get("error", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "scene",
        "descriptor",
        "image",
        "source_image",
        "status",
        "keypoints",
        "descriptor_rows",
        "descriptor_cols",
        "descriptor_dtype",
        "image_height",
        "image_width",
        "runtime_sec",
        "feature_file",
        "visualization_file",
        "metadata_file",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_descriptor_config(descriptor_dir: Path, descriptor: str, args: argparse.Namespace) -> None:
    payload = {
        "descriptor": descriptor,
        "descriptor_display_name": DESCRIPTOR_DISPLAY_NAMES[descriptor],
        "input_kind": args.input_kind,
        "opencv_version": cv2.__version__,
        "parameters": {
            "max_features": args.max_features,
            "orb_fast_threshold": args.orb_fast_threshold,
            "harris_max_corners": args.harris_max_corners,
            "harris_quality": args.harris_quality,
            "harris_min_distance": args.harris_min_distance,
            "hog_patch_size": args.hog_patch_size,
            "hog_cells": args.hog_cells,
            "hog_bins": args.hog_bins,
            "draw_keypoints_limit": args.draw_keypoints_limit,
        },
    }
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    write_json(descriptor_dir / "descriptor_config.json", payload)


def process_scene(
    split_name: str,
    scene_dir: Path,
    descriptor: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    image_files = list_image_files(scene_dir)
    if args.limit_images is not None:
        image_files = image_files[: args.limit_images]

    descriptor_dir = args.output_root / split_name / scene_dir.name / descriptor
    save_descriptor_config(descriptor_dir, descriptor, args)

    scene_metadata = []
    for image_path in image_files:
        metadata = process_image(image_path, split_name, scene_dir.name, descriptor, args)
        scene_metadata.append(metadata)

    rows = [metadata_to_row(metadata) for metadata in scene_metadata]
    write_csv(descriptor_dir / "summary.csv", rows)
    write_json(
        descriptor_dir / "summary.json",
        {
            "split": split_name,
            "scene": scene_dir.name,
            "descriptor": descriptor,
            "input_kind": args.input_kind,
            "image_count": len(image_files),
            "rows": scene_metadata,
        },
    )
    return scene_metadata


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.preprocessing_root = args.preprocessing_root.resolve()
    args.output_root = args.output_root.resolve()

    descriptors = normalize_descriptors(args.descriptor)
    split_names = resolve_splits(args.preprocessing_root, args.input_kind, args.split)

    work_items: list[tuple[str, Path, str]] = []
    for split_name in split_names:
        split_input_dir = args.preprocessing_root / split_name / args.input_kind
        scene_dirs = resolve_scene_dirs(split_input_dir, args.scenes)
        for scene_dir in scene_dirs:
            for descriptor in descriptors:
                work_items.append((split_name, scene_dir, descriptor))

    print(f"Preprocessing root: {args.preprocessing_root}")
    print(f"Input kind: {args.input_kind}")
    print(f"Output root: {args.output_root}")
    print(f"Descriptors: {', '.join(descriptors)}")
    print(f"Work items: {len(work_items)} scene/descriptor combinations")

    if args.dry_run:
        for split_name, scene_dir, descriptor in work_items:
            image_count = len(list_image_files(scene_dir))
            if args.limit_images is not None:
                image_count = min(image_count, args.limit_images)
            output_dir = args.output_root / split_name / scene_dir.name / descriptor
            print(f"- {split_name}/{scene_dir.name}/{descriptor}: {image_count} images -> {output_dir}")
        return 0

    all_metadata: list[dict[str, Any]] = []
    for index, (split_name, scene_dir, descriptor) in enumerate(work_items, start=1):
        print(f"[{index}/{len(work_items)}] {split_name}/{scene_dir.name}/{descriptor}")
        scene_metadata = process_scene(split_name, scene_dir, descriptor, args)
        all_metadata.extend(scene_metadata)

    manifest_rows = [metadata_to_row(metadata) for metadata in all_metadata]
    write_csv(args.output_root / "feature_extract_manifest.csv", manifest_rows)
    write_json(
        args.output_root / "feature_extract_manifest.json",
        {
            "preprocessing_root": relative_to_project(args.preprocessing_root),
            "input_kind": args.input_kind,
            "output_root": relative_to_project(args.output_root),
            "descriptors": descriptors,
            "opencv_version": cv2.__version__,
            "item_count": len(all_metadata),
            "rows": all_metadata,
        },
    )

    ok_count = sum(1 for metadata in all_metadata if metadata.get("status") in {"ok", "existing"})
    unavailable_count = sum(1 for metadata in all_metadata if metadata.get("status") == "unavailable")
    failed_count = sum(1 for metadata in all_metadata if metadata.get("status") == "failed")
    print(f"Saved feature extraction results for {ok_count} images.")
    if unavailable_count:
        print(f"Unavailable descriptor/image entries recorded: {unavailable_count}.")
    if failed_count:
        print(f"Failed images: {failed_count}. Check summary.csv files for details.")
    print(f"Manifest: {args.output_root / 'feature_extract_manifest.csv'}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

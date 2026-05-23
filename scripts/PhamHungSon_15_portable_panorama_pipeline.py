from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STATUS_SCORE = {"failure": 0, "hard_valid": 1, "success": 2}
PRESET_NAMES = {"custom", "weak_phone", "normal_phone", "best_quality", "student_debug"}
PROBLEM_FLAG_FIELDS = [
    "has_moving_objects",
    "has_repeated_patterns",
    "has_low_texture",
    "has_parallax",
    "has_exposure_change",
    "has_motion_blur",
    "has_insufficient_overlap",
]

PRESET_DESCRIPTIONS = {
    "custom": "Use the explicit technical options as configured.",
    "weak_phone": "Small memory/CPU budget: ORB, low width, low feature count, overwrite blending, partial output allowed.",
    "normal_phone": "Balanced default for normal users: OpenCV Stitcher, moderate width, CLAHE preprocessing.",
    "best_quality": "Slow high-quality mode: OpenCV Stitcher by default, larger input width, SIFT-capable configuration.",
    "student_debug": "Learning/report mode: manual stitcher with descriptor tables, keypoints, matches, inliers, and pair warp previews.",
}


class PanoramaPipelineError(RuntimeError):
    """Error with a structured payload for apps and CLI callers."""

    def __init__(self, message: str, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.payload = payload or {}


@dataclass
class PanoramaConfig:
    """Config for a portable panorama pipeline.

    The class deliberately uses plain values so it can be exposed as Android UI
    settings or serialized to JSON without custom converters.
    """

    preset: str = "custom"  # custom, weak_phone, normal_phone, best_quality, student_debug
    preset_applied: bool = False
    engine: str = "manual"  # manual or opencv
    profile: str = "balanced"  # fast, balanced, quality
    method: str = "auto"  # auto, ORB, AKAZE, SIFT
    candidate_methods: list[str] = field(default_factory=lambda: ["ORB", "AKAZE"])
    work_width: int = 1280
    max_features: int = 3000
    ratio_test: float = 0.75
    ransac_threshold: float = 4.0
    min_good_matches: int = 12
    min_inliers: int = 16
    min_inlier_ratio: float = 0.18
    blend_mode: str = "average"  # average, feather, overwrite
    anchor: str = "middle"  # middle, first, last, or integer index
    manual_motion_model: str = "affine"  # translation, similarity, affine, or homography
    crop: bool = True
    allow_partial: bool = False
    max_canvas_megapixels: float = 24.0
    max_canvas_side: int = 12000
    preprocess: str = "clahe"  # none, gray, clahe
    enable_gamma: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    gaussian_kernel: int = 3
    orb_fast_threshold: int = 10
    stitcher_mode: str = "PANORAMA"  # PANORAMA or SCANS for engine=opencv
    image_order: str = "meta"  # meta, name, mtime
    reverse_order: bool = False
    max_images: int = 0
    skip_every: int = 1
    feature_cache_root: str = ""
    split_name: str = ""
    scene_id: str = ""
    prefer_cache: bool = False
    pair_method_map: dict[str, str] = field(default_factory=dict)
    save_debug: bool = False
    save_pair_visualizations: bool = False
    save_score_table: bool = False
    visualization_dir: str = ""
    diagnostics_methods: list[str] = field(default_factory=list)
    visualization_max_matches: int = 80
    visualization_max_keypoints: int = 1000
    visualization_max_width: int = 1800
    visualization_jpeg_quality: int = 85
    harris_max_corners: int = 1500
    harris_quality: float = 0.01
    harris_min_distance: float = 8.0
    hog_patch_size: int = 32
    hog_cells: int = 4
    hog_bins: int = 8


@dataclass
class FeatureData:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None
    image_shape: tuple[int, int]
    source: str


@dataclass
class PairEstimate:
    pair_id: str
    pair_index: int
    image_a: str
    image_b: str
    method: str
    motion_model: str
    status: str
    homography: np.ndarray | None
    raw_matches: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    reprojection_error_mean: float | None
    error: str = ""
    feature_source_a: str = ""
    feature_source_b: str = ""
    keypoints_a: int = 0
    keypoints_b: int = 0
    lowe_pass_rate: float = 0.0
    median_lowe_ratio: float | None = None
    reprojection_error_median: float | None = None
    inlier_lowe_ratio: float | None = None
    spatial_coverage: float | None = None
    overlap_similarity: float | None = None
    overlap_pixels: int = 0
    homography_sanity: float | None = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["homography"] = None if self.homography is None else self.homography.tolist()
        return payload


def disable_diagnostics_for_average_preset(config: PanoramaConfig) -> None:
    config.save_debug = False
    config.save_pair_visualizations = False
    config.save_score_table = False
    config.diagnostics_methods = []


def apply_user_preset(config: PanoramaConfig) -> PanoramaConfig:
    if config.preset_applied:
        return config
    preset = (config.preset or "custom").strip().lower()
    if preset not in PRESET_NAMES:
        raise ValueError(f"Unknown preset: {config.preset}. Use one of: {', '.join(sorted(PRESET_NAMES))}.")
    config.preset = preset

    if preset == "weak_phone":
        config.engine = "manual"
        config.profile = "fast"
        config.method = "ORB"
        config.candidate_methods = ["ORB"]
        config.work_width = 800
        config.max_features = 1200
        config.ratio_test = 0.80
        config.min_good_matches = 10
        config.min_inliers = 12
        config.blend_mode = "overwrite"
        config.manual_motion_model = "similarity"
        config.preprocess = "gray"
        config.allow_partial = True
        config.max_canvas_megapixels = 8.0
        config.max_canvas_side = 6000
        config.visualization_max_width = 900
        disable_diagnostics_for_average_preset(config)
    elif preset == "normal_phone":
        config.engine = "opencv"
        config.profile = "balanced"
        config.method = "auto"
        config.candidate_methods = ["ORB", "AKAZE"]
        config.work_width = 1280
        config.max_features = 3000
        config.blend_mode = "average"
        config.manual_motion_model = "affine"
        config.preprocess = "clahe"
        config.max_canvas_megapixels = 12.0
        config.max_canvas_side = 12000
        disable_diagnostics_for_average_preset(config)
    elif preset == "best_quality":
        config.engine = "opencv"
        config.profile = "quality"
        config.method = "auto"
        config.candidate_methods = ["SIFT", "AKAZE", "ORB"]
        config.work_width = 1800
        config.max_features = 6000
        config.ratio_test = 0.72
        config.blend_mode = "feather"
        config.manual_motion_model = "affine"
        config.preprocess = "clahe"
        config.enable_gamma = True
        config.max_canvas_megapixels = 16.0
        config.max_canvas_side = 16000
        disable_diagnostics_for_average_preset(config)
    elif preset == "student_debug":
        config.engine = "manual"
        config.profile = "balanced"
        config.method = "auto"
        config.candidate_methods = ["ORB", "AKAZE", "SIFT", "HARRIS_HOG"]
        config.work_width = 1280
        config.max_features = 3000
        config.blend_mode = "average"
        config.manual_motion_model = "affine"
        config.preprocess = "clahe"
        config.allow_partial = True
        config.save_debug = True
        config.save_pair_visualizations = True
        config.save_score_table = True
        config.diagnostics_methods = ["all"]
        config.visualization_max_width = 1000
        config.visualization_jpeg_quality = 72
    config.preset_applied = True
    return config


def apply_profile(config: PanoramaConfig) -> PanoramaConfig:
    profile = config.profile.strip().lower()
    if profile == "fast":
        config.method = "ORB" if config.method == "auto" else config.method
        config.candidate_methods = ["ORB"]
        config.work_width = min(config.work_width, 960) if config.work_width > 0 else 960
        config.max_features = min(config.max_features, 1500)
        config.ratio_test = max(config.ratio_test, 0.78)
        config.blend_mode = "overwrite" if config.blend_mode == "average" else config.blend_mode
        config.preprocess = "gray" if config.preprocess == "clahe" else config.preprocess
        config.max_canvas_megapixels = min(config.max_canvas_megapixels, 12.0)
    elif profile == "quality":
        config.method = "auto" if config.method == "auto" else config.method
        if not config.candidate_methods or config.candidate_methods == ["ORB", "AKAZE"]:
            config.candidate_methods = ["SIFT", "AKAZE", "ORB"]
        config.work_width = max(config.work_width, 1600)
        config.max_features = max(config.max_features, 5000)
        config.ratio_test = min(config.ratio_test, 0.74)
        config.blend_mode = "feather" if config.blend_mode == "average" else config.blend_mode
        config.max_canvas_megapixels = max(config.max_canvas_megapixels, 16.0 if hasattr(sys, "getandroidapilevel") else 32.0)
    else:
        config.profile = "balanced"
        if not config.candidate_methods:
            config.candidate_methods = ["ORB", "AKAZE"]
    return config


def finalize_config(config: PanoramaConfig) -> PanoramaConfig:
    return apply_profile(apply_user_preset(config))


def normalize_method(value: str) -> str:
    token = value.strip().upper().replace("-", "_").replace("+", "_")
    aliases = {
        "AUTO": "auto",
        "ORB": "ORB",
        "AKAZE": "AKAZE",
        "SIFT": "SIFT",
        "HARRIS_HOG": "HARRIS_HOG",
        "HARRIS HOG": "HARRIS_HOG",
    }
    if token not in aliases:
        raise ValueError(f"Unknown method: {value}. Use auto, ORB, AKAZE, SIFT, or HARRIS_HOG.")
    return aliases[token]


def normalize_motion_model(value: str) -> str:
    token = str(value or "affine").strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "shift": "translation",
        "translate": "translation",
        "translation": "translation",
        "rigid": "similarity",
        "partial_affine": "similarity",
        "similarity": "similarity",
        "affine": "affine",
        "homography": "homography",
        "projective": "homography",
    }
    if token not in aliases:
        raise ValueError("Unknown manual_motion_model: " + str(value))
    return aliases[token]


def image_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    number = int(match.group(1)) if match else 10**9
    return number, path.name.lower()


def list_scene_images(scene_dir: Path, image_order: str = "meta") -> list[Path]:
    images = [path for path in scene_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS]
    if image_order == "mtime":
        return sorted(images, key=lambda path: (path.stat().st_mtime, path.name.lower()))

    if image_order == "meta":
        meta_path = scene_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ordered_names = meta.get("ordered_files") or []
                lookup = {path.name: path for path in images}
                if ordered_names and all(name in lookup for name in ordered_names):
                    ordered = [lookup[name] for name in ordered_names]
                    return ordered
            except Exception:
                pass

    return sorted(images, key=image_sort_key)


def select_input_images(scene_dir: Path, config: PanoramaConfig) -> list[Path]:
    images = list_scene_images(scene_dir, config.image_order)
    if config.reverse_order:
        images = list(reversed(images))
    skip_every = max(1, int(config.skip_every))
    if skip_every > 1:
        images = images[::skip_every]
    if config.max_images and config.max_images > 0:
        images = images[: config.max_images]
    if len(images) < 2:
        raise ValueError(f"Need at least two images to stitch, found {len(images)} in {scene_dir}")
    return images


def read_scene_metadata(scene_dir: Path) -> dict[str, Any]:
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        return {
            "meta_path": "",
            "category": "",
            "difficulty": "",
            "issues": [],
            "problem_flags": [],
            "notes": "",
        }
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "meta_path": str(meta_path),
            "category": "",
            "difficulty": "",
            "issues": [],
            "problem_flags": [],
            "notes": "",
            "error": f"Could not read meta.json: {exc}",
        }

    issues = [str(issue) for issue in meta.get("issues", []) if str(issue).strip()]
    flags = [field.removeprefix("has_") for field in PROBLEM_FLAG_FIELDS if meta.get(field) is True]
    combined: list[str] = []
    for value in issues + flags:
        if value not in combined:
            combined.append(value)

    audit_summary = meta.get("audit_summary", {}) or {}
    stability = audit_summary.get("stability_check", {}) or {}
    return {
        "meta_path": str(meta_path),
        "scene_id": meta.get("scene_id", scene_dir.name),
        "category": meta.get("category", ""),
        "difficulty": meta.get("difficulty", ""),
        "recommended_use": meta.get("recommended_use", ""),
        "capture_group": meta.get("capture_group", ""),
        "issues": combined,
        "problem_flags": flags,
        "notes": meta.get("notes", ""),
        "audit_stitcher_status": audit_summary.get("stitcher_status", ""),
        "audit_ok_rate": stability.get("ok_rate", ""),
        "audit_stability_label": stability.get("stability_label", ""),
        "audit_output_consistent": stability.get("is_output_consistent", ""),
    }


def quality_warnings(scene_metadata: dict[str, Any], pairs: list[PairEstimate], is_partial: bool) -> list[str]:
    warnings: list[str] = []
    issues = set(scene_metadata.get("issues") or [])
    if is_partial:
        warnings.append("Only a contiguous partial panorama was produced; at least one adjacent image transition failed.")
    if any(pair.status != "success" for pair in pairs):
        weak_pairs = [pair.pair_id for pair in pairs if pair.status != "success"]
        warnings.append("Some adjacent pairs are weak or failed: " + ", ".join(weak_pairs))
    if {"parallax", "sideways_scan"} & issues:
        warnings.append("Parallax or sideways-scan motion can create ghosting, bent structures, or perspective drift.")
    if "insufficient_overlap" in issues:
        warnings.append("Insufficient overlap can cause missing regions or broken image chains.")
    if "low_texture" in issues:
        warnings.append("Low-texture regions reduce stable keypoints and can make the homography unstable.")
    if "exposure_change" in issues:
        warnings.append("Exposure changes can create visible brightness seams.")
    if {"moving_objects", "motion_blur"} & issues:
        warnings.append("Moving objects or blur can create ghosting, duplicated objects, or weak matches.")
    return warnings


def write_json_log(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_progress(output_path: Path | None, message: str, percent: int) -> None:
    if output_path is None:
        return
    try:
        progress_file = output_path.parent / "progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps({"message": message, "progress": percent}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_keep_aspect(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return image.copy()
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy()
    scale = max_width / float(width)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def resize_to_shape(image: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    if image.shape[:2] == (height, width):
        return image.copy()
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def estimate_gamma(gray: np.ndarray, target: float = 128.0) -> float:
    mean = float(gray.mean())
    if mean <= 1.0 or mean >= 254.0:
        return 1.0
    current = np.clip(mean / 255.0, 1e-4, 1.0 - 1e-4)
    wanted = np.clip(target / 255.0, 1e-4, 1.0 - 1e-4)
    return float(np.clip(np.log(wanted) / np.log(current), 0.7, 1.5))


def adjust_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(float(gamma), 1e-4)
    lut = np.array([((value / 255.0) ** inv) * 255.0 for value in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, lut)


def prepare_gray(image_bgr: np.ndarray, config: PanoramaConfig) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mode = config.preprocess.strip().lower()
    if config.enable_gamma:
        gray = adjust_gamma(gray, estimate_gamma(gray))
    kernel = max(1, int(config.gaussian_kernel))
    if kernel % 2 == 0:
        kernel += 1
    if kernel > 1:
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    if mode == "clahe":
        grid = max(2, int(config.clahe_tile_grid))
        clahe = cv2.createCLAHE(clipLimit=float(config.clahe_clip_limit), tileGridSize=(grid, grid))
        gray = clahe.apply(gray)
    return gray


def keypoints_from_array(array: np.ndarray) -> list[cv2.KeyPoint]:
    keypoints: list[cv2.KeyPoint] = []
    for row in np.asarray(array, dtype=np.float32):
        if len(row) < 7:
            continue
        x, y, size, angle, response, octave, class_id = row[:7]
        keypoints.append(
            cv2.KeyPoint(
                float(x),
                float(y),
                max(float(size), 1e-6),
                float(angle),
                float(response),
                int(octave),
                int(class_id),
            )
        )
    return keypoints


def cache_feature_path(config: PanoramaConfig, image_path: Path, method: str) -> Path | None:
    if not config.feature_cache_root or not config.split_name or not config.scene_id:
        return None
    return (
        Path(config.feature_cache_root)
        / config.split_name
        / config.scene_id
        / method
        / "features"
        / f"{image_path.stem}.npz"
    )


def load_cached_features(config: PanoramaConfig, image_path: Path, method: str) -> FeatureData | None:
    path = cache_feature_path(config, image_path, method)
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        keypoints = keypoints_from_array(archive["keypoints"])
        descriptors = archive["descriptors"]
        image_shape = tuple(int(value) for value in archive["image_shape"].tolist())
    return FeatureData(keypoints, descriptors, image_shape, source="cache")


def harris_keypoints(gray: np.ndarray, config: PanoramaConfig) -> list[cv2.KeyPoint]:
    corner_limit = max(1, min(int(config.harris_max_corners), int(config.max_features)))
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=corner_limit,
        qualityLevel=float(config.harris_quality),
        minDistance=float(config.harris_min_distance),
        blockSize=3,
        useHarrisDetector=True,
        k=0.04,
    )
    if corners is None:
        return []
    return [cv2.KeyPoint(float(x), float(y), 31.0) for x, y in corners.reshape(-1, 2)]


def compute_hog_descriptors(
    gray: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    config: PanoramaConfig,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    if not keypoints:
        return [], None

    patch_size = max(8, int(config.hog_patch_size))
    cells_per_side = max(1, int(config.hog_cells))
    bins = max(4, int(config.hog_bins))
    if patch_size % cells_per_side != 0:
        patch_size = cells_per_side * max(1, patch_size // cells_per_side)

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


def extract_features(gray: np.ndarray, method: str, config: PanoramaConfig) -> FeatureData:
    method = normalize_method(method)
    max_features = max(1, int(config.max_features))
    if method == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is not available in this OpenCV build.")
        detector = cv2.SIFT_create(nfeatures=max_features)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return FeatureData(list(keypoints or []), descriptors, gray.shape[:2], source="computed")
    if method == "AKAZE":
        detector = cv2.AKAZE_create()
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        keypoints = list(keypoints or [])
        if len(keypoints) > max_features:
            order = sorted(range(len(keypoints)), key=lambda i: keypoints[i].response, reverse=True)[:max_features]
            keypoints = [keypoints[i] for i in order]
            descriptors = None if descriptors is None else descriptors[np.array(order)]
        return FeatureData(keypoints, descriptors, gray.shape[:2], source="computed")
    if method == "ORB":
        detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=int(config.orb_fast_threshold))
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return FeatureData(list(keypoints or []), descriptors, gray.shape[:2], source="computed")
    if method == "HARRIS_HOG":
        keypoints = harris_keypoints(gray, config)
        keypoints, descriptors = compute_hog_descriptors(gray, keypoints, config)
        return FeatureData(keypoints, descriptors, gray.shape[:2], source="computed")
    raise ValueError(f"Manual portable stitcher supports ORB, AKAZE, HARRIS_HOG, and SIFT, got {method}.")


def feature_norm(method: str) -> int:
    method = normalize_method(method)
    return cv2.NORM_HAMMING if method in {"ORB", "AKAZE"} else cv2.NORM_L2


def ensure_descriptor_type(descriptors: np.ndarray | None, method: str) -> np.ndarray | None:
    if descriptors is None:
        return None
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)
    if feature_norm(method) == cv2.NORM_HAMMING:
        return descriptors.astype(np.uint8, copy=False)
    return descriptors.astype(np.float32, copy=False)


def get_features(
    image_path: Path,
    image_bgr: np.ndarray,
    method: str,
    config: PanoramaConfig,
    feature_store: dict[tuple[str, str], FeatureData],
) -> FeatureData:
    key = (image_path.name, method)
    if key in feature_store:
        return feature_store[key]
    if config.prefer_cache:
        cached = load_cached_features(config, image_path, method)
        if cached is not None:
            feature_store[key] = cached
            return cached
    gray = prepare_gray(image_bgr, config)
    computed = extract_features(gray, method, config)
    feature_store[key] = computed
    return computed


def match_descriptors(
    desc_a: np.ndarray | None,
    desc_b: np.ndarray | None,
    method: str,
    ratio_test: float,
) -> tuple[list[list[cv2.DMatch]], list[cv2.DMatch], list[float]]:
    desc_a = ensure_descriptor_type(desc_a, method)
    desc_b = ensure_descriptor_type(desc_b, method)
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return [], [], []
    matcher = cv2.BFMatcher(feature_norm(method), crossCheck=False)
    raw_pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    ratios: list[float] = []
    for pair in raw_pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        ratio = math.inf if second.distance <= 1e-12 else float(first.distance / second.distance)
        ratios.append(ratio)
        if first.distance < ratio_test * second.distance:
            good.append(first)
    return raw_pairs, good, ratios


def reprojection_error(src: np.ndarray, dst: np.ndarray, homography: np.ndarray) -> np.ndarray:
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    return np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)


def classify_pair(inliers: int, inlier_ratio: float, mean_error: float | None, config: PanoramaConfig) -> str:
    if mean_error is None or not np.isfinite(mean_error):
        return "failure"
    if inliers >= config.min_inliers * 2 and inlier_ratio >= 0.35 and mean_error <= config.ransac_threshold:
        return "success"
    if inliers >= config.min_inliers and inlier_ratio >= config.min_inlier_ratio and mean_error <= config.ransac_threshold * 2.0:
        return "hard_valid"
    return "failure"


def affine_to_homography(matrix: np.ndarray) -> np.ndarray:
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = matrix.astype(np.float64)
    return homography


def transform_corners_for_shape(shape: tuple[int, int] | tuple[int, int, int], transform: np.ndarray) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, transform).reshape(-1, 2)


def transform_is_sane(transform: np.ndarray, image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[bool, str]:
    if transform is None or transform.shape != (3, 3) or not np.isfinite(transform).all():
        return False, "non-finite transform"
    if abs(float(transform[2, 2])) <= 1e-12:
        return False, "degenerate transform scale"

    height, width = int(image_shape[0]), int(image_shape[1])
    max_dim = float(max(width, height))
    try:
        corners = transform_corners_for_shape(image_shape, transform)
    except cv2.error:
        return False, "corner projection failed"
    if not np.isfinite(corners).all():
        return False, "non-finite projected corners"

    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)
    bbox_w, bbox_h = max_xy - min_xy
    if bbox_w < width * 0.15 or bbox_h < height * 0.15:
        return False, "projected image collapsed"
    if bbox_w > max_dim * 4.0 or bbox_h > max_dim * 4.0:
        return False, "projected image expanded too much"

    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    base_area = float(max(width * height, 1))
    area_ratio = area / base_area
    if area_ratio < 0.08 or area_ratio > 8.0:
        return False, f"unstable area scale {area_ratio:.2f}"
    return True, ""


def estimate_translation_transform(src: np.ndarray, dst: np.ndarray, threshold: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(src) < 2:
        return None, None
    offsets = dst.reshape(-1, 2) - src.reshape(-1, 2)
    center = np.median(offsets, axis=0)
    errors = np.linalg.norm(offsets - center, axis=1)
    inlier_mask = errors <= float(threshold)
    if int(inlier_mask.sum()) < 2:
        return None, None
    refined = offsets[inlier_mask].mean(axis=0)
    homography = np.array(
        [[1.0, 0.0, float(refined[0])], [0.0, 1.0, float(refined[1])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return homography, inlier_mask.reshape(-1, 1).astype(np.uint8)


def estimate_geometric_transform(
    src: np.ndarray,
    dst: np.ndarray,
    config: PanoramaConfig,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[np.ndarray | None, np.ndarray | None, str, str]:
    model = normalize_motion_model(config.manual_motion_model)
    threshold = float(config.ransac_threshold)
    found: np.ndarray | None = None
    mask: np.ndarray | None = None

    if model == "translation":
        found, mask = estimate_translation_transform(src, dst, threshold)
    elif model == "similarity":
        affine, mask = cv2.estimateAffinePartial2D(
            src.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        found = None if affine is None else affine_to_homography(affine)
    elif model == "affine":
        affine, mask = cv2.estimateAffine2D(
            src.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        found = None if affine is None else affine_to_homography(affine)
    else:
        found, mask = cv2.findHomography(
            src.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            cv2.RANSAC,
            threshold,
        )

    if found is None or mask is None:
        return None, None, model, f"{model} estimation failed"
    found = found.astype(np.float64)
    found /= found[2, 2] if abs(found[2, 2]) > 1e-12 else 1.0
    ok, reason = transform_is_sane(found, image_shape)
    if not ok:
        return None, mask, model, reason
    return found, mask, model, ""


def estimate_pair_with_method(
    pair_index: int,
    image_a_path: Path,
    image_b_path: Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    method: str,
    config: PanoramaConfig,
    feature_store: dict[tuple[str, str], FeatureData],
) -> PairEstimate:
    diagnostic_result = estimate_pair_diagnostics(
        pair_index,
        image_a_path,
        image_b_path,
        image_a,
        image_b,
        method,
        config,
        feature_store,
    )
    return diagnostic_result["estimate"]


def pair_score(pair: PairEstimate) -> tuple[int, int, float, float]:
    return (
        STATUS_SCORE.get(pair.status, 0),
        pair.inliers,
        pair.inlier_ratio,
        -float(pair.reprojection_error_mean if pair.reprojection_error_mean is not None else 1e9),
    )


def estimate_pair(
    pair_index: int,
    image_paths: list[Path],
    images: list[np.ndarray],
    config: PanoramaConfig,
    feature_store: dict[tuple[str, str], FeatureData],
) -> PairEstimate:
    pair_id = f"pair_{pair_index + 1:02d}"
    if pair_id in config.pair_method_map:
        methods = [normalize_method(config.pair_method_map[pair_id])]
    elif config.method != "auto":
        methods = [normalize_method(config.method)]
    else:
        methods = [normalize_method(method) for method in config.candidate_methods]

    estimates = [
        estimate_pair_with_method(
            pair_index,
            image_paths[pair_index],
            image_paths[pair_index + 1],
            images[pair_index],
            images[pair_index + 1],
            method,
            config,
            feature_store,
        )
        for method in methods
    ]
    return max(estimates, key=pair_score)


def longest_valid_segment(pair_estimates: list[PairEstimate]) -> tuple[int, int]:
    best_start = 0
    best_end = 0
    start = 0
    while start < len(pair_estimates):
        while start < len(pair_estimates) and pair_estimates[start].homography is None:
            start += 1
        end = start
        while end < len(pair_estimates) and pair_estimates[end].homography is not None:
            end += 1
        if end - start > best_end - best_start:
            best_start, best_end = start, end
        start = end + 1
    return best_start, best_end + 1


def anchor_index_for_count(count: int, anchor: str) -> int:
    token = str(anchor).strip().lower()
    if token == "first":
        return 0
    if token == "last":
        return count - 1
    if token == "middle":
        return count // 2
    value = int(token)
    if value < 0:
        value = count + value
    if value < 0 or value >= count:
        raise ValueError(f"Anchor index {anchor} is outside 0..{count - 1}")
    return value


def chained_transforms(pair_estimates: list[PairEstimate], image_count: int, anchor: str) -> list[np.ndarray]:
    if len(pair_estimates) != image_count - 1:
        raise ValueError("Need one pair homography for every adjacent image pair.")
    if any(pair.homography is None for pair in pair_estimates):
        failed = [pair.pair_id for pair in pair_estimates if pair.homography is None]
        raise ValueError("Cannot chain homographies; failed pairs: " + ", ".join(failed))

    anchor_i = anchor_index_for_count(image_count, anchor)
    transforms: list[np.ndarray | None] = [None] * image_count
    transforms[anchor_i] = np.eye(3, dtype=np.float64)

    for index in range(anchor_i - 1, -1, -1):
        h_to_next = pair_estimates[index].homography
        transforms[index] = transforms[index + 1] @ h_to_next
        transforms[index] /= transforms[index][2, 2] if abs(transforms[index][2, 2]) > 1e-12 else 1.0

    for index in range(anchor_i + 1, image_count):
        h_prev_to_current = pair_estimates[index - 1].homography
        inv_h = np.linalg.inv(h_prev_to_current)
        transforms[index] = transforms[index - 1] @ inv_h
        transforms[index] /= transforms[index][2, 2] if abs(transforms[index][2, 2]) > 1e-12 else 1.0

    return [transform.astype(np.float64) for transform in transforms if transform is not None]


def transformed_corners(image: np.ndarray, transform: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, transform).reshape(-1, 2)


def output_geometry(images: list[np.ndarray], transforms: list[np.ndarray], config: PanoramaConfig) -> tuple[list[np.ndarray], tuple[int, int], np.ndarray, float]:
    all_corners = np.vstack([transformed_corners(image, transform) for image, transform in zip(images, transforms)])
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(np.float64)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(np.float64)
    width = max(1, int(max_x - min_x))
    height = max(1, int(max_y - min_y))

    scale_by_side = 1.0
    max_side = max(width, height)
    if config.max_canvas_side > 0 and max_side > config.max_canvas_side:
        scale_by_side = config.max_canvas_side / float(max_side)

    scale_by_mp = 1.0
    mp_limit = max(float(config.max_canvas_megapixels), 1.0) * 1_000_000.0
    if width * height > mp_limit:
        scale_by_mp = math.sqrt(mp_limit / float(width * height))

    canvas_scale = min(1.0, scale_by_side, scale_by_mp)
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    scale = np.array([[canvas_scale, 0, 0], [0, canvas_scale, 0], [0, 0, 1]], dtype=np.float64)
    adjusted = [scale @ translation @ transform for transform in transforms]
    out_size = (max(1, int(math.ceil(width * canvas_scale))), max(1, int(math.ceil(height * canvas_scale))))
    return adjusted, out_size, translation, canvas_scale


def feather_weight(mask: np.ndarray) -> np.ndarray:
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    max_value = float(distance.max())
    if max_value <= 1e-6:
        return (mask > 0).astype(np.float32)
    return np.clip(distance / max_value, 0.0, 1.0).astype(np.float32)


def blend_images(images: list[np.ndarray], transforms: list[np.ndarray], config: PanoramaConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    adjusted, out_size, translation, canvas_scale = output_geometry(images, transforms, config)
    width, height = out_size
    blend_mode = config.blend_mode.strip().lower()

    import gc
    coverage = np.zeros((height, width), dtype=np.uint8)
    if blend_mode == "overwrite":
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for image, transform in zip(images, adjusted):
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            warped = cv2.warpPerspective(image, transform, (width, height))
            warped_mask = cv2.warpPerspective(mask, transform, (width, height), flags=cv2.INTER_NEAREST)
            keep = warped_mask > 0
            canvas[keep] = warped[keep]
            coverage[keep] = 255
            del mask, warped, warped_mask, keep
            gc.collect()
        return canvas, coverage, {"canvas_scale": canvas_scale, "output_width": width, "output_height": height}

    accum = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for image, transform in zip(images, adjusted):
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped = cv2.warpPerspective(image, transform, (width, height)).astype(np.float32)
        if blend_mode == "feather":
            weight_image = feather_weight(mask)
            warped_weight = cv2.warpPerspective(weight_image, transform, (width, height), flags=cv2.INTER_LINEAR)
            del weight_image
        else:
            warped_mask = cv2.warpPerspective(mask, transform, (width, height), flags=cv2.INTER_NEAREST)
            warped_weight = (warped_mask > 0).astype(np.float32)
            del warped_mask
        accum += warped * warped_weight[..., None]
        weights += warped_weight
        del mask, warped, warped_weight
        gc.collect()

    valid = weights > 1e-6
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[valid] = np.clip(accum[valid] / weights[valid, None], 0, 255).astype(np.uint8)
    coverage[valid] = 255
    del accum, weights, valid
    gc.collect()
    return output, coverage, {"canvas_scale": canvas_scale, "output_width": width, "output_height": height}


def crop_to_coverage(image: np.ndarray, coverage: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = cv2.findNonZero(coverage)
    if coords is None:
        return image, (0, 0, image.shape[1], image.shape[0])
    x, y, width, height = cv2.boundingRect(coords)
    return image[y : y + height, x : x + width], (x, y, width, height)


ALL_DIAGNOSTIC_METHODS = ["ORB", "AKAZE", "HARRIS_HOG", "SIFT"]


def diagnostics_requested(config: PanoramaConfig) -> bool:
    return bool(config.save_debug or config.save_pair_visualizations or config.save_score_table)


def diagnostic_methods(config: PanoramaConfig) -> list[str]:
    raw_methods = config.diagnostics_methods or ALL_DIAGNOSTIC_METHODS
    methods: list[str] = []
    for raw_method in raw_methods:
        if str(raw_method).strip().lower() == "all":
            methods.extend(ALL_DIAGNOSTIC_METHODS)
        else:
            methods.append(normalize_method(str(raw_method)))
    unique: list[str] = []
    for method in methods:
        if method not in unique:
            unique.append(method)
    return unique


def default_diagnostics_dir(output_path: Path, log_path: Path | None, scene_dir: Path, config: PanoramaConfig) -> Path:
    if config.visualization_dir:
        return Path(config.visualization_dir)
    if log_path is not None:
        return log_path.parent / f"{scene_dir.name}_diagnostics"
    return output_path.parent / f"{output_path.stem}_diagnostics"


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_visual_image(path: Path, image: np.ndarray, config: PanoramaConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        quality = int(np.clip(config.visualization_jpeg_quality, 30, 100))
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(str(path), image)


def resize_preview(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image.shape[1] <= max_width:
        return image
    scale = max_width / float(image.shape[1])
    return cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    pad = 34
    header = np.full((pad, output.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(header, text[:160], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([header, output])


def blank_pair_canvas(image_a: np.ndarray, image_b: np.ndarray, label: str) -> np.ndarray:
    height = max(image_a.shape[0], image_b.shape[0])
    width = image_a.shape[1] + image_b.shape[1]
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    canvas[: image_a.shape[0], : image_a.shape[1]] = image_a
    canvas[: image_b.shape[0], image_a.shape[1] : image_a.shape[1] + image_b.shape[1]] = image_b
    return add_label(canvas, label)


def limit_keypoints(keypoints: list[cv2.KeyPoint], max_keypoints: int) -> list[cv2.KeyPoint]:
    if max_keypoints <= 0 or len(keypoints) <= max_keypoints:
        return keypoints
    return sorted(keypoints, key=lambda keypoint: keypoint.response, reverse=True)[:max_keypoints]


def draw_keypoints_visual(image: np.ndarray, keypoints: list[cv2.KeyPoint], label: str, config: PanoramaConfig) -> np.ndarray:
    draw_keypoints = limit_keypoints(keypoints, int(config.visualization_max_keypoints))
    visual = cv2.drawKeypoints(
        image,
        draw_keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    visual = add_label(visual, f"{label} | keypoints drawn: {len(draw_keypoints)}/{len(keypoints)}")
    return resize_preview(visual, int(config.visualization_max_width))


def select_match_subset(matches: list[cv2.DMatch], max_matches: int) -> list[cv2.DMatch]:
    if max_matches <= 0 or len(matches) <= max_matches:
        return matches
    return sorted(matches, key=lambda match: match.distance)[:max_matches]


def draw_matches_visual(
    image_a: np.ndarray,
    image_b: np.ndarray,
    keypoints_a: list[cv2.KeyPoint],
    keypoints_b: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    label: str,
    config: PanoramaConfig,
    color: tuple[int, int, int] = (0, 220, 0),
) -> np.ndarray:
    draw_matches = select_match_subset(matches, int(config.visualization_max_matches))
    if not draw_matches:
        visual = blank_pair_canvas(image_a, image_b, f"{label} | no matches to draw")
    else:
        visual = cv2.drawMatches(
            image_a,
            keypoints_a,
            image_b,
            keypoints_b,
            draw_matches,
            None,
            matchColor=color,
            singlePointColor=(180, 180, 180),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        visual = add_label(visual, f"{label} | matches drawn: {len(draw_matches)}/{len(matches)}")
    return resize_preview(visual, int(config.visualization_max_width))


def draw_pair_warp_preview(
    image_a: np.ndarray,
    image_b: np.ndarray,
    homography: np.ndarray | None,
    label: str,
    config: PanoramaConfig,
) -> np.ndarray | None:
    if homography is None:
        return None
    preview_config = PanoramaConfig(
        blend_mode="average",
        crop=True,
        max_canvas_megapixels=min(float(config.max_canvas_megapixels), 8.0),
        max_canvas_side=min(int(config.max_canvas_side), 5000),
    )
    try:
        panorama, coverage, _ = blend_images([image_a, image_b], [homography, np.eye(3, dtype=np.float64)], preview_config)
        panorama, _ = crop_to_coverage(panorama, coverage)
        panorama = add_label(panorama, label)
        return resize_preview(panorama, int(config.visualization_max_width))
    except Exception:
        return None


def point_bbox_coverage(points: np.ndarray, image_shape: tuple[int, int] | tuple[int, int, int]) -> float:
    if len(points) < 2:
        return 0.0
    height, width = image_shape[:2]
    if width <= 0 or height <= 0:
        return 0.0
    x_span = float(points[:, 0].max() - points[:, 0].min())
    y_span = float(points[:, 1].max() - points[:, 1].min())
    return float(np.clip((x_span * y_span) / float(width * height), 0.0, 1.0))


def spatial_coverage_score(
    keypoints_a: list[cv2.KeyPoint],
    keypoints_b: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    inlier_mask: np.ndarray | None,
    image_a_shape: tuple[int, int] | tuple[int, int, int],
    image_b_shape: tuple[int, int] | tuple[int, int, int],
) -> float:
    if inlier_mask is None or len(inlier_mask) != len(good_matches) or not np.any(inlier_mask):
        return 0.0

    inlier_matches = [match for match, keep in zip(good_matches, inlier_mask) if keep]
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in inlier_matches])
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in inlier_matches])
    coverage_a = point_bbox_coverage(points_a, image_a_shape)
    coverage_b = point_bbox_coverage(points_b, image_b_shape)
    return float((coverage_a + coverage_b) / 2.0)


def signed_polygon_area(points: np.ndarray) -> float:
    pts = points.reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def homography_sanity_score(
    homography: np.ndarray | None,
    image_a_shape: tuple[int, int] | tuple[int, int, int],
    image_b_shape: tuple[int, int] | tuple[int, int, int],
) -> float:
    if homography is None or not np.all(np.isfinite(homography)):
        return 0.0

    height_a, width_a = image_a_shape[:2]
    height_b, width_b = image_b_shape[:2]
    if width_a <= 0 or height_a <= 0 or width_b <= 0 or height_b <= 0:
        return 0.0

    corners_a = np.float32([[0, 0], [width_a, 0], [width_a, height_a], [0, height_a]]).reshape(-1, 1, 2)
    corners_b = np.float32([[0, 0], [width_b, 0], [width_b, height_b], [0, height_b]]).reshape(-1, 1, 2)
    try:
        warped_corners_a = cv2.perspectiveTransform(corners_a, homography).reshape(-1, 2)
    except Exception:
        return 0.0

    if not np.all(np.isfinite(warped_corners_a)):
        return 0.0

    signed_area = signed_polygon_area(warped_corners_a)
    warped_area = abs(signed_area)
    original_area = float(width_a * height_a)
    area_ratio = warped_area / original_area if original_area else np.inf

    side_lengths = np.linalg.norm(warped_corners_a - np.roll(warped_corners_a, -1, axis=0), axis=1)
    original_lengths = np.array([width_a, height_a, width_a, height_a], dtype=np.float32)
    scale_ratios = side_lengths / np.maximum(original_lengths, 1.0)

    try:
        all_corners = np.concatenate([warped_corners_a.reshape(-1, 1, 2), corners_b], axis=0).reshape(-1, 2)
        min_xy = np.floor(all_corners.min(axis=0))
        max_xy = np.ceil(all_corners.max(axis=0))
        canvas_width, canvas_height = max_xy - min_xy
        canvas_area = float(canvas_width * canvas_height)
    except Exception:
        canvas_area = np.inf

    input_area = float(width_a * height_a + width_b * height_b)
    canvas_ratio = canvas_area / input_area if input_area else np.inf

    checks = [
        signed_area > 0.0,
        0.15 <= area_ratio <= 6.0,
        float(np.nanmin(scale_ratios)) >= 0.15,
        float(np.nanmax(scale_ratios)) <= 6.0,
        canvas_ratio <= 4.0,
    ]
    return float(sum(bool(check) for check in checks) / len(checks))


def warp_pair_to_canvas(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    homography_a_to_b: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    if homography_a_to_b is None:
        return None, None, None, None, "No homography"

    height_a, width_a = image_a_bgr.shape[:2]
    height_b, width_b = image_b_bgr.shape[:2]

    corners_a = np.float32(
        [[0, 0], [width_a, 0], [width_a, height_a], [0, height_a]]
    ).reshape(-1, 1, 2)
    corners_b = np.float32(
        [[0, 0], [width_b, 0], [width_b, height_b], [0, height_b]]
    ).reshape(-1, 1, 2)

    try:
        warped_corners_a = cv2.perspectiveTransform(corners_a, homography_a_to_b)
        all_corners = np.concatenate([warped_corners_a, corners_b], axis=0)
        min_xy = np.floor(all_corners.min(axis=0).ravel()).astype(int)
        max_xy = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    except Exception as e:
        return None, None, None, None, f"Error projecting corners: {e}"

    min_x, min_y = int(min_xy[0]), int(min_xy[1])
    max_x, max_y = int(max_xy[0]), int(max_xy[1])
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    if canvas_width <= 0 or canvas_height <= 0:
        return None, None, None, None, "Invalid panorama canvas"
    if canvas_width * canvas_height > 20000000:
        return None, None, None, None, f"Panorama canvas too large: {canvas_width}x{canvas_height}"

    offset = np.array(
        [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    try:
        warped_a = cv2.warpPerspective(
            image_a_bgr,
            offset @ homography_a_to_b,
            (canvas_width, canvas_height),
        )
        mask_a = cv2.warpPerspective(
            np.full((height_a, width_a), 255, dtype=np.uint8),
            offset @ homography_a_to_b,
            (canvas_width, canvas_height),
        ) > 0

        canvas_b = np.zeros_like(warped_a)
        mask_b = np.zeros((canvas_height, canvas_width), dtype=bool)
        tx, ty = -min_x, -min_y
        canvas_b[ty : ty + height_b, tx : tx + width_b] = image_b_bgr
        mask_b[ty : ty + height_b, tx : tx + width_b] = True

        return warped_a, canvas_b, mask_a, mask_b, "OK"
    except Exception as e:
        return None, None, None, None, f"Warping error: {e}"


def overlap_similarity_score(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    homography_a_to_b: np.ndarray | None,
    min_pixels: int = 200,
) -> tuple[float, int]:
    warped_a, canvas_b, mask_a, mask_b, status = warp_pair_to_canvas(image_a_bgr, image_b_bgr, homography_a_to_b)
    if status != "OK" or warped_a is None or canvas_b is None or mask_a is None or mask_b is None:
        return 0.0, 0

    overlap = mask_a & mask_b
    overlap_pixels = int(overlap.sum())
    if overlap_pixels < min_pixels:
        return 0.0, overlap_pixels

    try:
        gray_warped_a = cv2.cvtColor(warped_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_canvas_b = cv2.cvtColor(canvas_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        values_a = gray_warped_a[overlap]
        values_b = gray_canvas_b[overlap]

        values_a = values_a - float(values_a.mean())
        values_b = values_b - float(values_b.mean())
        denom = float(np.linalg.norm(values_a) * np.linalg.norm(values_b))
        if denom <= 1e-12:
            return 0.0, overlap_pixels

        ncc = float(np.dot(values_a, values_b) / denom)
        return float(np.clip((ncc + 1.0) / 2.0, 0.0, 1.0)), overlap_pixels
    except Exception:
        return 0.0, overlap_pixels


def estimate_pair_diagnostics(
    pair_index: int,
    image_a_path: Path,
    image_b_path: Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    method: str,
    config: PanoramaConfig,
    feature_store: dict[tuple[str, str], FeatureData],
) -> dict[str, Any]:
    pair_id = f"pair_{pair_index + 1:02d}"
    try:
        feature_a = get_features(image_a_path, image_a, method, config, feature_store)
        feature_b = get_features(image_b_path, image_b, method, config, feature_store)
        raw_pairs, good_matches, ratios = match_descriptors(
            feature_a.descriptors,
            feature_b.descriptors,
            method,
            config.ratio_test,
        )
        lowe_pass_rate = float(len(good_matches) / max(len(raw_pairs), 1))
        median_lowe_ratio = float(np.median(ratios)) if ratios else None
        homography: np.ndarray | None = None
        homography_for_preview: np.ndarray | None = None
        inlier_mask: np.ndarray | None = None
        inliers = 0
        inlier_ratio = 0.0
        mean_error: float | None = None
        median_error: float | None = None
        error = ""
        motion_model = normalize_motion_model(config.manual_motion_model)

        inlier_lowe_ratio: float | None = None
        spatial_coverage: float | None = None
        overlap_similarity: float | None = None
        overlap_pixels: int = 0
        homography_sanity: float | None = None

        if len(good_matches) >= max(4, int(config.min_good_matches)):
            src = np.float32([feature_a.keypoints[m.queryIdx].pt for m in good_matches])
            dst = np.float32([feature_b.keypoints[m.trainIdx].pt for m in good_matches])
            found_h, mask, motion_model, error = estimate_geometric_transform(
                src,
                dst,
                config,
                image_a.shape,
            )
            if found_h is not None and mask is not None:
                homography_for_preview = found_h.astype(np.float64)
                inlier_mask = mask.ravel().astype(bool)
                inliers = int(inlier_mask.sum())
                inlier_ratio = float(inliers / max(len(good_matches), 1))
                errors = reprojection_error(src[inlier_mask], dst[inlier_mask], found_h) if inliers else np.array([])
                mean_error = float(np.mean(errors)) if len(errors) else None
                median_error = float(np.median(errors)) if len(errors) else None
                status = classify_pair(inliers, inlier_ratio, mean_error, config)
                homography = homography_for_preview if status != "failure" else None

                # Compute extra metrics
                # 1. inlier_lowe_ratio
                ratio_by_match = {}
                ratio_idx = 0
                for pair in raw_pairs:
                    if len(pair) < 2:
                        continue
                    first, _ = pair
                    if ratio_idx < len(ratios):
                        ratio_by_match[(first.queryIdx, first.trainIdx)] = ratios[ratio_idx]
                        ratio_idx += 1

                inlier_matches = [m for m, keep in zip(good_matches, inlier_mask) if keep]
                inlier_ratios = [ratio_by_match[(m.queryIdx, m.trainIdx)] for m in inlier_matches if (m.queryIdx, m.trainIdx) in ratio_by_match]
                if inlier_ratios:
                    inlier_lowe_ratio = float(np.median(inlier_ratios))

                # 2. spatial_coverage
                spatial_coverage = spatial_coverage_score(
                    feature_a.keypoints,
                    feature_b.keypoints,
                    good_matches,
                    inlier_mask,
                    image_a.shape,
                    image_b.shape
                )

                # 3. homography_sanity
                homography_sanity = homography_sanity_score(
                    homography_for_preview,
                    image_a.shape,
                    image_b.shape
                )

                # 4. overlap_similarity & overlap_pixels
                overlap_similarity, overlap_pixels = overlap_similarity_score(
                    image_a,
                    image_b,
                    homography_for_preview
                )
            else:
                status = "failure"
                error = error or f"{motion_model} estimation failed"
        else:
            status = "failure"
            error = "not enough good matches"

        estimate = PairEstimate(
            pair_id=pair_id,
            pair_index=pair_index + 1,
            image_a=image_a_path.name,
            image_b=image_b_path.name,
            method=method,
            motion_model=motion_model,
            status=status,
            homography=homography,
            raw_matches=len(raw_pairs),
            good_matches=len(good_matches),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reprojection_error_mean=mean_error,
            error=error,
            feature_source_a=feature_a.source,
            feature_source_b=feature_b.source,
            keypoints_a=len(feature_a.keypoints),
            keypoints_b=len(feature_b.keypoints),
            lowe_pass_rate=lowe_pass_rate,
            median_lowe_ratio=median_lowe_ratio,
            reprojection_error_median=median_error,
            inlier_lowe_ratio=inlier_lowe_ratio,
            spatial_coverage=spatial_coverage,
            overlap_similarity=overlap_similarity,
            overlap_pixels=overlap_pixels,
            homography_sanity=homography_sanity,
        )
        return {
            "estimate": estimate,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "good_matches": good_matches,
            "inlier_mask": inlier_mask,
            "homography_for_preview": homography_for_preview,
        }
    except Exception as exc:
        estimate = PairEstimate(
            pair_id=pair_id,
            pair_index=pair_index + 1,
            image_a=image_a_path.name,
            image_b=image_b_path.name,
            method=method,
            motion_model=normalize_motion_model(config.manual_motion_model),
            status="failure",
            homography=None,
            raw_matches=0,
            good_matches=0,
            inliers=0,
            inlier_ratio=0.0,
            reprojection_error_mean=None,
            error=str(exc),
        )
        return {
            "estimate": estimate,
            "feature_a": None,
            "feature_b": None,
            "good_matches": [],
            "inlier_mask": None,
            "homography_for_preview": None,
        }


SCORE_FIELDS = [
    "pair_id",
    "pair_index",
    "image_a",
    "image_b",
    "method",
    "motion_model",
    "selected_for_panorama",
    "status",
    "feature_source_a",
    "feature_source_b",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "good_matches",
    "lowe_pass_rate",
    "median_lowe_ratio",
    "inliers",
    "inlier_ratio",
    "reprojection_error_mean",
    "reprojection_error_median",
    "inlier_lowe_ratio",
    "spatial_coverage",
    "overlap_similarity",
    "overlap_pixels",
    "homography_sanity",
    "error",
]


def score_row_from_estimate(estimate: PairEstimate, selected_method: str | None) -> dict[str, Any]:
    def fmt(value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return ""
        return f"{float(value):.4f}"

    return {
        "pair_id": estimate.pair_id,
        "pair_index": estimate.pair_index,
        "image_a": estimate.image_a,
        "image_b": estimate.image_b,
        "method": estimate.method,
        "motion_model": estimate.motion_model,
        "selected_for_panorama": "yes" if selected_method == estimate.method else "",
        "status": estimate.status,
        "feature_source_a": estimate.feature_source_a,
        "feature_source_b": estimate.feature_source_b,
        "keypoints_a": estimate.keypoints_a,
        "keypoints_b": estimate.keypoints_b,
        "raw_matches": estimate.raw_matches,
        "good_matches": estimate.good_matches,
        "lowe_pass_rate": fmt(estimate.lowe_pass_rate),
        "median_lowe_ratio": fmt(estimate.median_lowe_ratio),
        "inliers": estimate.inliers,
        "inlier_ratio": fmt(estimate.inlier_ratio),
        "reprojection_error_mean": fmt(estimate.reprojection_error_mean),
        "reprojection_error_median": fmt(estimate.reprojection_error_median),
        "inlier_lowe_ratio": fmt(estimate.inlier_lowe_ratio),
        "spatial_coverage": fmt(estimate.spatial_coverage),
        "overlap_similarity": fmt(estimate.overlap_similarity),
        "overlap_pixels": estimate.overlap_pixels,
        "homography_sanity": fmt(estimate.homography_sanity),
        "error": estimate.error,
    }


def save_table_image(path: Path, rows: list[dict[str, Any]], title: str, columns: list[tuple[str, str]], config: PanoramaConfig) -> None:
    if not rows:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    row_height = 28
    title_height = 42
    padding = 10
    widths: list[int] = []
    for key, label in columns:
        longest = max([len(str(row.get(key, ""))) for row in rows] + [len(label)])
        widths.append(int(np.clip(longest * 8 + 18, 70, 210)))
    width = sum(widths) + 2 * padding
    height = title_height + row_height * (len(rows) + 1) + padding
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, title[:160], (padding, 27), font, 0.62, (25, 25, 25), 1, cv2.LINE_AA)

    y = title_height
    x = padding
    for (key, label), col_width in zip(columns, widths):
        cv2.rectangle(canvas, (x, y), (x + col_width, y + row_height), (232, 232, 232), -1)
        cv2.putText(canvas, label[:24], (x + 5, y + 19), font, font_scale, (20, 20, 20), 1, cv2.LINE_AA)
        x += col_width
    y += row_height
    for row_index, row in enumerate(rows):
        x = padding
        fill = (250, 250, 250) if row_index % 2 == 0 else (242, 242, 242)
        for (key, _), col_width in zip(columns, widths):
            cv2.rectangle(canvas, (x, y), (x + col_width, y + row_height), fill, -1)
            value = str(row.get(key, ""))[:28]
            color = (30, 30, 30)
            if key == "status" and value == "failure":
                color = (30, 30, 180)
            elif key == "status" and value == "success":
                color = (30, 130, 30)
            elif key == "selected_for_panorama" and value:
                color = (180, 80, 20)
            cv2.putText(canvas, value, (x + 5, y + 19), font, font_scale, color, 1, cv2.LINE_AA)
            x += col_width
        y += row_height
    canvas = resize_preview(canvas, int(config.visualization_max_width))
    save_visual_image(path, canvas, config)


def save_descriptor_diagnostics(
    scene_dir: Path,
    image_paths: list[Path],
    images: list[np.ndarray],
    config: PanoramaConfig,
    feature_store: dict[tuple[str, str], FeatureData],
    selected_pairs: list[PairEstimate],
    output_path: Path,
    log_path: Path | None,
) -> dict[str, Any]:
    if not diagnostics_requested(config):
        return {}

    diagnostics_dir = default_diagnostics_dir(output_path, log_path, scene_dir, config)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    methods = diagnostic_methods(config)
    selected_by_pair = {pair.pair_id: pair.method for pair in selected_pairs}
    rows: list[dict[str, Any]] = []

    for pair_index in range(len(images) - 1):
        pair_id = f"pair_{pair_index + 1:02d}"
        pair_rows: list[dict[str, Any]] = []
        for method_idx, method in enumerate(methods):
            progress_val = 65 + int(((pair_index * len(methods) + method_idx) / max(1, (len(images) - 1) * len(methods))) * 25)
            write_progress(output_path, f"Generating {method} diagnostics for Pair {pair_index + 1}...", progress_val)
            result = estimate_pair_diagnostics(
                pair_index,
                image_paths[pair_index],
                image_paths[pair_index + 1],
                images[pair_index],
                images[pair_index + 1],
                method,
                config,
                feature_store,
            )
            estimate: PairEstimate = result["estimate"]
            row = score_row_from_estimate(estimate, selected_by_pair.get(pair_id))
            rows.append(row)
            pair_rows.append(row)

            if config.save_pair_visualizations or config.save_debug:
                method_dir = diagnostics_dir / pair_id / method
                feature_a: FeatureData | None = result.get("feature_a")
                feature_b: FeatureData | None = result.get("feature_b")
                if feature_a is not None:
                    visual = draw_keypoints_visual(images[pair_index], feature_a.keypoints, f"{pair_id} {method} keypoints A", config)
                    save_visual_image(method_dir / f"{pair_id}_{method}_keypoints_a.jpg", visual, config)
                if feature_b is not None:
                    visual = draw_keypoints_visual(images[pair_index + 1], feature_b.keypoints, f"{pair_id} {method} keypoints B", config)
                    save_visual_image(method_dir / f"{pair_id}_{method}_keypoints_b.jpg", visual, config)
                if feature_a is not None and feature_b is not None:
                    good_matches: list[cv2.DMatch] = result.get("good_matches") or []
                    visual = draw_matches_visual(
                        images[pair_index],
                        images[pair_index + 1],
                        feature_a.keypoints,
                        feature_b.keypoints,
                        good_matches,
                        f"{pair_id} {method} Lowe-filtered matches",
                        config,
                        color=(0, 190, 255),
                    )
                    save_visual_image(method_dir / f"{pair_id}_{method}_good_matches.jpg", visual, config)

                    inlier_mask = result.get("inlier_mask")
                    if inlier_mask is not None and len(good_matches):
                        inlier_matches = [match for match, keep in zip(good_matches, inlier_mask) if bool(keep)]
                        visual = draw_matches_visual(
                            images[pair_index],
                            images[pair_index + 1],
                            feature_a.keypoints,
                            feature_b.keypoints,
                            inlier_matches,
                            f"{pair_id} {method} RANSAC inliers",
                            config,
                            color=(0, 220, 0),
                        )
                        save_visual_image(method_dir / f"{pair_id}_{method}_inlier_matches.jpg", visual, config)

                    warp_preview = draw_pair_warp_preview(
                        images[pair_index],
                        images[pair_index + 1],
                        result.get("homography_for_preview"),
                        f"{pair_id} {method} pair warp preview",
                        config,
                    )
                    if warp_preview is not None:
                        save_visual_image(method_dir / f"{pair_id}_{method}_pair_warp_preview.jpg", warp_preview, config)

        if config.save_score_table or config.save_debug:
            pair_csv = diagnostics_dir / pair_id / f"{pair_id}_descriptor_scores.csv"
            write_rows_csv(pair_csv, pair_rows, SCORE_FIELDS)
            save_table_image(
                diagnostics_dir / pair_id / f"{pair_id}_descriptor_scores.png",
                pair_rows,
                f"{scene_dir.name} {pair_id} descriptor comparison",
                [
                    ("method", "Method"),
                    ("motion_model", "Geometry"),
                    ("selected_for_panorama", "Selected"),
                    ("status", "Status"),
                    ("good_matches", "Good"),
                    ("inliers", "Inliers"),
                    ("inlier_ratio", "Inlier"),
                    ("reprojection_error_mean", "Reproj"),
                    ("lowe_pass_rate", "Lowe pass"),
                ],
                config,
            )

    score_csv = diagnostics_dir / "descriptor_comparison_scores.csv"
    write_rows_csv(score_csv, rows, SCORE_FIELDS)
    save_table_image(
        diagnostics_dir / "descriptor_comparison_scores.png",
        rows,
        f"{scene_dir.name} all-pair descriptor comparison",
        [
            ("pair_id", "Pair"),
            ("method", "Method"),
            ("motion_model", "Geometry"),
            ("selected_for_panorama", "Selected"),
            ("status", "Status"),
            ("good_matches", "Good"),
            ("inliers", "Inliers"),
            ("inlier_ratio", "Inlier"),
            ("reprojection_error_mean", "Reproj"),
        ],
        config,
    )
    summary_json = diagnostics_dir / "descriptor_comparison_scores.json"
    summary_json.write_text(
        json.dumps(
            {
                "scene": scene_dir.name,
                "methods": methods,
                "pair_count": max(0, len(images) - 1),
                "score_csv": str(score_csv),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "diagnostics_dir": str(diagnostics_dir),
        "descriptor_score_csv": str(score_csv),
        "descriptor_score_table": str(diagnostics_dir / "descriptor_comparison_scores.png"),
        "descriptor_score_json": str(summary_json),
        "diagnostic_methods": methods,
        "diagnostic_row_count": len(rows),
    }


def prepare_work_images(image_paths: list[Path], config: PanoramaConfig) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    if config.prefer_cache and config.feature_cache_root and config.candidate_methods:
        method = config.candidate_methods[0] if config.method == "auto" else config.method
        shapes: list[tuple[int, int] | None] = []
        for path in image_paths:
            cached = load_cached_features(config, path, normalize_method(method))
            shapes.append(None if cached is None else cached.image_shape)
        if all(shape is not None for shape in shapes):
            for path, shape in zip(image_paths, shapes):
                if shape is not None:
                    images.append(resize_to_shape(load_bgr(path), shape))
            return images

    for path in image_paths:
        images.append(resize_keep_aspect(load_bgr(path), int(config.work_width)))
    return images


def stitch_manual_scene(
    scene_dir: Path,
    output_path: Path,
    config: PanoramaConfig,
    log_path: Path | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    config = finalize_config(config)
    scene_dir = scene_dir.resolve()
    scene_metadata = read_scene_metadata(scene_dir)
    write_progress(output_path, "Loading and decoding input images...", 10)
    image_paths = select_input_images(scene_dir, config)
    original_image_count = len(image_paths)
    images = prepare_work_images(image_paths, config)

    write_progress(output_path, f"Detecting features & estimating homographies using {config.method}...", 30)
    feature_store: dict[tuple[str, str], FeatureData] = {}
    pair_estimates = []
    for index in range(len(images) - 1):
        progress_val = 30 + int((index / max(1, len(images) - 1)) * 30)
        write_progress(output_path, f"Estimating keypoint transformations for Pair {index + 1}/{len(images) - 1}...", progress_val)
        pair_estimates.append(estimate_pair(index, image_paths, images, config, feature_store))
    
    write_progress(output_path, "Running descriptor diagnostics...", 65)
    all_pair_estimates = list(pair_estimates)
    diagnostics = save_descriptor_diagnostics(
        scene_dir,
        image_paths,
        images,
        config,
        feature_store,
        pair_estimates,
        output_path,
        log_path,
    )

    used_image_offset = 0
    failed_pairs = [pair.pair_id for pair in pair_estimates if pair.homography is None]
    if any(pair.homography is None for pair in pair_estimates):
        if not config.allow_partial:
            elapsed = time.perf_counter() - start
            payload = {
                "status": "error",
                "engine": "manual",
                "error_type": "manual_chain_failure",
                "message": "Manual stitcher could not bridge the whole scene: " + ", ".join(failed_pairs),
                "user_message": "The images do not form a reliable full panorama. Try weak_phone/student_debug preset with allow_partial, lower work_width, or recapture with more overlap.",
                "scene_dir": str(scene_dir),
                "scene_id": config.scene_id or scene_dir.name,
                "split_name": config.split_name,
                "output_path": str(output_path),
                "image_count": len(image_paths),
                "images": [path.name for path in image_paths],
                "scene_metadata": scene_metadata,
                "config": asdict(config),
                "failed_pairs": failed_pairs,
                "pairs": [pair.to_json() for pair in pair_estimates],
                "diagnostics": diagnostics,
                "runtime_sec": float(elapsed),
            }
            write_json_log(log_path, payload)
            raise PanoramaPipelineError(payload["message"], payload)
        start_pair, end_image = longest_valid_segment(pair_estimates)
        if end_image - start_pair < 2:
            elapsed = time.perf_counter() - start
            payload = {
                "status": "error",
                "engine": "manual",
                "error_type": "no_valid_partial_segment",
                "message": "No valid partial segment found for manual stitching.",
                "user_message": "No adjacent image pair was reliable enough to stitch. Try better-lit images with more overlap.",
                "scene_dir": str(scene_dir),
                "scene_id": config.scene_id or scene_dir.name,
                "split_name": config.split_name,
                "output_path": str(output_path),
                "image_count": len(image_paths),
                "images": [path.name for path in image_paths],
                "scene_metadata": scene_metadata,
                "config": asdict(config),
                "failed_pairs": failed_pairs,
                "pairs": [pair.to_json() for pair in pair_estimates],
                "diagnostics": diagnostics,
                "runtime_sec": float(elapsed),
            }
            write_json_log(log_path, payload)
            raise PanoramaPipelineError(payload["message"], payload)
        images = images[start_pair:end_image]
        image_paths = image_paths[start_pair:end_image]
        pair_estimates = pair_estimates[start_pair : end_image - 1]
        used_image_offset = start_pair

    write_progress(output_path, "Blending canvas edges...", 90)
    transforms = chained_transforms(pair_estimates, len(images), config.anchor)
    panorama, coverage, geometry = blend_images(images, transforms, config)
    crop_box = (0, 0, panorama.shape[1], panorama.shape[0])
    if config.crop:
        panorama, crop_box = crop_to_coverage(panorama, coverage)

    write_progress(output_path, "Writing panorama output file...", 95)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), panorama):
        raise OSError(f"Could not write panorama: {output_path}")

    elapsed = time.perf_counter() - start
    payload = {
        "status": "ok",
        "engine": "manual",
        "scene_dir": str(scene_dir),
        "scene_id": config.scene_id or scene_dir.name,
        "split_name": config.split_name,
        "output_path": str(output_path),
        "image_count": len(image_paths),
        "original_image_count": original_image_count,
        "is_partial": len(image_paths) != original_image_count or used_image_offset != 0,
        "used_image_offset": used_image_offset,
        "images": [path.name for path in image_paths],
        "scene_metadata": scene_metadata,
        "config": asdict(config),
        "pairs": [pair.to_json() for pair in pair_estimates],
        "all_pairs": [pair.to_json() for pair in all_pair_estimates],
        "failed_pairs": failed_pairs,
        "geometry": geometry,
        "crop_box": list(crop_box),
        "panorama_shape": {"height": int(panorama.shape[0]), "width": int(panorama.shape[1])},
        "quality_warnings": quality_warnings(scene_metadata, all_pair_estimates, len(image_paths) != original_image_count or used_image_offset != 0),
        "diagnostics": diagnostics,
        "runtime_sec": float(elapsed),
    }
    write_json_log(log_path, payload)
    return payload


def stitch_opencv_scene(
    scene_dir: Path,
    output_path: Path,
    config: PanoramaConfig,
    log_path: Path | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    config = finalize_config(config)
    scene_dir = scene_dir.resolve()
    scene_metadata = read_scene_metadata(scene_dir)
    write_progress(output_path, "Loading input images...", 10)
    image_paths = select_input_images(scene_dir, config)
    images = [resize_keep_aspect(load_bgr(path), config.work_width) for path in image_paths]
    mode = config.stitcher_mode.strip().upper()
    stitcher_mode = cv2.Stitcher_SCANS if mode == "SCANS" else cv2.Stitcher_PANORAMA
    stitcher = cv2.Stitcher_create(stitcher_mode)
    
    write_progress(output_path, "Running OpenCV Stitcher engine...", 40)
    status_code, panorama = stitcher.stitch(images)
    status_name = {
        int(cv2.Stitcher_OK): "OK",
        int(cv2.Stitcher_ERR_NEED_MORE_IMGS): "ERR_NEED_MORE_IMGS",
        int(cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL): "ERR_HOMOGRAPHY_EST_FAIL",
        int(cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL): "ERR_CAMERA_PARAMS_ADJUST_FAIL",
    }.get(int(status_code), f"UNKNOWN_{status_code}")
    if status_code != int(cv2.Stitcher_OK) or panorama is None:
        payload = {
            "status": "error",
            "engine": "opencv",
            "error_type": "opencv_stitcher_failure",
            "message": f"OpenCV Stitcher failed with status {status_name}",
            "user_message": "OpenCV could not estimate a full panorama. Try manual engine with --allow-partial, lower work_width, or recapture with more overlap.",
            "stitcher_mode": mode,
            "stitcher_status_code": int(status_code),
            "stitcher_status_name": status_name,
            "scene_dir": str(scene_dir),
            "scene_id": config.scene_id or scene_dir.name,
            "split_name": config.split_name,
            "output_path": str(output_path),
            "image_count": len(image_paths),
            "images": [path.name for path in image_paths],
            "scene_metadata": scene_metadata,
            "config": asdict(config),
            "runtime_sec": float(time.perf_counter() - start),
        }
        write_json_log(log_path, payload)
        raise PanoramaPipelineError(payload["message"], payload)
    
    write_progress(output_path, "Writing panorama output file...", 80)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), panorama):
        raise OSError(f"Could not write panorama: {output_path}")
    
    write_progress(output_path, "Generating OpenCV descriptor diagnostics...", 85)
    feature_store: dict[tuple[str, str], FeatureData] = {}
    diagnostics = save_descriptor_diagnostics(
        scene_dir,
        image_paths,
        images,
        config,
        feature_store,
        [],
        output_path,
        log_path,
    )
    payload = {
        "status": "ok",
        "engine": "opencv",
        "stitcher_mode": mode,
        "stitcher_status_code": int(status_code),
        "stitcher_status_name": status_name,
        "scene_dir": str(scene_dir),
        "scene_id": config.scene_id or scene_dir.name,
        "split_name": config.split_name,
        "output_path": str(output_path),
        "image_count": len(image_paths),
        "images": [path.name for path in image_paths],
        "scene_metadata": scene_metadata,
        "config": asdict(config),
        "panorama_shape": {"height": int(panorama.shape[0]), "width": int(panorama.shape[1])},
        "quality_warnings": quality_warnings(scene_metadata, [], False),
        "diagnostics": diagnostics,
        "runtime_sec": float(time.perf_counter() - start),
    }
    write_json_log(log_path, payload)
    return payload


def stitch_scene_folder(
    scene_folder: str | Path,
    output_path: str | Path,
    config: PanoramaConfig | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = finalize_config(config or PanoramaConfig())
    scene_dir = Path(scene_folder)
    out_path = Path(output_path)
    log = None if log_path is None else Path(log_path)
    if cfg.engine.strip().lower() == "opencv":
        return stitch_opencv_scene(scene_dir, out_path, cfg, log)
    return stitch_manual_scene(scene_dir, out_path, cfg, log)


def load_pair_method_map(csv_path: Path, split_name: str, scene_id: str, allowed_methods: list[str]) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    allowed = {normalize_method(method) for method in allowed_methods}
    by_pair: dict[str, list[dict[str, str]]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") != split_name or row.get("scene_id") != scene_id:
                continue
            method = normalize_method(row.get("method", ""))
            if method not in allowed:
                continue
            by_pair.setdefault(row.get("pair_id", ""), []).append(row)

    selected: dict[str, str] = {}
    for pair_id, rows in by_pair.items():
        def score(row: dict[str, str]) -> tuple[int, int, float, float]:
            status = row.get("status", "failure")
            inliers = int(float(row.get("inliers") or 0))
            ratio = float(row.get("inlier_ratio") or 0.0)
            err_text = row.get("reprojection_error_mean")
            try:
                error = float(err_text) if err_text not in {"", None, "nan"} else 1e9
            except ValueError:
                error = 1e9
            return STATUS_SCORE.get(status, 0), inliers, ratio, -error

        best = max(rows, key=score)
        selected[pair_id] = normalize_method(best.get("method", "ORB"))
    return selected


def print_presets() -> None:
    print("Available presets:")
    for name in ["normal_phone", "weak_phone", "best_quality", "student_debug", "custom"]:
        print(f"  {name:13s} - {PRESET_DESCRIPTIONS[name]}")


def config_payload(config: PanoramaConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["preset_description"] = PRESET_DESCRIPTIONS.get(config.preset, "")
    return payload


def load_config_json(path: Path) -> PanoramaConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "config" in data and isinstance(data["config"], dict):
        data = data["config"]
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object or an object with a 'config' field.")

    allowed = set(PanoramaConfig.__dataclass_fields__)
    filtered = {key: value for key, value in data.items() if key in allowed}
    config = PanoramaConfig(**filtered)
    # A JSON config is treated as already materialized user/app settings.
    # Pass --preset explicitly if you want to rebuild it from a preset.
    config.preset_applied = bool(filtered.get("preset_applied", True))
    config.method = normalize_method(config.method)
    config.candidate_methods = [normalize_method(method) for method in config.candidate_methods]
    config.diagnostics_methods = [str(method) for method in config.diagnostics_methods]
    config.manual_motion_model = normalize_motion_model(config.manual_motion_model)
    return config


def write_config_template(path: Path, preset: str = "normal_phone") -> None:
    config = finalize_config(PanoramaConfig(preset=preset))
    template = {
        "description": "Portable panorama pipeline config. Edit values, then run with --config-json this_file.json.",
        "preset_options": PRESET_DESCRIPTIONS,
        "config": config_payload(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")


def provided_cli_options(argv: list[str]) -> set[str]:
    provided: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        provided.add(token.split("=", 1)[0])
    return provided


def should_apply_arg(provided: set[str], option: str, use_cli_defaults: bool) -> bool:
    return use_cli_defaults or option in provided


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Portable single-scene panorama pipeline. Run with --scene-folder/--output for CLI/app use, "
            "or omit them to open a beginner-friendly wizard."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=sorted(PRESET_NAMES), default="custom", help="User-friendly processing preset.")
    parser.add_argument("--print-presets", action="store_true", help="Print preset descriptions and exit.")
    parser.add_argument("--config-json", type=Path, default=None, help="Load PanoramaConfig values from JSON.")
    parser.add_argument("--write-config-template", type=Path, default=None, help="Write an editable config JSON template and exit.")
    parser.add_argument("--show-traceback", action="store_true", help="Include Python traceback in structured error JSON.")
    parser.add_argument("--scene-folder", type=Path, default=None, help="Folder containing one ordered panorama scene.")
    parser.add_argument("--output", type=Path, default=None, help="Output panorama image path.")
    parser.add_argument("--log", type=Path, default=None, help="Optional JSON log path.")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for app/CLI symmetry; output images are overwritten by default.")
    parser.add_argument("--engine", choices=["manual", "opencv"], default="manual")
    parser.add_argument("--profile", choices=["fast", "balanced", "quality"], default="balanced")
    parser.add_argument("--method", default="auto", help="auto, ORB, AKAZE, HARRIS_HOG, or SIFT.")
    parser.add_argument("--candidate-method", action="append", dest="candidate_methods", help="Candidate for --method auto. Repeatable.")
    parser.add_argument("--work-width", type=int, default=1280, help="Resize image width before processing. Lower is faster.")
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=4.0)
    parser.add_argument("--min-good-matches", type=int, default=12)
    parser.add_argument("--min-inliers", type=int, default=16)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.18)
    parser.add_argument("--blend-mode", choices=["average", "feather", "overwrite"], default="average")
    parser.add_argument("--anchor", default="middle", help="middle, first, last, or numeric image index.")
    parser.add_argument(
        "--manual-motion-model",
        choices=["translation", "similarity", "affine", "homography"],
        default="affine",
        help="Geometric model used by the manual stitcher. Affine is safer for phone panoramas; homography is more flexible but can over-warp.",
    )
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="If a link fails, stitch the longest valid contiguous segment.")
    parser.add_argument("--max-canvas-megapixels", type=float, default=24.0)
    parser.add_argument("--max-canvas-side", type=int, default=12000)
    parser.add_argument("--preprocess", choices=["none", "gray", "clahe"], default="clahe")
    parser.add_argument("--enable-gamma", action="store_true")
    parser.add_argument("--orb-fast-threshold", type=int, default=10)
    parser.add_argument("--stitcher-mode", choices=["PANORAMA", "SCANS"], default="PANORAMA")
    parser.add_argument("--image-order", choices=["meta", "name", "mtime"], default="meta")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--skip-every", type=int, default=1)
    parser.add_argument("--feature-cache-root", default="")
    parser.add_argument("--split-name", default="")
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--prefer-cache", action="store_true")
    diagnostics = parser.add_argument_group("Descriptor diagnostics and visualization")
    diagnostics.add_argument("--save-debug", action="store_true", help="Save descriptor score tables and pair visualizations.")
    diagnostics.add_argument("--save-visualizations", action="store_true", help="Save keypoint, match, inlier, and pair-warp previews for every adjacent pair.")
    diagnostics.add_argument("--save-score-table", action="store_true", help="Save CSV and PNG descriptor score tables for every adjacent pair.")
    diagnostics.add_argument("--visualization-dir", default="", help="Optional diagnostics output folder. Defaults beside the log/output.")
    diagnostics.add_argument("--diagnostic-method", action="append", dest="diagnostics_methods", help="Descriptor to visualize: ORB, AKAZE, HARRIS_HOG, SIFT, or all. Repeatable.")
    diagnostics.add_argument("--visualization-max-matches", type=int, default=80, help="Maximum matches drawn per match image.")
    diagnostics.add_argument("--visualization-max-keypoints", type=int, default=1000, help="Maximum keypoints drawn per keypoint image.")
    diagnostics.add_argument("--visualization-max-width", type=int, default=1800, help="Resize visual outputs to this max width. 0 disables resizing.")
    diagnostics.add_argument("--visualization-jpeg-quality", type=int, default=85, help="JPEG quality for visual diagnostics.")
    diagnostics.add_argument("--harris-max-corners", type=int, default=1500, help="Maximum Harris corners for HARRIS_HOG diagnostics/computation.")
    diagnostics.add_argument("--harris-quality", type=float, default=0.01, help="Harris qualityLevel for HARRIS_HOG.")
    diagnostics.add_argument("--harris-min-distance", type=float, default=8.0, help="Harris minDistance for HARRIS_HOG.")
    diagnostics.add_argument("--hog-patch-size", type=int, default=32, help="Patch size for HARRIS_HOG descriptors.")
    diagnostics.add_argument("--hog-cells", type=int, default=4, help="Cells per side for HARRIS_HOG descriptors.")
    diagnostics.add_argument("--hog-bins", type=int, default=8, help="Orientation bins for HARRIS_HOG descriptors.")
    return parser


def config_from_args(args: argparse.Namespace, provided: set[str] | None = None) -> PanoramaConfig:
    provided = provided or set()
    config = load_config_json(args.config_json) if args.config_json else PanoramaConfig()
    if "--preset" in provided or not args.config_json:
        config.preset = args.preset
        config.preset_applied = False

    config = apply_user_preset(config)
    use_cli_defaults = (not args.config_json) and config.preset == "custom"

    if should_apply_arg(provided, "--engine", use_cli_defaults):
        config.engine = args.engine
    if should_apply_arg(provided, "--profile", use_cli_defaults):
        config.profile = args.profile
    if should_apply_arg(provided, "--method", use_cli_defaults):
        config.method = normalize_method(args.method)
    if should_apply_arg(provided, "--candidate-method", use_cli_defaults):
        config.candidate_methods = [normalize_method(value) for value in (args.candidate_methods or [])]
    if should_apply_arg(provided, "--work-width", use_cli_defaults):
        config.work_width = args.work_width
    if should_apply_arg(provided, "--max-features", use_cli_defaults):
        config.max_features = args.max_features
    if should_apply_arg(provided, "--ratio-test", use_cli_defaults):
        config.ratio_test = args.ratio_test
    if should_apply_arg(provided, "--ransac-threshold", use_cli_defaults):
        config.ransac_threshold = args.ransac_threshold
    if should_apply_arg(provided, "--min-good-matches", use_cli_defaults):
        config.min_good_matches = args.min_good_matches
    if should_apply_arg(provided, "--min-inliers", use_cli_defaults):
        config.min_inliers = args.min_inliers
    if should_apply_arg(provided, "--min-inlier-ratio", use_cli_defaults):
        config.min_inlier_ratio = args.min_inlier_ratio
    if should_apply_arg(provided, "--blend-mode", use_cli_defaults):
        config.blend_mode = args.blend_mode
    if should_apply_arg(provided, "--anchor", use_cli_defaults):
        config.anchor = args.anchor
    if should_apply_arg(provided, "--manual-motion-model", use_cli_defaults):
        config.manual_motion_model = normalize_motion_model(args.manual_motion_model)
    if should_apply_arg(provided, "--no-crop", False):
        config.crop = False
    elif use_cli_defaults:
        config.crop = not args.no_crop
    if should_apply_arg(provided, "--allow-partial", False):
        config.allow_partial = True
    elif use_cli_defaults:
        config.allow_partial = args.allow_partial
    if should_apply_arg(provided, "--max-canvas-megapixels", use_cli_defaults):
        config.max_canvas_megapixels = args.max_canvas_megapixels
    if should_apply_arg(provided, "--max-canvas-side", use_cli_defaults):
        config.max_canvas_side = args.max_canvas_side
    if should_apply_arg(provided, "--preprocess", use_cli_defaults):
        config.preprocess = args.preprocess
    if should_apply_arg(provided, "--enable-gamma", False):
        config.enable_gamma = True
    elif use_cli_defaults:
        config.enable_gamma = args.enable_gamma
    if should_apply_arg(provided, "--orb-fast-threshold", use_cli_defaults):
        config.orb_fast_threshold = args.orb_fast_threshold
    if should_apply_arg(provided, "--stitcher-mode", use_cli_defaults):
        config.stitcher_mode = args.stitcher_mode
    if should_apply_arg(provided, "--image-order", use_cli_defaults):
        config.image_order = args.image_order
    if should_apply_arg(provided, "--reverse-order", False):
        config.reverse_order = True
    elif use_cli_defaults:
        config.reverse_order = args.reverse_order
    if should_apply_arg(provided, "--max-images", use_cli_defaults):
        config.max_images = args.max_images
    if should_apply_arg(provided, "--skip-every", use_cli_defaults):
        config.skip_every = args.skip_every
    if should_apply_arg(provided, "--feature-cache-root", use_cli_defaults):
        config.feature_cache_root = args.feature_cache_root
    if should_apply_arg(provided, "--split-name", use_cli_defaults):
        config.split_name = args.split_name
    if should_apply_arg(provided, "--scene-id", use_cli_defaults):
        config.scene_id = args.scene_id
    if should_apply_arg(provided, "--prefer-cache", False):
        config.prefer_cache = True
    elif use_cli_defaults:
        config.prefer_cache = args.prefer_cache
    if should_apply_arg(provided, "--save-debug", False):
        config.save_debug = True
    elif use_cli_defaults:
        config.save_debug = args.save_debug
    if should_apply_arg(provided, "--save-visualizations", False):
        config.save_pair_visualizations = True
    elif use_cli_defaults:
        config.save_pair_visualizations = args.save_visualizations
    if should_apply_arg(provided, "--save-score-table", False):
        config.save_score_table = True
    elif use_cli_defaults:
        config.save_score_table = args.save_score_table
    if should_apply_arg(provided, "--visualization-dir", use_cli_defaults):
        config.visualization_dir = args.visualization_dir
    if should_apply_arg(provided, "--diagnostic-method", use_cli_defaults):
        config.diagnostics_methods = [value for value in (args.diagnostics_methods or [])]
    if should_apply_arg(provided, "--visualization-max-matches", use_cli_defaults):
        config.visualization_max_matches = args.visualization_max_matches
    if should_apply_arg(provided, "--visualization-max-keypoints", use_cli_defaults):
        config.visualization_max_keypoints = args.visualization_max_keypoints
    if should_apply_arg(provided, "--visualization-max-width", use_cli_defaults):
        config.visualization_max_width = args.visualization_max_width
    if should_apply_arg(provided, "--visualization-jpeg-quality", use_cli_defaults):
        config.visualization_jpeg_quality = args.visualization_jpeg_quality
    if should_apply_arg(provided, "--harris-max-corners", use_cli_defaults):
        config.harris_max_corners = args.harris_max_corners
    if should_apply_arg(provided, "--harris-quality", use_cli_defaults):
        config.harris_quality = args.harris_quality
    if should_apply_arg(provided, "--harris-min-distance", use_cli_defaults):
        config.harris_min_distance = args.harris_min_distance
    if should_apply_arg(provided, "--hog-patch-size", use_cli_defaults):
        config.hog_patch_size = args.hog_patch_size
    if should_apply_arg(provided, "--hog-cells", use_cli_defaults):
        config.hog_cells = args.hog_cells
    if should_apply_arg(provided, "--hog-bins", use_cli_defaults):
        config.hog_bins = args.hog_bins

    return apply_profile(config)


def run_interactive_wizard() -> tuple[Path, Path, PanoramaConfig, Path | None]:
    print("=" * 60)
    print("      PORTABLE PANORAMA STITCHING PIPELINE WIZARD")
    print("=" * 60)
    print("This wizard will guide you through stitching your images.")
    print()

    # 1. Scene Folder Selection
    base_split = Path("data/split")
    candidate_folders = []
    if base_split.exists():
        for sub in base_split.iterdir():
            if sub.is_dir():
                for scene in sub.iterdir():
                    if scene.is_dir():
                        try:
                            imgs = [p for p in scene.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
                            if len(imgs) >= 2:
                                candidate_folders.append((scene, sub.name))
                        except Exception:
                            pass

    scene_folder = None
    if candidate_folders:
        candidate_folders.sort(key=lambda x: (x[1], x[0].name))
        print("Available scene folders found in project:")
        for idx, (folder, split_name) in enumerate(candidate_folders):
            image_count = len([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS])
            print(f"  [{idx + 1:2d}] {split_name}/{folder.name} ({image_count} images)")
        print(f"  [ C] Enter custom folder path")
        print()
        
        while True:
            choice = input(f"Select a scene folder [1-{len(candidate_folders)} or C]: ").strip()
            if choice.upper() == 'C':
                break
            try:
                val = int(choice)
                if 1 <= val <= len(candidate_folders):
                    scene_folder = candidate_folders[val - 1][0]
                    break
            except ValueError:
                pass
            print("Invalid selection, please try again.")
    
    if scene_folder is None:
        while True:
            custom_path = input("Enter the path to your scene folder: ").strip()
            if not custom_path:
                print("Folder path cannot be empty.")
                continue
            path = Path(custom_path)
            if not path.exists():
                print(f"Path '{custom_path}' does not exist. Please enter a valid path.")
                continue
            if not path.is_dir():
                print(f"Path '{custom_path}' is not a directory.")
                continue
            try:
                imgs = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
                if len(imgs) < 2:
                    print(f"Directory contains only {len(imgs)} images. Need at least 2 images to stitch.")
                    continue
                scene_folder = path
                break
            except Exception as e:
                print(f"Error accessing directory: {e}")
                continue

    scene_name = scene_folder.name
    
    # 2. Output Path Selection
    default_output = Path("outputs") / f"{scene_name}_panorama.jpg"
    print(f"\nDefault output path: {default_output}")
    out_input = input("Enter output image path (or press Enter to use default): ").strip()
    if out_input:
        output_path = Path(out_input)
    else:
        output_path = default_output

    # 3. Preset Selection
    print("\nSelect a processing preset (controls quality, speed, and CPU/memory usage):")
    print(f"  [1] Normal Phone  - {PRESET_DESCRIPTIONS['normal_phone']} (Recommended)")
    print(f"  [2] Weak Phone    - {PRESET_DESCRIPTIONS['weak_phone']}")
    print(f"  [3] Best Quality  - {PRESET_DESCRIPTIONS['best_quality']}")
    print(f"  [4] Student Debug - {PRESET_DESCRIPTIONS['student_debug']}")
    print(f"  [5] Custom        - {PRESET_DESCRIPTIONS['custom']}")
    
    preset_map = {
        "1": "normal_phone",
        "2": "weak_phone",
        "3": "best_quality",
        "4": "student_debug",
        "5": "custom"
    }
    
    preset_choice = "1"
    while True:
        p_val = input("Select preset [1-5, default=1]: ").strip()
        if not p_val:
            break
        if p_val in preset_map:
            preset_choice = p_val
            break
        print("Invalid selection, please select a number from 1 to 5.")
        
    selected_preset = preset_map[preset_choice]
    
    config = PanoramaConfig(preset=selected_preset)
    config = apply_user_preset(config)
    
    # 4. Optional Custom Overrides
    if selected_preset == "custom":
        print("\n--- Custom Settings ---")
        
        # Engine
        engine_choice = input("Stitching engine (manual or opencv) [default=manual]: ").strip().lower()
        if engine_choice in ["manual", "opencv"]:
            config.engine = engine_choice
            
        # Method
        method_choice = input("Feature descriptor (auto, ORB, AKAZE, SIFT, HARRIS_HOG) [default=auto]: ").strip()
        if method_choice:
            try:
                config.method = normalize_method(method_choice)
            except ValueError as e:
                print(f"Invalid method, using auto. Error: {e}")
                
        # Work width
        width_choice = input("Processing width (lower is faster, e.g., 800, 1280, 1800) [default=1280]: ").strip()
        if width_choice:
            try:
                config.work_width = int(width_choice)
            except ValueError:
                pass
                
        # Allow partial
        partial_choice = input("Allow partial stitching if a link fails? (y/n) [default=n]: ").strip().lower()
        if partial_choice in ["y", "yes"]:
            config.allow_partial = True
        elif partial_choice in ["n", "no"]:
            config.allow_partial = False

        # Visualizations
        vis_choice = input("Save pair-wise match/keypoint visualizations? (y/n) [default=n]: ").strip().lower()
        if vis_choice in ["y", "yes"]:
            config.save_pair_visualizations = True
            config.save_debug = True
            
    else:
        print(f"\nUsing preset '{selected_preset}'.")
        quick_override = input("Would you like to customize any settings (e.g. save visualizations, enable auto-crop)? (y/N): ").strip().lower()
        if quick_override in ["y", "yes"]:
            crop_choice = input(f"Auto-crop black borders? (y/n) [default={'y' if config.crop else 'n'}]: ").strip().lower()
            if crop_choice in ["y", "yes"]:
                config.crop = True
            elif crop_choice in ["n", "no"]:
                config.crop = False
                
            partial_choice = input(f"Allow partial stitching if alignment fails? (y/n) [default={'y' if config.allow_partial else 'n'}]: ").strip().lower()
            if partial_choice in ["y", "yes"]:
                config.allow_partial = True
            elif partial_choice in ["n", "no"]:
                config.allow_partial = False
                
            vis_choice = input(f"Save step-by-step debug/matching images? (y/n) [default={'y' if config.save_pair_visualizations else 'n'}]: ").strip().lower()
            if vis_choice in ["y", "yes"]:
                config.save_pair_visualizations = True
                config.save_debug = True
            elif vis_choice in ["n", "no"]:
                config.save_pair_visualizations = False
                config.save_debug = False

    # Optional Log Path
    log_input = input("\nEnter log output file path (optional, press Enter to skip): ").strip()
    log_path = Path(log_input) if log_input else None

    # Print summary
    print("\n" + "=" * 40)
    print("SUMMARY OF CONFIGURATION")
    print("=" * 40)
    print(f"Scene Folder:     {scene_folder}")
    print(f"Output Path:      {output_path}")
    print(f"Preset:           {config.preset}")
    print(f"Engine:           {config.engine}")
    print(f"Method:           {config.method}")
    print(f"Work Width:       {config.work_width} px")
    print(f"Auto-Crop:        {config.crop}")
    print(f"Allow Partial:    {config.allow_partial}")
    print(f"Save Diagnostics: {config.save_pair_visualizations}")
    if log_path:
        print(f"Log Path:         {log_path}")
    print("=" * 40)
    
    confirm = input("Start panorama stitching? [Y/n]: ").strip().lower()
    if confirm in ["n", "no"]:
        print("Operation cancelled by user.")
        sys.exit(0)
        
    return scene_folder, output_path, config, log_path


def main() -> None:
    args = build_parser().parse_args()
    provided = provided_cli_options(sys.argv[1:])

    if args.print_presets:
        print_presets()
        return

    if args.write_config_template is not None:
        template_preset = args.preset if "--preset" in provided else "normal_phone"
        write_config_template(args.write_config_template, template_preset)
        print(json.dumps({"status": "ok", "config_template": str(args.write_config_template), "preset": template_preset}, indent=2))
        return

    config: PanoramaConfig
    log = args.log
    try:
        if args.scene_folder is None or args.output is None:
            scene_folder, output, config, log = run_interactive_wizard()
            args.scene_folder = scene_folder
            args.output = output
            args.log = log
        else:
            config = config_from_args(args, provided)
            log = args.log

        payload = stitch_scene_folder(args.scene_folder, args.output, config, log)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "engine": payload.get("engine"),
                    "output_path": payload.get("output_path"),
                    "is_partial": payload.get("is_partial", False),
                    "quality_warnings": payload.get("quality_warnings", []),
                    "diagnostics": payload.get("diagnostics", {}),
                    "runtime_sec": payload["runtime_sec"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    except (KeyboardInterrupt, EOFError):
        payload = {
            "status": "error",
            "error_type": "interrupted",
            "message": "Operation interrupted by user.",
            "user_message": "Panorama stitching was cancelled before completion.",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        sys.exit(130)
    except PanoramaPipelineError as exc:
        payload = exc.payload or {
            "status": "error",
            "error_type": "panorama_pipeline_error",
            "message": str(exc),
            "user_message": "The panorama could not be created with the selected settings.",
        }
        if args.show_traceback:
            payload["traceback"] = traceback.format_exc()
        write_json_log(log, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        sys.exit(2)
    except Exception as exc:
        payload = {
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "user_message": "The panorama could not be created. Check image order, overlap, permissions, and the selected preset.",
        }
        if args.scene_folder is not None:
            payload["scene_folder"] = str(args.scene_folder)
        if args.output is not None:
            payload["output_path"] = str(args.output)
        if args.show_traceback:
            payload["traceback"] = traceback.format_exc()
        write_json_log(log, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()

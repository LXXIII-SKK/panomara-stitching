from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cv2.ocl.setUseOpenCL(False)

from project_utils.panorama_dataset import list_scene_dirs, ordered_scene_files
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
ORB_NFEATURES = 4000
RATIO_TEST = 0.75
RANSAC_REPROJ_THRESHOLD = 4.0
STITCH_MAX_INPUT_WIDTH = 3000
DEFAULT_STABILITY_RUNS = 10
STABILITY_SUCCESS_RATE = 0.9
STABILITY_FAILURE_RATE = 0.2
STITCHER_EXCEPTION_CODE = -999
PANORAMA_SHAPE_BUCKET_SIZE = 100

STATUS_NAMES = {
    int(cv2.Stitcher_OK): "OK",
    int(cv2.Stitcher_ERR_NEED_MORE_IMGS): "ERR_NEED_MORE_IMGS",
    int(cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL): "ERR_HOMOGRAPHY_EST_FAIL",
    int(cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL): "ERR_CAMERA_PARAMS_ADJUST_FAIL",
    STITCHER_EXCEPTION_CODE: "EXCEPTION",
}

MANUAL_FIELDS = [
    "type",
    "capture_group",
    "category",
    "difficulty",
    "recommended_use",
    "issues",
    "has_moving_objects",
    "has_repeated_patterns",
    "has_low_texture",
    "has_parallax",
    "has_exposure_change",
    "has_motion_blur",
    "has_insufficient_overlap",
    "notes",
]

DEFAULT_META = {
    "type": "unknown",
    "capture_group": "core",
    "category": "pending_review",
    "difficulty": "unknown",
    "recommended_use": "manual_review",
    "issues": [],
    "has_moving_objects": None,
    "has_repeated_patterns": None,
    "has_low_texture": None,
    "has_parallax": None,
    "has_exposure_change": None,
    "has_motion_blur": None,
    "has_insufficient_overlap": None,
    "notes": "",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate scene meta.json files from the current dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DATA_ROOT,
        help="Root directory containing scene folders.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Specific scene_id to regenerate. Repeat the flag to target multiple scenes.",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=DEFAULT_STABILITY_RUNS,
        help="How many repeated cv2.Stitcher runs to use for stability_check. Use 0 to skip.",
    )
    return parser


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_keep_aspect(image: np.ndarray, max_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def entropy_score(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def parse_capture_time(path: Path) -> datetime | None:
    try:
        exif = Image.open(path).getexif()
    except Exception:
        return None

    for tag_id in (36867, 36868, 306):
        value = exif.get(tag_id)
        if not value:
            continue
        try:
            return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
    return None


def compute_capture_stats(files: list[Path], previous_meta: dict) -> tuple[int | None, int | None]:
    timestamps = [parse_capture_time(path) for path in files]
    if all(ts is not None for ts in timestamps) and timestamps:
        ordered = sorted(timestamps)
        span = int((ordered[-1] - ordered[0]).total_seconds()) if len(ordered) >= 2 else 0
        gaps = [
            int((current - previous).total_seconds())
            for previous, current in zip(ordered[:-1], ordered[1:])
        ]
        return span, (max(gaps) if gaps else 0)

    return previous_meta.get("capture_span_seconds"), previous_meta.get("max_capture_gap_seconds")


def detect_and_describe(gray: np.ndarray):
    detector = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if keypoints is None:
        keypoints = []
    return keypoints, descriptors


def knn_ratio_match(descriptors_a, descriptors_b):
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(descriptors_a) == 0
        or len(descriptors_b) == 0
    ):
        return [], []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        match_a, match_b = pair
        if match_a.distance < RATIO_TEST * match_b.distance:
            good_matches.append(match_a)

    return knn_matches, good_matches


def keypoints_to_xy(keypoints, matches, query: bool) -> np.ndarray:
    indices = [match.queryIdx if query else match.trainIdx for match in matches]
    points = [keypoints[index].pt for index in indices]
    return np.float32(points).reshape(-1, 1, 2)


def estimate_homography(keypoints_a, keypoints_b, good_matches):
    if len(good_matches) < 4:
        return None, None

    src = keypoints_to_xy(keypoints_a, good_matches, query=True)
    dst = keypoints_to_xy(keypoints_b, good_matches, query=False)
    return cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)


def median_reprojection_error(keypoints_a, keypoints_b, matches, homography, mask) -> float:
    if homography is None or mask is None:
        return 0.0

    keep_mask = mask.ravel().astype(bool)
    inlier_matches = [match for match, keep in zip(matches, keep_mask) if keep]
    if not inlier_matches:
        return 0.0

    src = keypoints_to_xy(keypoints_a, inlier_matches, query=True)
    dst = keypoints_to_xy(keypoints_b, inlier_matches, query=False)
    projected = cv2.perspectiveTransform(src, homography)
    errors = np.linalg.norm(projected - dst, axis=2).reshape(-1)
    return float(np.median(errors)) if len(errors) else 0.0


def classify_pair(good_matches: int, inliers: int, inlier_ratio: float, homography_ok: bool) -> str:
    if not homography_ok or inliers < 10:
        return "fail"
    if good_matches >= 120 and inliers >= 60 and inlier_ratio >= 0.5:
        return "strong"
    if good_matches >= 40 and inliers >= 20 and inlier_ratio >= 0.25:
        return "ok"
    return "weak"


def image_metrics(path: Path) -> tuple[dict, np.ndarray, list, np.ndarray | None]:
    bgr = load_bgr(path)
    gray = to_gray(bgr)
    keypoints, descriptors = detect_and_describe(gray)
    metrics = {
        "file": path.name,
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "blur_score": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "brightness_mean": round(float(gray.mean()), 2),
        "contrast_std": round(float(gray.std()), 2),
        "entropy": round(entropy_score(gray), 3),
        "keypoints": int(len(keypoints)),
    }
    return metrics, gray, keypoints, descriptors


def pair_metrics(file_a: Path, file_b: Path, gray_a, gray_b, keypoints_a, keypoints_b, descriptors_a, descriptors_b, pair_index: int) -> dict:
    knn_matches, good_matches = knn_ratio_match(descriptors_a, descriptors_b)
    homography, mask = estimate_homography(keypoints_a, keypoints_b, good_matches)
    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = float(inliers / len(good_matches)) if good_matches else 0.0
    homography_ok = homography is not None and inliers >= 4
    return {
        "pair_index": pair_index,
        "image_a": file_a.name,
        "image_b": file_b.name,
        "raw_matches": int(len(knn_matches)),
        "good_matches": int(len(good_matches)),
        "inliers": int(inliers),
        "inlier_ratio": round(inlier_ratio, 3),
        "homography_ok": bool(homography_ok),
        "median_reproj_error": round(
            median_reprojection_error(keypoints_a, keypoints_b, good_matches, homography, mask),
            3,
        ),
        "pair_label": classify_pair(len(good_matches), inliers, inlier_ratio, homography_ok),
    }


def stitch_once(images: list[np.ndarray]) -> tuple[int, str, dict | None, str | None]:
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    try:
        status_code, panorama = stitcher.stitch(images)
        status_code = int(status_code)
        panorama_shape = None
        if status_code == int(cv2.Stitcher_OK) and panorama is not None:
            panorama_shape = {
                "width": int(panorama.shape[1]),
                "height": int(panorama.shape[0]),
            }
        return status_code, STATUS_NAMES.get(status_code, f"CODE_{status_code}"), panorama_shape, None
    except cv2.error as exc:
        error_message = " ".join(str(exc).split())
        return STITCHER_EXCEPTION_CODE, STATUS_NAMES[STITCHER_EXCEPTION_CODE], None, error_message


def classify_stability(
    ok_rate: float,
    dominant_status: str,
    dominant_rate: float,
    distinct_statuses: int,
    dominant_shape_rate: float | None,
    output_consistent: bool | None,
) -> str:
    if distinct_statuses == 1 and dominant_status == "OK" and output_consistent is not False:
        return "stable_success"
    if distinct_statuses == 1 and dominant_status != "OK":
        return "stable_failure"
    if ok_rate >= STABILITY_SUCCESS_RATE and output_consistent is False:
        return "success_with_output_variation"
    if ok_rate >= STABILITY_SUCCESS_RATE:
        return "borderline_success"
    if ok_rate <= STABILITY_FAILURE_RATE and dominant_status != "OK" and dominant_rate >= STABILITY_SUCCESS_RATE:
        return "borderline_failure"
    return "unstable_mix"


def stitcher_stability_check(images: list[np.ndarray], runs: int) -> dict | None:
    if runs <= 0:
        return None

    status_counter: Counter[str] = Counter()
    shape_counter: Counter[tuple[int, int]] = Counter()
    shape_bucket_counter: Counter[tuple[int, int]] = Counter()
    error_counter: Counter[str] = Counter()
    for _ in range(runs):
        _, status_name, panorama_shape, error_message = stitch_once(images)
        status_counter.update([status_name])
        if panorama_shape is not None:
            exact_shape = (panorama_shape["width"], panorama_shape["height"])
            shape_counter.update([exact_shape])
            bucket_shape = (
                int(round(panorama_shape["width"] / PANORAMA_SHAPE_BUCKET_SIZE) * PANORAMA_SHAPE_BUCKET_SIZE),
                int(round(panorama_shape["height"] / PANORAMA_SHAPE_BUCKET_SIZE) * PANORAMA_SHAPE_BUCKET_SIZE),
            )
            shape_bucket_counter.update([bucket_shape])
        if error_message:
            error_counter.update([error_message])

    dominant_status, dominant_count = status_counter.most_common(1)[0]
    ok_runs = int(status_counter.get("OK", 0))
    ok_rate = float(ok_runs / runs)
    dominant_rate = float(dominant_count / runs)
    shape_samples = [
        {"width": width, "height": height, "count": int(count)}
        for (width, height), count in shape_counter.most_common()
    ]
    dominant_shape_rate = None
    dominant_shape = None
    output_consistent = None
    bucket_samples = [
        {"width": width, "height": height, "count": int(count)}
        for (width, height), count in shape_bucket_counter.most_common()
    ]
    dominant_bucket_shape = None
    dominant_bucket_shape_rate = None
    if shape_counter:
        (dominant_width, dominant_height), dominant_shape_count = shape_counter.most_common(1)[0]
        dominant_shape = {"width": int(dominant_width), "height": int(dominant_height)}
        dominant_shape_rate = float(dominant_shape_count / max(ok_runs, 1))
    if shape_bucket_counter:
        (bucket_width, bucket_height), bucket_count = shape_bucket_counter.most_common(1)[0]
        dominant_bucket_shape = {"width": int(bucket_width), "height": int(bucket_height)}
        dominant_bucket_shape_rate = float(bucket_count / max(ok_runs, 1))
        output_consistent = dominant_bucket_shape_rate >= STABILITY_SUCCESS_RATE

    return {
        "runs": int(runs),
        "status_counts": dict(sorted(status_counter.items())),
        "ok_runs": int(ok_runs),
        "ok_rate": round(ok_rate, 3),
        "dominant_status": dominant_status,
        "dominant_rate": round(dominant_rate, 3),
        "is_consistent": len(status_counter) == 1,
        "ok_panorama_shape_counts": shape_samples,
        "dominant_ok_panorama_shape": dominant_shape,
        "dominant_ok_panorama_shape_rate": None if dominant_shape_rate is None else round(dominant_shape_rate, 3),
        "ok_panorama_shape_bucket_size": PANORAMA_SHAPE_BUCKET_SIZE,
        "ok_panorama_shape_bucket_counts": bucket_samples,
        "dominant_ok_panorama_shape_bucket": dominant_bucket_shape,
        "dominant_ok_panorama_shape_bucket_rate": None if dominant_bucket_shape_rate is None else round(dominant_bucket_shape_rate, 3),
        "is_output_consistent": output_consistent,
        "error_counts": dict(error_counter),
        "is_stable": bool(
            dominant_rate >= STABILITY_SUCCESS_RATE
            and (
                dominant_status != "OK"
                or dominant_bucket_shape_rate is None
                or dominant_bucket_shape_rate >= STABILITY_SUCCESS_RATE
            )
        ),
        "stability_label": classify_stability(
            ok_rate,
            dominant_status,
            dominant_rate,
            len(status_counter),
            dominant_bucket_shape_rate,
            output_consistent,
        ),
    }


def regenerate_scene_meta(scene_dir: Path, stability_runs: int = DEFAULT_STABILITY_RUNS) -> dict:
    ordered_files, reference_files, previous_meta, _ = ordered_scene_files(scene_dir)
    if len(ordered_files) < 2:
        raise ValueError(f"{scene_dir.name}: need at least 2 ordered images")

    image_rows = []
    grays = []
    keypoints_list = []
    descriptors_list = []

    for path in ordered_files:
        metrics, gray, keypoints, descriptors = image_metrics(path)
        image_rows.append(metrics)
        grays.append(gray)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    pair_rows = []
    for pair_index, (file_a, file_b) in enumerate(zip(ordered_files[:-1], ordered_files[1:]), start=1):
        pair_rows.append(
            pair_metrics(
                file_a,
                file_b,
                grays[pair_index - 1],
                grays[pair_index],
                keypoints_list[pair_index - 1],
                keypoints_list[pair_index],
                descriptors_list[pair_index - 1],
                descriptors_list[pair_index],
                pair_index,
            )
        )

    stitcher_images = [resize_keep_aspect(load_bgr(path), STITCH_MAX_INPUT_WIDTH) for path in ordered_files]
    stitch_code, stitch_name, stitch_output, stitch_error = stitch_once(stitcher_images)
    stability_check = stitcher_stability_check(stitcher_images, stability_runs)
    pair_counter = Counter(row["pair_label"] for row in pair_rows)
    capture_span_seconds, max_capture_gap_seconds = compute_capture_stats(ordered_files, previous_meta)

    refreshed_meta = {"scene_id": scene_dir.name}
    for field in MANUAL_FIELDS:
        refreshed_meta[field] = previous_meta.get(field, DEFAULT_META[field])

    refreshed_meta.update(
        {
            "num_images": len(ordered_files),
            "ordered_files": [path.name for path in ordered_files],
            "reference_files": [path.name for path in reference_files],
            "capture_span_seconds": capture_span_seconds,
            "max_capture_gap_seconds": max_capture_gap_seconds,
            "audit_summary": {
                "stitcher_status_code": stitch_code,
                "stitcher_status": stitch_name,
                "stitcher_output": stitch_output,
                "stitcher_error": stitch_error,
                "avg_keypoints": round(float(np.mean([row["keypoints"] for row in image_rows])), 1),
                "min_keypoints": int(min(row["keypoints"] for row in image_rows)),
                "avg_blur_score": round(float(np.mean([row["blur_score"] for row in image_rows])), 2),
                "min_blur_score": round(float(min(row["blur_score"] for row in image_rows)), 2),
                "brightness_span": round(
                    float(max(row["brightness_mean"] for row in image_rows) - min(row["brightness_mean"] for row in image_rows)),
                    2,
                ),
                "avg_entropy": round(float(np.mean([row["entropy"] for row in image_rows])), 3),
                "pair_label_counts": {
                    "strong": int(pair_counter.get("strong", 0)),
                    "ok": int(pair_counter.get("ok", 0)),
                    "weak": int(pair_counter.get("weak", 0)),
                    "fail": int(pair_counter.get("fail", 0)),
                },
                "stability_check": stability_check,
            },
            "image_stats": image_rows,
            "pair_audit": pair_rows,
        }
    )

    return refreshed_meta


def main():
    args = build_parser().parse_args()
    root = args.root.resolve()
    target_scenes = set(args.scenes or [])

    updated = []
    for scene_dir in list_scene_dirs(root):
        if target_scenes and scene_dir.name not in target_scenes:
            continue
        meta = regenerate_scene_meta(scene_dir, stability_runs=args.stability_runs)
        meta_path = scene_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        updated.append(scene_dir.name)
        print(f"[OK] updated {scene_dir.name}")

    print(f"\nUpdated {len(updated)} scene metadata files.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_utils.panorama_dataset import list_scene_dirs, ordered_scene_files

OUTPUT_OVERLAY_DIR = PROJECT_ROOT / "outputs" / "openCV" / "overlays"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "openCV" / "logs"
OUTPUT_PANORAMA_DIR = PROJECT_ROOT / "outputs" / "openCV" / "panoramas"
SPLIT_ROOT = PROJECT_ROOT / "data" / "split"
SPLITS_TO_SEARCH = ["test", "failure_analysis", "development"]

OUTPUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def resolve_scene_dir(scene_id: str, split_name: str | None = None) -> Path:
    if split_name:
        scene_dir = SPLIT_ROOT / split_name / scene_id
        if scene_dir.exists():
            return scene_dir
        raise FileNotFoundError(f"Scene not found: {scene_dir}")

    matches = []
    for split_dir in list_scene_dirs(SPLIT_ROOT):
        if split_dir.name not in SPLITS_TO_SEARCH:
            continue
        scene_dir = split_dir / scene_id
        if scene_dir.exists():
            matches.append(scene_dir)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        found = ", ".join(str(path.relative_to(SPLIT_ROOT)) for path in matches)
        raise ValueError(f"Scene {scene_id} appears in multiple splits ({found}); pass split_name.")
    raise FileNotFoundError(f"Scene {scene_id} was not found under {SPLIT_ROOT}")

PALETTE = [
    (235, 99, 71),
    (60, 180, 75),
    (66, 133, 244),
    (255, 193, 7),
    (171, 71, 188),
    (0, 188, 212),
    (255, 112, 67),
    (124, 179, 66),
]


def resize_keep_aspect(image: np.ndarray, max_width: int = 3000) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def gray_clahe(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detector_for(method: str):
    if method == "orb":
        return cv2.ORB_create(nfeatures=6000, fastThreshold=10), cv2.NORM_HAMMING
    if method == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=4000), cv2.NORM_L2
    return None, None


def detect_and_describe(gray: np.ndarray, method: str):
    detector, norm = detector_for(method)
    if detector is None:
        return [], None, None
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if keypoints is None:
        keypoints = []
    return keypoints, descriptors, norm


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return float(abs(cv2.contourArea(points.reshape(-1, 1, 2).astype(np.float32))))


def reasonable_projected_polygon(points: np.ndarray, pano_w: int, pano_h: int) -> bool:
    if len(points) != 4 or not np.isfinite(points).all():
        return False
    area = polygon_area(points)
    if area <= 0:
        return False
    if area > pano_w * pano_h * 2.0:
        return False
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    margin_x = pano_w * 0.75
    margin_y = pano_h * 0.75
    return (
        min_x >= -margin_x
        and min_y >= -margin_y
        and max_x <= pano_w + margin_x
        and max_y <= pano_h + margin_y
    )


def convex_hull_from_points(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.float32))
    return hull.reshape(-1, 2)


def choose_best_localization(image_bgr: np.ndarray, panorama_bgr: np.ndarray) -> dict:
    pano_h, pano_w = panorama_bgr.shape[:2]
    gray_image = gray_clahe(image_bgr)
    gray_pano = gray_clahe(panorama_bgr)

    candidates = []
    for method in ["orb", "sift"]:
        keypoints_img, descriptors_img, norm = detect_and_describe(gray_image, method)
        keypoints_pano, descriptors_pano, _ = detect_and_describe(gray_pano, method)
        if norm is None or descriptors_img is None or descriptors_pano is None:
            continue

        matcher = cv2.BFMatcher(norm)
        raw_matches = matcher.knnMatch(descriptors_img, descriptors_pano, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

        record = {
            "method": f"{method}_clahe",
            "raw_matches": int(len(raw_matches)),
            "good_matches": int(len(good_matches)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "homography_ok": False,
            "confidence": "fail",
            "projected_corners": None,
            "projected_corners_ok": False,
            "projected_area_ratio": None,
            "match_hull": None,
            "match_hull_area_ratio": None,
        }

        if len(good_matches) < 8:
            candidates.append(record)
            continue

        src = np.float32([keypoints_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst = np.float32([keypoints_pano[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)

        if mask is not None:
            inliers = int(mask.ravel().sum())
            record["inliers"] = inliers
            record["inlier_ratio"] = round(float(inliers / max(len(good_matches), 1)), 3)
            record["homography_ok"] = bool(homography is not None and inliers >= 8)

            inlier_points_pano = dst[mask.ravel().astype(bool)].reshape(-1, 2)
            hull = convex_hull_from_points(inlier_points_pano)
            if hull is not None:
                record["match_hull"] = [[round(float(x), 1), round(float(y), 1)] for x, y in hull]
                record["match_hull_area_ratio"] = round(
                    polygon_area(hull) / max(float(pano_w * pano_h), 1.0),
                    4,
                )

            if homography is not None:
                img_h, img_w = image_bgr.shape[:2]
                corners = np.float32(
                    [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
                ).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
                if reasonable_projected_polygon(projected, pano_w, pano_h):
                    record["projected_area_ratio"] = round(
                        polygon_area(projected) / max(float(pano_w * pano_h), 1.0),
                        4,
                    )
                    record["projected_corners"] = [
                        [round(float(x), 1), round(float(y), 1)] for x, y in projected
                    ]
                    record["projected_corners_ok"] = True

        if record["inliers"] >= 20:
            record["confidence"] = "high"
        elif record["inliers"] >= 10:
            record["confidence"] = "medium"
        elif record["inliers"] >= 8:
            record["confidence"] = "low"

        candidates.append(record)

    if not candidates:
        return {
            "method": None,
            "raw_matches": 0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "homography_ok": False,
            "confidence": "fail",
            "projected_corners": None,
            "projected_corners_ok": False,
            "projected_area_ratio": None,
            "match_hull": None,
            "match_hull_area_ratio": None,
        }

    return max(
        candidates,
        key=lambda record: (
            record["inliers"],
            record["good_matches"],
            record["method"] == "sift_clahe",
        ),
    )


def label_anchor(points: np.ndarray) -> tuple[int, int]:
    center = points.mean(axis=0)
    return int(round(center[0])), int(round(center[1]))


def draw_overlay(panorama_bgr: np.ndarray, per_image_records: list[dict]) -> np.ndarray:
    overlay = panorama_bgr.copy()
    output = panorama_bgr.copy()

    for index, record in enumerate(per_image_records):
        color = PALETTE[index % len(PALETTE)]
        hull = record.get("match_hull")
        polygon = None
        if hull:
            polygon = np.array(hull, dtype=np.int32)
            if len(polygon) >= 3 and record.get("match_hull_area_ratio", 0) >= 0.0005:
                cv2.fillPoly(overlay, [polygon], color)
                cv2.polylines(output, [polygon], isClosed=True, color=color, thickness=5)
            else:
                center_x, center_y = label_anchor(polygon.astype(np.float32))
                cv2.circle(output, (center_x, center_y), 14, color, -1)

        if polygon is not None:
            x, y = label_anchor(polygon.astype(np.float32))
            label = record["file"]
            if record["confidence"] == "low":
                label += " (low)"
            cv2.putText(
                output,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )

    blended = cv2.addWeighted(overlay, 0.20, output, 0.80, 0)
    return blended


def generate_overlay(scene_id: str, split_name: str | None = None) -> dict:
    scene_dir = resolve_scene_dir(scene_id, split_name=split_name)
    ordered_files, reference_files, meta, used_meta_order = ordered_scene_files(scene_dir)
    panorama_path = OUTPUT_PANORAMA_DIR / f"{scene_id}_opencv_panorama.jpg"
    if not panorama_path.exists():
        raise FileNotFoundError(
            f"Panorama image not found: {panorama_path}. Run the OpenCV stitcher first."
        )

    panorama_bgr = load_bgr(panorama_path)
    panorama_h, panorama_w = panorama_bgr.shape[:2]

    per_image_records = []
    for path in ordered_files:
        image_bgr = resize_keep_aspect(load_bgr(path), max_width=3000)
        best = choose_best_localization(image_bgr, panorama_bgr)
        best.update(
            {
                "file": path.name,
                "width": int(image_bgr.shape[1]),
                "height": int(image_bgr.shape[0]),
            }
        )
        per_image_records.append(best)

    overlay_bgr = draw_overlay(panorama_bgr, per_image_records)
    overlay_path = OUTPUT_OVERLAY_DIR / f"{scene_id}_opencv_input_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay_bgr)

    summary = {
        "scene_id": scene_id,
        "split": scene_dir.parent.name,
        "scene_dir": str(scene_dir),
        "panorama_path": str(panorama_path),
        "overlay_path": str(overlay_path),
        "panorama_shape": {"width": int(panorama_w), "height": int(panorama_h)},
        "ordered_files": [path.name for path in ordered_files],
        "reference_files": [path.name for path in reference_files],
        "used_meta_order": used_meta_order,
        "meta_category": meta.get("category"),
        "per_image_overlay": per_image_records,
    }

    log_path = OUTPUT_LOG_DIR / f"{scene_id}_opencv_input_overlay.json"
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["log_path"] = str(log_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate source-image overlay on top of an OpenCV panorama.")
    parser.add_argument("scene_id", help="Scene folder name, for example scene_16")
    parser.add_argument("--split", choices=SPLITS_TO_SEARCH, help="Optional split name if scene IDs are not unique.")
    args = parser.parse_args()

    summary = generate_overlay(args.scene_id, split_name=args.split)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

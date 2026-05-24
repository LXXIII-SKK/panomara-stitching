from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


BRIGHTNESS_RECOMMENDATIONS = {
    "gamma_or_brightness_lift",
    "brightness_normalization_trial",
    "exposure_normalization_trial",
    "highlight_clipping_review",
}
CLAHE_RECOMMENDATIONS = {
    "clahe_or_local_contrast_boost",
    "apply_clahe_trial",
}
SHARPEN_RECOMMENDATIONS = {"mild_sharpen_trial"}
DROP_RECOMMENDATIONS = {
    "retake_or_drop_blurry_image",
    "drop_or_retake_blurry_frames",
}


@dataclass(frozen=True)
class PreprocessConfig:
    max_width: int = 1600
    gaussian_kernel: int = 3
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    enable_clahe: bool = True
    enable_brightness_normalization: bool = True
    enable_denoise: bool = False
    denoise_strength: int = 7
    enable_unsharp: bool = False
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 0.8
    target_brightness: float = 128.0
    gamma_min: float = 0.7
    gamma_max: float = 1.5


def parse_recommendation_text(value: str | None) -> set[str]:
    if not value:
        return set()
    return {token.strip() for token in str(value).split(",") if token.strip()}


def load_audit_image_recommendations(csv_path: Path) -> dict[Path, set[str]]:
    if not csv_path.exists():
        return {}

    recommendations: dict[Path, set[str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path_text = row.get("path")
            if not path_text:
                continue
            path = Path(path_text).resolve()
            recommendations[path] = parse_recommendation_text(row.get("preprocess_recommendations"))
    return recommendations


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
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def compute_gray_metrics(gray: np.ndarray) -> dict[str, float]:
    return {
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def adjust_gamma(gray_img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction using gamma < 1 to brighten and gamma > 1 to darken."""
    gamma = max(float(gamma), 1e-4)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray_img, lut)


def estimate_auto_gamma(
    gray_img: np.ndarray,
    target_mean: float = 128.0,
    gamma_min: float = 0.7,
    gamma_max: float = 1.5,
) -> float:
    current_mean = float(gray_img.mean())
    if current_mean <= 1.0 or current_mean >= 254.0:
        return 1.0

    current_norm = np.clip(current_mean / 255.0, 1e-4, 1 - 1e-4)
    target_norm = np.clip(target_mean / 255.0, 1e-4, 1 - 1e-4)
    gamma = np.log(target_norm) / np.log(current_norm)
    return float(np.clip(gamma, gamma_min, gamma_max))


def ensure_odd_kernel(kernel_size: int) -> int:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def unsharp_mask(gray_img: np.ndarray, sigma: float = 1.0, amount: float = 0.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray_img, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(gray_img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def unsharp_mask_color(image_bgr: np.ndarray, sigma: float = 1.0, amount: float = 0.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_feature_image(
    image_bgr: np.ndarray,
    config: PreprocessConfig | None = None,
) -> dict[str, object]:
    cfg = config or PreprocessConfig()
    resized_bgr = resize_keep_aspect(image_bgr, cfg.max_width)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    before_metrics = compute_gray_metrics(gray)

    applied_steps: list[str] = []
    normalized = gray.copy()
    gamma = 1.0
    if cfg.enable_brightness_normalization:
        gamma = estimate_auto_gamma(
            gray,
            target_mean=cfg.target_brightness,
            gamma_min=cfg.gamma_min,
            gamma_max=cfg.gamma_max,
        )
        if abs(gamma - 1.0) > 0.03:
            normalized = adjust_gamma(gray, gamma)
            applied_steps.append(f"gamma:{gamma:.3f}")

    filtered = normalized.copy()
    if cfg.enable_denoise:
        filtered = cv2.fastNlMeansDenoising(
            filtered,
            None,
            h=cfg.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        applied_steps.append(f"denoise:h={cfg.denoise_strength}")

    kernel_size = ensure_odd_kernel(cfg.gaussian_kernel)
    if kernel_size > 1:
        filtered = cv2.GaussianBlur(filtered, (kernel_size, kernel_size), 0)
        applied_steps.append(f"gaussian:{kernel_size}")

    enhanced = filtered.copy()
    if cfg.enable_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_tile_grid_size, cfg.clahe_tile_grid_size),
        )
        enhanced = clahe.apply(enhanced)
        applied_steps.append(
            f"clahe:{cfg.clahe_clip_limit:.1f}/{cfg.clahe_tile_grid_size}x{cfg.clahe_tile_grid_size}"
        )

    final = enhanced.copy()
    if cfg.enable_unsharp:
        final = unsharp_mask(final, sigma=cfg.unsharp_sigma, amount=cfg.unsharp_amount)
        applied_steps.append(f"unsharp:{cfg.unsharp_amount:.2f}")

    after_metrics = compute_gray_metrics(final)
    return {
        "resized_bgr": resized_bgr,
        "gray": gray,
        "normalized": normalized,
        "filtered": filtered,
        "final": final,
        "applied_steps": applied_steps,
        "gamma": float(gamma),
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
    }


def preprocess_color_image(
    image_bgr: np.ndarray,
    config: PreprocessConfig | None = None,
) -> dict[str, object]:
    cfg = config or PreprocessConfig()
    working_bgr = resize_keep_aspect(image_bgr, cfg.max_width)

    applied_steps: list[str] = []
    if cfg.enable_denoise:
        working_bgr = cv2.fastNlMeansDenoisingColored(
            working_bgr,
            None,
            h=cfg.denoise_strength,
            hColor=cfg.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        applied_steps.append(f"denoise_color:h={cfg.denoise_strength}")

    lab = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    gamma = 1.0
    if cfg.enable_brightness_normalization:
        gamma = estimate_auto_gamma(
            l_channel,
            target_mean=cfg.target_brightness,
            gamma_min=cfg.gamma_min,
            gamma_max=cfg.gamma_max,
        )
        if abs(gamma - 1.0) > 0.03:
            l_channel = adjust_gamma(l_channel, gamma)
            applied_steps.append(f"gamma:{gamma:.3f}")

    if cfg.enable_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_tile_grid_size, cfg.clahe_tile_grid_size),
        )
        l_channel = clahe.apply(l_channel)
        applied_steps.append(
            f"lab_clahe:{cfg.clahe_clip_limit:.1f}/{cfg.clahe_tile_grid_size}x{cfg.clahe_tile_grid_size}"
        )

    merged_lab = cv2.merge([l_channel, a_channel, b_channel])
    final_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    if cfg.enable_unsharp:
        final_bgr = unsharp_mask_color(final_bgr, sigma=cfg.unsharp_sigma, amount=cfg.unsharp_amount)
        applied_steps.append(f"unsharp:{cfg.unsharp_amount:.2f}")

    return {
        "final": final_bgr,
        "applied_steps": applied_steps,
        "gamma": float(gamma),
    }

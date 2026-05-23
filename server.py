import os
import io
import uuid
import time
import shutil
import base64
import socket
import json
import dataclasses
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import preprocessing utilities from the project
from project_utils.preprocessing import (
    PreprocessConfig,
    preprocess_feature_image,
    preprocess_color_image,
    resize_keep_aspect,
    compute_gray_metrics
)
from scripts.PhamHungSon_15_portable_panorama_pipeline import (
    PanoramaConfig,
    PanoramaPipelineError,
    stitch_scene_folder,
)

app = FastAPI(title="Android Panorama Stitching Backend")

# Enable CORS for local network devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_SPLIT_ROOT = PROJECT_ROOT / "data" / "split"
SESSIONS_ROOT = PROJECT_ROOT / "data" / "sessions"

SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

# Mount presets for direct static access if needed
if DATA_SPLIT_ROOT.exists():
    app.mount("/static/presets", StaticFiles(directory=str(DATA_SPLIT_ROOT)), name="presets")
else:
    print(f"Warning: split data root not found at {DATA_SPLIT_ROOT}")

# Helper: Get local IP addresses to print in status endpoint
def get_local_ips() -> List[str]:
    ips = []
    try:
        # Get hostname
        hostname = socket.gethostname()
        # Get list of IP addresses associated with hostname
        ips = socket.gethostbyname_ex(hostname)[2]
        # Filter loopback and select IPv4
        ips = [ip for ip in ips if not ip.startswith("127.")]
    except Exception:
        pass
    
    # Fallback method
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
        
    return list(set(ips))

# Models for Request Validation
class PreprocessParams(BaseModel):
    max_width: int = 1200
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    enable_denoise: bool = False
    denoise_strength: int = 7
    enable_unsharp: bool = False
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 0.8
    enable_brightness_normalization: bool = True
    target_brightness: float = 128.0
    gaussian_kernel: int = 3
    color_mode: str = "grayscale" # "grayscale" or "color"

class FeatureParams(BaseModel):
    method: str = "SIFT"  # SIFT, ORB, AKAZE, HARRIS_HOG
    max_features: int = 2000
    orb_fast_threshold: int = 10
    harris_max_corners: int = 1500
    harris_quality: float = 0.01
    harris_min_distance: float = 8.0
    hog_patch_size: int = 32
    hog_cells: int = 4
    hog_bins: int = 8

class MatchingParams(BaseModel):
    lowe_ratio: float = 0.75
    ransac_threshold: float = 4.0

class PipelineRequest(BaseModel):
    source_type: str  # "preset" or "session"
    split: Optional[str] = None  # development, test, failure_analysis (if preset)
    scene_id: Optional[str] = None  # e.g., scene_03 (if preset)
    session_id: Optional[str] = None  # if session
    preprocessing: PreprocessParams = PreprocessParams()
    feature_extraction: FeatureParams = FeatureParams()
    matching: MatchingParams = MatchingParams()

class PortablePipelineRequest(BaseModel):
    source_type: str = "session"  # "preset" or "session"
    split: Optional[str] = None
    scene_id: Optional[str] = None
    session_id: Optional[str] = None
    config: Dict[str, Any] = {}

# Helper: encode cv2 image as base64 jpeg
def encode_img_to_b64(img: np.ndarray, quality: int = 80) -> str:
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

# Helper: parse and load image files from directory
def get_sorted_images_in_dir(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    
    # Try using sequence sorting if files match img_XX.jpg
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts]
    
    def sort_key(p: Path):
        # Extract numbers from stem
        nums = [int(s) for s in os.path.split(p.stem)[-1].replace('_', '').replace('img', '') if s.isdigit()]
        if nums:
            # Join digits to form a number
            try:
                num_str = "".join(str(s) for s in nums)
                return (int(num_str), p.name)
            except ValueError:
                pass
        # Fallback to string sort
        return (999999, p.name.lower())
        
    return sorted(files, key=sort_key)

# Endpoints
@app.get("/api/status")
def get_status():
    return {
        "status": "ok",
        "message": "Android Panorama Stitching Backend is active.",
        "local_ips": get_local_ips(),
        "opencv_version": cv2.__version__
    }

@app.get("/api/presets")
def get_presets():
    if not DATA_SPLIT_ROOT.exists():
        return {"splits": {}}
        
    result = {}
    for split_dir in DATA_SPLIT_ROOT.iterdir():
        if not split_dir.is_dir() or split_dir.name.startswith("."):
            continue
            
        scenes = {}
        for scene_dir in split_dir.iterdir():
            if not scene_dir.is_dir() or scene_dir.name.startswith("."):
                continue
                
            img_files = get_sorted_images_in_dir(scene_dir)
            images_list = [p.name for p in img_files]
            
            # Load metadata if exists
            meta_file = scene_dir / "meta.json"
            meta_content = {}
            if meta_file.exists():
                try:
                    import json
                    meta_content = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            
            scenes[scene_dir.name] = {
                "scene_id": scene_dir.name,
                "images": images_list,
                "meta": meta_content
            }
            
        result[split_dir.name] = scenes
        
    return {"splits": result}

@app.post("/api/session/new")
def create_session():
    session_id = str(uuid.uuid4())
    session_dir = SESSIONS_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return {"session_id": session_id, "message": f"Session {session_id} created."}

@app.post("/api/session/{session_id}/upload")
async def upload_image(session_id: str, file: UploadFile = File(...)):
    session_dir = SESSIONS_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found. Create a new one first.")
        
    # Get current files to determine next index
    existing_files = get_sorted_images_in_dir(session_dir)
    next_idx = len(existing_files) + 1
    
    # Save the file. Force extension to jpg or keep original
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        ext = ".jpg"
        
    filename = f"img_{next_idx:04d}{ext}"
    target_path = session_dir / filename
    
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "filename": filename,
        "index": next_idx,
        "message": f"Successfully uploaded {filename}",
        "session_images": [p.name for p in get_sorted_images_in_dir(session_dir)]
    }

@app.get("/api/session/{session_id}/images")
def get_session_images(session_id: str):
    session_dir = SESSIONS_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    
    files = get_sorted_images_in_dir(session_dir)
    return {
        "session_id": session_id,
        "images": [p.name for p in files]
    }

@app.post("/api/session/{session_id}/clear")
def clear_session(session_id: str):
    session_dir = SESSIONS_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
        
    for p in session_dir.iterdir():
        if p.is_file():
            p.unlink()
            
    return {"session_id": session_id, "message": "Session directory cleared."}

def resolve_scene_dir(source_type: str, split: Optional[str], scene_id: Optional[str], session_id: Optional[str]) -> Path:
    if source_type == "preset":
        if not split or not scene_id:
            raise HTTPException(status_code=400, detail="split and scene_id are required for source_type='preset'")
        scene_dir = DATA_SPLIT_ROOT / split / scene_id
        if not scene_dir.exists():
            raise HTTPException(status_code=404, detail=f"Preset scene {split}/{scene_id} not found.")
        return scene_dir
    if source_type == "session":
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required for source_type='session'")
        scene_dir = SESSIONS_ROOT / session_id
        if not scene_dir.exists():
            raise HTTPException(status_code=404, detail=f"Upload session {session_id} not found.")
        return scene_dir
    raise HTTPException(status_code=400, detail="Invalid source_type. Use 'preset' or 'session'")

def config_from_portable_payload(config_payload: Dict[str, Any], visualization_dir: Path) -> PanoramaConfig:
    allowed = {field.name for field in dataclasses.fields(PanoramaConfig)}
    filtered = {}
    for key, value in config_payload.items():
        if key not in allowed:
            continue
        if key in {"candidate_methods", "diagnostics_methods"} and isinstance(value, str):
            filtered[key] = [value]
        else:
            filtered[key] = value
    config = PanoramaConfig(**filtered)
    config.image_order = "name"
    if config.save_debug or config.save_pair_visualizations or config.save_score_table:
        config.visualization_dir = str(visualization_dir)
    return config

def encode_file_to_data_url(path: Path, mime: str = "image/jpeg") -> str:
    if not path.exists():
        return ""
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"

def file_md5(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def read_score_rows(score_csv: Path) -> List[Dict[str, Any]]:
    if not score_csv.exists():
        return []
    import csv
    with score_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def coerce_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def pair_from_score_row(row: Dict[str, Any], index: int, method: str) -> Dict[str, Any]:
    pair_id = row.get("pair_id") or f"pair_{index + 1:02d}"
    return {
        "pair_id": pair_id,
        "image_a": row.get("image_a") or f"img_{index + 1:04d}.jpg",
        "image_b": row.get("image_b") or f"img_{index + 2:04d}.jpg",
        "method": row.get("method") or method,
        "motion_model": row.get("motion_model") or "",
        "status": row.get("status") or "success",
        "raw_matches": int(coerce_number(row.get("raw_matches"), 0)),
        "good_matches": int(coerce_number(row.get("good_matches"), 0)),
        "inliers": int(coerce_number(row.get("inliers"), 0)),
        "inlier_ratio": coerce_number(row.get("inlier_ratio"), 0.0),
        "reprojection_error_mean": coerce_number(row.get("reprojection_error_mean") or row.get("mean_reprojection_error"), 0.0),
    }

def synthesize_pairs_from_scores(method_rows: List[Dict[str, Any]], selected_method: str) -> List[Dict[str, Any]]:
    rows_by_pair: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(method_rows):
        pair_id = row.get("pair_id") or f"pair_{index + 1:02d}"
        row_method = row.get("method") or ""
        if selected_method and row_method and row_method != selected_method:
            continue
        rows_by_pair.setdefault(pair_id, row)

    if not rows_by_pair:
        for index, row in enumerate(method_rows):
            pair_id = row.get("pair_id") or f"pair_{index + 1:02d}"
            rows_by_pair.setdefault(pair_id, row)

    return [pair_from_score_row(row, index, selected_method) for index, row in enumerate(rows_by_pair.values())]

def pair_visualization_payload(diagnostics_dir: Path, pair_id: str, methods: List[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for method in methods:
        method_dir = diagnostics_dir / pair_id / method
        method_payload = {
            "keypoints_a_uri": encode_file_to_data_url(method_dir / f"{pair_id}_{method}_keypoints_a.jpg"),
            "keypoints_b_uri": encode_file_to_data_url(method_dir / f"{pair_id}_{method}_keypoints_b.jpg"),
            "good_matches_uri": encode_file_to_data_url(method_dir / f"{pair_id}_{method}_good_matches.jpg"),
            "inlier_matches_uri": encode_file_to_data_url(method_dir / f"{pair_id}_{method}_inlier_matches.jpg"),
            "pair_warp_preview_uri": encode_file_to_data_url(method_dir / f"{pair_id}_{method}_pair_warp_preview.jpg"),
        }
        if any(method_payload.values()):
            payload[method] = method_payload
    return payload

def enrich_portable_server_payload(payload: Dict[str, Any], output_path: Path, log_path: Path, config_path: Path, visualization_dir: Path) -> Dict[str, Any]:
    diagnostics = payload.get("diagnostics") or {}
    diagnostics_dir = Path(diagnostics.get("diagnostics_dir") or visualization_dir)
    score_csv = Path(diagnostics.get("descriptor_score_csv") or diagnostics_dir / "descriptor_comparison_scores.csv")
    score_png = Path(diagnostics.get("descriptor_score_table") or diagnostics_dir / "descriptor_comparison_scores.png")
    score_json = Path(diagnostics.get("descriptor_score_json") or diagnostics_dir / "descriptor_comparison_scores.json")
    methods = diagnostics.get("diagnostic_methods") or []
    method_rows = read_score_rows(score_csv)
    config_payload = payload.get("config") or {}
    selected_method = str(config_payload.get("method") or "")
    pairs = payload.get("pairs") or []
    if not pairs and method_rows:
        pairs = synthesize_pairs_from_scores(method_rows, selected_method)

    enriched_pairs = []
    for index, pair in enumerate(pairs):
        pair_id = pair.get("pair_id") or f"pair_{index + 1:02d}"
        method = pair.get("method") or ""
        method_dir = diagnostics_dir / pair_id / method if method else diagnostics_dir / pair_id
        pair_methods = list(methods)
        if method and method not in pair_methods:
            pair_methods.insert(0, method)
        visualizations = pair_visualization_payload(diagnostics_dir, pair_id, pair_methods)
        selected_visuals = visualizations.get(method, {})
        enriched = dict(pair)
        enriched["pair_id"] = pair_id
        enriched["visualizations"] = visualizations
        enriched["keypoints_a_uri"] = encode_file_to_data_url(method_dir / f"{pair_id}_{method}_keypoints_a.jpg")
        enriched["keypoints_b_uri"] = encode_file_to_data_url(method_dir / f"{pair_id}_{method}_keypoints_b.jpg")
        enriched["good_matches_uri"] = encode_file_to_data_url(method_dir / f"{pair_id}_{method}_good_matches.jpg")
        enriched["inlier_matches_uri"] = encode_file_to_data_url(method_dir / f"{pair_id}_{method}_inlier_matches.jpg")
        enriched["pair_warp_preview_uri"] = encode_file_to_data_url(method_dir / f"{pair_id}_{method}_pair_warp_preview.jpg")
        enriched.update({key: value for key, value in selected_visuals.items() if value})
        enriched_pairs.append(enriched)

    payload["success"] = payload.get("status") == "ok"
    payload["panorama_b64"] = encode_file_to_data_url(output_path)
    payload["panoramaUri"] = payload["panorama_b64"]
    payload["panorama_md5"] = file_md5(output_path)
    payload["panorama_bytes"] = output_path.stat().st_size if output_path.exists() else 0
    payload["log_path"] = str(log_path)
    payload["config_path"] = str(config_path)
    payload["pairs"] = enriched_pairs
    payload["method_rows"] = method_rows
    payload["diagnostics"] = {
        **diagnostics,
        "diagnostics_dir": str(diagnostics_dir),
        "descriptor_score_csv": str(score_csv),
        "descriptor_score_table": str(score_png),
        "descriptor_score_json": str(score_json),
        "diagnostic_methods": methods,
    }
    payload["output_files"] = {
        "panorama": str(output_path),
        "log_json": str(log_path),
        "config_json": str(config_path),
        "descriptor_score_csv": str(score_csv),
        "descriptor_score_table": str(score_png),
        "descriptor_score_json": str(score_json),
    }
    payload["timing"] = {
        "total_sec": float(payload.get("runtime_sec") or 0.0),
        "stitching_sec": float(payload.get("runtime_sec") or 0.0),
    }
    return payload

@app.post("/api/process-portable")
def process_portable_pipeline(req: PortablePipelineRequest):
    scene_dir = resolve_scene_dir(req.source_type, req.split, req.scene_id, req.session_id)
    image_paths = get_sorted_images_in_dir(scene_dir)
    if len(image_paths) < 2:
        return {
            "success": False,
            "status": "error",
            "error_code": "ERR_NEED_MORE_IMGS",
            "error_message": f"Scene contains {len(image_paths)} images. At least 2 images are required.",
            "user_message": "Please select at least two images."
        }

    run_id = req.session_id or f"{req.split}_{req.scene_id}" or str(uuid.uuid4())
    pipeline_id = str((req.config or {}).get("pipeline_id") or "").strip()
    safe_pipeline_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pipeline_id)
    run_dir = SESSIONS_ROOT / str(run_id)
    if safe_pipeline_id:
        run_dir = run_dir / "runs" / safe_pipeline_id
    output_dir = run_dir / "output"
    config_dir = run_dir / "config"
    visualization_dir = run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "panorama.jpg"
    log_path = output_dir / "log.json"
    config_path = config_dir / "config.json"
    config = config_from_portable_payload(req.config, visualization_dir)
    config_path.write_text(json.dumps(dataclasses.asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        payload = stitch_scene_folder(scene_dir, output_path, config, log_path)
        return enrich_portable_server_payload(payload, output_path, log_path, config_path, visualization_dir)
    except PanoramaPipelineError as exc:
        payload = getattr(exc, "payload", {}) or {}
        payload["success"] = False
        payload["status"] = "error"
        payload["error_message"] = str(exc)
        payload.setdefault("user_message", str(exc))
        return enrich_portable_server_payload(payload, output_path, log_path, config_path, visualization_dir)
    except Exception as exc:
        return {
            "success": False,
            "status": "error",
            "error_type": "server_portable_error",
            "error_message": str(exc),
            "user_message": f"Own-PC server pipeline error: {exc}",
        }

# Master Stitching Pipeline execution
@app.post("/api/process-pipeline")
def process_pipeline(req: PipelineRequest):
    start_time_all = time.perf_counter()
    
    # 1. Resolve source directory
    if req.source_type == "preset":
        if not req.split or not req.scene_id:
            raise HTTPException(status_code=400, detail="split and scene_id are required for source_type='preset'")
        scene_dir = DATA_SPLIT_ROOT / req.split / req.scene_id
        if not scene_dir.exists():
            raise HTTPException(status_code=404, detail=f"Preset scene {req.split}/{req.scene_id} not found.")
    elif req.source_type == "session":
        if not req.session_id:
            raise HTTPException(status_code=400, detail="session_id is required for source_type='session'")
        scene_dir = SESSIONS_ROOT / req.session_id
        if not scene_dir.exists():
            raise HTTPException(status_code=404, detail=f"Upload session {req.session_id} not found.")
    else:
        raise HTTPException(status_code=400, detail="Invalid source_type. Use 'preset' or 'session'")
        
    image_paths = get_sorted_images_in_dir(scene_dir)
    if len(image_paths) < 2:
        return {
            "success": False,
            "error_message": f"Scene contains {len(image_paths)} images. At least 2 images are required to perform stitching.",
            "error_code": "ERR_NEED_MORE_IMGS"
        }
        
    # Loaded original images
    t_load_start = time.perf_counter()
    raw_images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise HTTPException(status_code=500, detail=f"Failed to read image: {path.name}")
        raw_images.append((path.name, img))
    t_load = time.perf_counter() - t_load_start
    
    # Define preprocessing configuration
    pp_config = PreprocessConfig(
        max_width=req.preprocessing.max_width,
        gaussian_kernel=req.preprocessing.gaussian_kernel,
        clahe_clip_limit=req.preprocessing.clahe_clip_limit,
        clahe_tile_grid_size=req.preprocessing.clahe_tile_grid_size,
        enable_clahe=req.preprocessing.enable_clahe,
        enable_brightness_normalization=req.preprocessing.enable_brightness_normalization,
        enable_denoise=req.preprocessing.enable_denoise,
        denoise_strength=req.preprocessing.denoise_strength,
        enable_unsharp=req.preprocessing.enable_unsharp,
        unsharp_sigma=req.preprocessing.unsharp_sigma,
        unsharp_amount=req.preprocessing.unsharp_amount,
        target_brightness=req.preprocessing.target_brightness
    )
    
    # 2. Stage: PREPROCESSING
    t_preprocess_start = time.perf_counter()
    preprocessed_gray = []
    preprocessed_color = []
    visual_raw_list = []
    visual_prep_list = []
    prep_applied_steps = []
    
    for name, img in raw_images:
        # Preprocess features (always grayscale)
        prep_feat = preprocess_feature_image(img, pp_config)
        # Preprocess color (for final color stitching if color mode enabled)
        if req.preprocessing.color_mode == "color":
            prep_col = preprocess_color_image(img, pp_config)
            col_img = prep_col["final"]
        else:
            col_img = cv2.cvtColor(prep_feat["final"], cv2.COLOR_GRAY2BGR)
            
        preprocessed_gray.append((name, prep_feat["final"]))
        preprocessed_color.append((name, col_img))
        prep_applied_steps = prep_feat["applied_steps"]
        
        # Prepare small versions of raw and prep images for client response to save bandwidth
        raw_small = resize_keep_aspect(img, 480)
        prep_small = resize_keep_aspect(col_img, 480)
        visual_raw_list.append(encode_img_to_b64(raw_small, 75))
        visual_prep_list.append(encode_img_to_b64(prep_small, 75))
        
    t_preprocess = time.perf_counter() - t_preprocess_start
    
    # 3. Stage: FEATURE EXTRACTION
    t_features_start = time.perf_counter()
    keypoints_per_img = []
    descriptors_per_img = []
    visual_keypoint_list = []
    keypoint_counts = []
    
    method = req.feature_extraction.method.upper()
    max_features = req.feature_extraction.max_features
    
    # Setup detector
    if method == "SIFT":
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm_type = cv2.NORM_L2
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=req.feature_extraction.orb_fast_threshold)
        norm_type = cv2.NORM_HAMMING
    elif method == "AKAZE":
        detector = cv2.AKAZE_create()
        norm_type = cv2.NORM_HAMMING
    elif method == "HARRIS_HOG":
        # Custom Harris + HOG implementation helper
        pass
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported feature extraction method: {method}")
        
    for name, gray_img in preprocessed_gray:
        # Detect and describe
        if method == "HARRIS_HOG":
            # Harris corners
            corner_limit = min(req.feature_extraction.harris_max_corners, max_features) if max_features > 0 else req.feature_extraction.harris_max_corners
            corners = cv2.goodFeaturesToTrack(
                gray_img,
                maxCorners=corner_limit,
                qualityLevel=req.feature_extraction.harris_quality,
                minDistance=req.feature_extraction.harris_min_distance,
                blockSize=3,
                useHarrisDetector=True,
                k=0.04
            )
            kps = []
            if corners is not None:
                for x, y in corners.reshape(-1, 2):
                    kps.append(cv2.KeyPoint(float(x), float(y), 31.0))
            
            # Simple HOG description
            from scripts.PhamHungSon_15_extract_features import compute_hog_descriptors, sort_and_limit_keypoints
            kps, descs = compute_hog_descriptors(
                gray_img,
                kps,
                patch_size=req.feature_extraction.hog_patch_size,
                cells_per_side=req.feature_extraction.hog_cells,
                bins=req.feature_extraction.hog_bins
            )
            kps, descs = sort_and_limit_keypoints(kps, descs, max_features)
            norm_type = cv2.NORM_L2
        else:
            kps, descs = detector.detectAndCompute(gray_img, None)
            if kps is None:
                kps = []
            # For AKAZE, limit keypoints if they exceed max_features
            if method == "AKAZE" and max_features > 0 and len(kps) > max_features:
                order = sorted(range(len(kps)), key=lambda idx: kps[idx].response, reverse=True)[:max_features]
                kps = [kps[i] for i in order]
                if descs is not None:
                    descs = descs[np.array(order)]
                    
        keypoints_per_img.append(kps)
        descriptors_per_img.append(descs)
        keypoint_counts.append(len(kps))
        
        # Render keypoint overlay
        # Load color preprocessed image to draw keypoints on
        prep_col = next(col for n, col in preprocessed_color if n == name)
        kp_draw_img = prep_col.copy()
        # Draw rich keypoints (limit to 600 for performance/clarity on mobile)
        draw_limit = min(600, len(kps))
        draw_kps = sorted(kps, key=lambda k: k.response, reverse=True)[:draw_limit] if draw_limit < len(kps) else kps
        cv2.drawKeypoints(
            kp_draw_img,
            draw_kps,
            kp_draw_img,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        visual_keypoint_list.append(encode_img_to_b64(resize_keep_aspect(kp_draw_img, 480), 75))
        
    t_features = time.perf_counter() - t_features_start
    
    # 4. Stage: PAIRWISE MATCHING
    t_matching_start = time.perf_counter()
    pairs_info = []
    
    for i in range(len(preprocessed_gray) - 1):
        name_a, gray_a = preprocessed_gray[i]
        name_b, gray_b = preprocessed_gray[i+1]
        kps_a = keypoints_per_img[i]
        kps_b = keypoints_per_img[i+1]
        descs_a = descriptors_per_img[i]
        descs_b = descriptors_per_img[i+1]
        color_a = preprocessed_color[i][1]
        color_b = preprocessed_color[i+1][1]
        
        good_matches = []
        inliers_count = 0
        inlier_ratio = 0.0
        homography_ok = False
        match_img_b64 = ""
        
        # Perform match if both have descriptors
        if descs_a is not None and descs_b is not None and len(descs_a) >= 4 and len(descs_b) >= 4:
            matcher = cv2.BFMatcher(norm_type, crossCheck=False)
            try:
                knn_matches = matcher.knnMatch(descs_a, descs_b, k=2)
                # Lowe's ratio test
                for pair in knn_matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < req.matching.lowe_ratio * n.distance:
                            good_matches.append(m)
            except Exception as e:
                print(f"Matcher error: {e}")
                
            # Estimate Homography using RANSAC
            if len(good_matches) >= 4:
                src_pts = np.float32([kps_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kps_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, req.matching.ransac_threshold)
                if mask is not None:
                    inliers_count = int(mask.sum())
                    inlier_ratio = float(inliers_count / len(good_matches)) if len(good_matches) > 0 else 0.0
                    homography_ok = bool(homography is not None and inliers_count >= 8)
                    
                    # Create custom side-by-side match overlay
                    # Draw inliers in green, outliers in red
                    h_a, w_a = color_a.shape[:2]
                    h_b, w_b = color_b.shape[:2]
                    
                    # Resize to same heights if mismatch
                    target_h = max(h_a, h_b)
                    
                    # Make combined canvas
                    canvas = np.zeros((target_h, w_a + w_b, 3), dtype=np.uint8)
                    canvas[0:h_a, 0:w_a] = color_a
                    canvas[0:h_b, w_a:w_a+w_b] = color_b
                    
                    mask_flat = mask.ravel()
                    # Limit drawings for visual clarity (max 100 matches)
                    draw_indices = list(range(len(good_matches)))
                    if len(good_matches) > 100:
                        # Prioritize inliers
                        inlier_indices = [idx for idx in draw_indices if mask_flat[idx]]
                        outlier_indices = [idx for idx in draw_indices if not mask_flat[idx]]
                        draw_indices = inlier_indices[:70] + outlier_indices[:30]
                        
                    for idx in draw_indices:
                        m = good_matches[idx]
                        pt_a = kps_a[m.queryIdx].pt
                        pt_b = kps_b[m.trainIdx].pt
                        
                        pt_a_canvas = (int(round(pt_a[0])), int(round(pt_a[1])))
                        pt_b_canvas = (int(round(pt_b[0])) + w_a, int(round(pt_b[1])))
                        
                        is_inlier = bool(mask_flat[idx])
                        line_color = (0, 255, 0) if is_inlier else (0, 0, 255) # Green vs Red
                        thickness = 1 if is_inlier else 1
                        
                        cv2.line(canvas, pt_a_canvas, pt_b_canvas, line_color, thickness, cv2.LINE_AA)
                        cv2.circle(canvas, pt_a_canvas, 3, line_color, -1)
                        cv2.circle(canvas, pt_b_canvas, 3, line_color, -1)
                        
                    # Scale down matching canvas for transmission
                    canvas_small = resize_keep_aspect(canvas, 640)
                    match_img_b64 = encode_img_to_b64(canvas_small, 70)
                    
        pairs_info.append({
            "image_a": name_a,
            "image_b": name_b,
            "raw_matches": len(good_matches), # Note: we count Lowe filtered as good matches
            "inliers": inliers_count,
            "inlier_ratio": round(inlier_ratio, 3),
            "homography_ok": homography_ok,
            "matches_b64": match_img_b64
        })
        
    t_matching = time.perf_counter() - t_matching_start
    
    # 5. Stage: GLOBAL PANORAMA STITCHING
    t_stitch_start = time.perf_counter()
    stitcher_status_code = -1
    stitcher_status_msg = "UNKNOWN"
    panorama_b64 = ""
    stitching_success = False
    diagnostic_details = ""
    
    # We will use preprocessed color images for stitching
    stitcher_images = [img for _, img in preprocessed_color]
    
    # Initialize OpenCV Stitcher
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    try:
        # Disable OpenCL to prevent threading crashes in python scripts/notebooks
        cv2.ocl.setUseOpenCL(False)
        status_code, panorama = stitcher.stitch(stitcher_images)
        stitcher_status_code = int(status_code)
        
        STATUS_NAMES = {
            0: "OK",
            1: "ERR_NEED_MORE_IMGS",
            2: "ERR_HOMOGRAPHY_EST_FAIL",
            3: "ERR_CAMERA_PARAMS_ADJUST_FAIL"
        }
        stitcher_status_msg = STATUS_NAMES.get(stitcher_status_code, f"ERR_CODE_{stitcher_status_code}")
        
        if stitcher_status_code == 0 and panorama is not None:
            stitching_success = True
            # Optional: Crop black boundaries if needed
            # We will send the full panorama back, let the user see details
            pano_small = resize_keep_aspect(panorama, 1000)
            panorama_b64 = encode_img_to_b64(pano_small, 85)
        else:
            # Generate custom diagnostics
            if stitcher_status_code == 1:
                diagnostic_details = "OpenCV Stitcher needs more images to determine overlaps. This usually means the overlap area between captured frames is too narrow."
            elif stitcher_status_code == 2:
                # Find which pair failed
                failed_pairs = [p for p in pairs_info if not p["homography_ok"]]
                if failed_pairs:
                    pair_names = [f"'{p['image_a']}' & '{p['image_b']}' (Inliers: {p['inliers']})" for p in failed_pairs]
                    diagnostic_details = f"Homography estimation failed between: {', '.join(pair_names)}. Ensure there are enough distinct visual features in overlapping areas."
                else:
                    diagnostic_details = "Homography estimation failed globally. The images might not be in the correct sequence, or camera orientations differ too much."
            elif stitcher_status_code == 3:
                diagnostic_details = "Camera parameter adjustment failed. OpenCV could not bundle-adjust focal length and translation variables into a single projection canvas."
                
    except Exception as e:
        stitcher_status_code = -999
        stitcher_status_msg = "EXCEPTION"
        diagnostic_details = f"Stitcher crashed with exception: {str(e)}"
        
    t_stitch = time.perf_counter() - t_stitch_start
    t_total = time.perf_counter() - start_time_all
    
    # 6. Assemble complete JSON response
    response_payload = {
        "success": stitching_success,
        "error_code": stitcher_status_msg,
        "error_message": diagnostic_details if not stitching_success else "",
        "images": [
            {
                "name": raw_images[idx][0],
                "raw_b64": visual_raw_list[idx],
                "preprocessed_b64": visual_prep_list[idx],
                "keypoints_count": keypoint_counts[idx],
                "keypoints_b64": visual_keypoint_list[idx]
            } for idx in range(len(raw_images))
        ],
        "pairs": pairs_info,
        "panorama_b64": panorama_b64,
        "preprocessing_steps": prep_applied_steps,
        "timing": {
            "io_sec": round(t_load, 4),
            "preprocess_sec": round(t_preprocess, 4),
            "features_sec": round(t_features, 4),
            "matching_sec": round(t_matching, 4),
            "stitching_sec": round(t_stitch, 4),
            "total_sec": round(t_total, 4)
        }
    }
    
    return response_payload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

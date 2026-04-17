import json
import os
from collections import Counter

def auto_flag_scene(scene_path):
    meta_path = os.path.join(scene_path, "meta.json")
    if not os.path.exists(meta_path): return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    image_stats = meta.get("image_stats", [])
    pair_audit = meta.get("pair_audit", [])
    if not image_stats or not pair_audit: return None

    # Calculate metrics
    blur_scores = [stat.get("blur_score", 999) for stat in image_stats]
    min_blur = min(blur_scores)

    brightness_list = [stat.get("brightness_mean", 100) for stat in image_stats]
    brightness_span = max(brightness_list) - min(brightness_list)
    max_adj_diff = 0
    for i in range(len(brightness_list) - 1):
        diff = abs(brightness_list[i] - brightness_list[i+1])
        if diff > max_adj_diff: max_adj_diff = diff

    entropies = [stat.get("entropy", 8.0) for stat in image_stats]
    min_entropy = min(entropies)
    
    contrasts = [stat.get("contrast_std", 50) for stat in image_stats]
    min_contrast = min(contrasts)

    inliers = [pair.get("inliers", 999) for pair in pair_audit]
    min_inliers = min(inliers)
    inlier_ratios = [pair.get("inlier_ratio", 1.0) for pair in pair_audit]
    min_ratio = min(inlier_ratios)
    
    reproj_errors = [pair.get("median_reproj_error", 0) for pair in pair_audit]
    max_reproj = max(reproj_errors)
    
    raw_matches = [pair.get("raw_matches", 0) for pair in pair_audit]
    good_matches = [pair.get("good_matches", 0) for pair in pair_audit]

    # Evaluate Flags
    flags = {}
    flags["has_motion_blur"] = bool(min_blur < 65.0)
    flags["has_exposure_change"] = bool(brightness_span > 25.0 or max_adj_diff > 15.0)
    flags["has_insufficient_overlap"] = bool(min_inliers < 40 or min_ratio < 0.35)
    flags["has_low_texture"] = bool(min_entropy < 6.8 or min_contrast < 30.0)
    flags["has_parallax"] = bool(max_reproj > 1.8)
    
    # Check repeated patterns (lots of good_matches but very few inliers due to RANSAC throwing them out)
    has_repeated = False
    for i in range(len(pair_audit)):
        if good_matches[i] > 400 and inlier_ratios[i] < 0.35:
            has_repeated = True
    flags["has_repeated_patterns"] = bool(has_repeated)

    # We keep moving_objects as whatever it was, or false
    flags["has_moving_objects"] = meta.get("has_moving_objects", False)
    if flags["has_moving_objects"] is None: flags["has_moving_objects"] = False

    # Apply flags
    for k, v in flags.items():
        meta[k] = v

    # Auto-generate issues array (preserve manual non-heuristic issues if any, but reconstruct heuristic ones)
    current_issues = set(meta.get("issues", []))
    heuristic_issue_names = ["motion_blur", "exposure_change", "insufficient_overlap", "low_texture", "parallax", "repeated_patterns", "moving_objects"]
    # Remove heuristic issues from current_issues to rebuild them freshly
    current_issues = {issue for issue in current_issues if issue not in heuristic_issue_names}
    
    if flags["has_motion_blur"]: current_issues.add("motion_blur")
    if flags["has_exposure_change"]: current_issues.add("exposure_change")
    if flags["has_insufficient_overlap"]: current_issues.add("insufficient_overlap")
    if flags["has_low_texture"]: current_issues.add("low_texture")
    if flags["has_parallax"]: current_issues.add("parallax")
    if flags["has_repeated_patterns"]: current_issues.add("repeated_patterns")
    if flags["has_moving_objects"]: current_issues.add("moving_objects")

    meta["issues"] = sorted(list(current_issues))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        
    return flags

if __name__ == "__main__":
    base_dir = "data/raw"
    summary = Counter()
    for scene in sorted(os.listdir(base_dir)):
        if not scene.startswith("scene_"): continue
        path = os.path.join(base_dir, scene)
        if not os.path.isdir(path): continue
        res = auto_flag_scene(path)
        if res:
            for k, v in res.items():
                if v: summary[k] += 1
            print(f"Processed {scene}")
    
    print("\\n--- FLAG DETECTIONS ---")
    for k, v in summary.items():
        print(f"{k}: {v} scenes")

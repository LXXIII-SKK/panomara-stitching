import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audit"

def main():
    image_metrics_path = AUDIT_ROOT / "image_metrics.csv"
    scene_metrics_path = AUDIT_ROOT / "scene_metrics.csv"
    pair_metrics_path = AUDIT_ROOT / "pair_metrics.csv"
    
    if not (image_metrics_path.exists() and scene_metrics_path.exists() and pair_metrics_path.exists()):
        print("Error: Audit CSV files not found.")
        return
        
    df_img = pd.read_csv(image_metrics_path)
    df_scene = pd.read_csv(scene_metrics_path)
    df_pair = pd.read_csv(pair_metrics_path)
    
    print("======================================================================")
    print("                     DATASET CURATION & AUDIT ANALYSIS                ")
    print("======================================================================")
    
    # 1. Basic Curation Breakdown
    print(f"\n1. DATA SPLITS & IMAGES BREAKDOWN:")
    for split in ['development', 'test', 'failure_analysis']:
        split_scenes = df_scene[df_scene['split'] == split]['scene_id'].nunique()
        split_imgs = df_img[df_img['split'] == split].drop_duplicates(subset=['scene_id', 'image_id']).shape[0]
        print(f"   • {split:<18}: {split_scenes:>2} scenes, {split_imgs:>3} images")
        
    # 2. Image Properties (Resolution, Orientation, Layouts)
    print(f"\n2. IMAGE RESOLUTIONS & GEOMETRIES:")
    # Deduplicate image rows (since they might be duplicated per feature method)
    df_img_uniq = df_img.drop_duplicates(subset=['scene_id', 'image_id'])
    
    resolutions = df_img_uniq.groupby(['width', 'height']).size().reset_index(name='count')
    print("   Common resolutions in dataset:")
    for idx, row in resolutions.sort_values(by='count', ascending=False).iterrows():
        aspect = row['width'] / row['height']
        orient = "Landscape" if aspect > 1.05 else ("Portrait" if aspect < 0.95 else "Square")
        print(f"   • {row['width']} x {row['height']} ({orient}, Aspect: {aspect:.2f}) -> {row['count']} images")
        
    # Landscape vs Portrait count
    landscape_cnt = df_img_uniq[df_img_uniq['width'] > df_img_uniq['height']].shape[0]
    portrait_cnt = df_img_uniq[df_img_uniq['width'] < df_img_uniq['height']].shape[0]
    square_cnt = df_img_uniq[df_img_uniq['width'] == df_img_uniq['height']].shape[0]
    print(f"   • Geometry counts: Landscape: {landscape_cnt}, Portrait: {portrait_cnt}, Square: {square_cnt}")
    
    # Check if there are scenes that mix orientations (which usually breaks homographies)
    mixed_scenes = []
    for scene_id, group in df_img_uniq.groupby('scene_id'):
        widths = group['width'].unique()
        heights = group['height'].unique()
        if len(widths) > 1 or len(heights) > 1:
            mixed_scenes.append(scene_id)
    if mixed_scenes:
        print(f"   • WARNING: Mixed-resolution scenes detected! {mixed_scenes}")
    else:
        print(f"   • Success: All images within any single scene share identical resolutions and aspect ratios.")
        
    # 3. Quality & Preprocessing Flags
    print(f"\n3. SCENE DIFFICULTIES & PROBLEM TAGS:")
    df_scene_uniq = df_scene.drop_duplicates(subset=['scene_id'])
    
    diff_counts = df_scene_uniq['meta_difficulty'].value_counts()
    print("   Difficulty levels distribution:")
    for diff, count in diff_counts.items():
        print(f"   • {diff:<10}: {count} scenes")
        
    # Preprocess Recommendations
    recs = df_img_uniq['preprocess_recommendations'].dropna().value_counts()
    if not recs.empty:
        print("\n   Key Preprocessing Recommendations:")
        for rec, count in recs.items():
            print(f"   • {rec}: {count} images recommended")
            
    # Exposure Imbalances (Brightness gaps between adjacent images)
    print(f"\n4. EXPOSURE & CONTRAST ANOMALIES:")
    large_brightness_gaps = df_pair[df_pair['brightness_gap'] > 20.0].drop_duplicates(subset=['scene_id', 'pair_id'])
    print(f"   • Adjacent pairs with extreme exposure imbalances (>20.0 gray levels): {large_brightness_gaps.shape[0]} pairs")
    if not large_brightness_gaps.empty:
        for idx, row in large_brightness_gaps.head(5).iterrows():
            print(f"     - Scene {row['scene_id']}, Pair {row['pair_id']}: Brightness Gap = {row['brightness_gap']:.1f}")
            
    # Blurry and Low-Contrast Detection
    blurry_imgs = df_img_uniq[df_img_uniq['blur_score'] < 100.0]
    low_contrast_imgs = df_img_uniq[df_img_uniq['contrast_std'] < 30.0]
    print(f"   • Blurry images (Laplacian variance < 100.0): {blurry_imgs.shape[0]} images")
    print(f"   • Low-contrast images (Intensity std-dev < 30.0): {low_contrast_imgs.shape[0]} images")
    
    # 5. RANSAC Geometry Gaps
    print(f"\n5. GEOMETRIC STABILITY (RANSAC & HOMOGRAPHY SANITY):")
    invalid_homographies = df_pair[df_pair['homography_ok'] == False].drop_duplicates(subset=['scene_id', 'pair_id'])
    print(f"   • Pairs where RANSAC failed to find a valid Homography: {invalid_homographies.shape[0]} pairs")
    
    bad_sanity = df_pair[df_pair['transform_sanity_label'] == 'fail'].drop_duplicates(subset=['scene_id', 'pair_id'])
    print(f"   • Pairs failing transform geometric sanity check (severe skew/distortion/drift): {bad_sanity.shape[0]} pairs")
    if not bad_sanity.empty:
        for idx, row in bad_sanity.head(5).iterrows():
            print(f"     - Scene {row['scene_id']}, Pair {row['pair_id']}: Sanity Flags = {row['transform_sanity_flags']}")

if __name__ == "__main__":
    main()

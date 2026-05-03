# Panorama Stitching Project

This repository contains the implementation and experimental report materials for a classical panorama stitching pipeline. The project studies how local feature detectors, descriptors, matching, geometric verification, warping, and blending affect panorama quality across easy, difficult, and failure-oriented scenes.

## Project Scope

The current project stage covers:

- dataset organization and scene metadata
- image quality and overlap audit
- preprocessing for feature extraction
- single-scene feature/descriptor case study
- OpenCV Stitcher baseline
- manual projection previews for geometry analysis

The full-dataset feature, matching, stitching, and evaluation batches will be built on top of the existing scripts.

## Repository Structure

```text
data/
  raw/                         original scene folders with meta.json
  split/                       development, test, and failure-analysis splits
  preprocessing/               saved preprocessing outputs

notebooks/
  01_data_audit.ipynb
  02_preprocessing.ipynb
  03_feature_extractor.ipynb
  06_opencv_scene_stitcher.ipynb
  07_manual_projection_previews.ipynb

project_utils/
  panorama_dataset.py          dataset and metadata helpers
  preprocessing.py             reusable preprocessing functions

scripts/
  apply_preprocessing.py       batch preprocessing
  extract_features.py          batch feature/descriptor extraction
  generate_opencv_overlay.py   visualization for OpenCV panorama placement
  regenerate_scene_meta.py     scene metadata and audit regeneration
  rename_scene_images.py       dataset image naming utility
  validate_dataset.py          dataset consistency checks
```

`data/` and `outputs/` are ignored by Git because they contain local datasets and generated experiment artifacts.

## Notebook Order

1. `01_data_audit.ipynb`  
   Audits scene quality, overlap, image statistics, keypoints, pair matches, and failure indicators.

2. `02_preprocessing.ipynb`  
   Documents preprocessing concepts and the saved preprocessing pipeline used before feature extraction.

3. `03_feature_extractor.ipynb`  
   Compares SIFT, ORB, Harris + ORB, Harris + HOG, AKAZE, and BRISK on one preprocessed scene pair.

4. `06_opencv_scene_stitcher.ipynb`  
   Runs the OpenCV Stitcher baseline and records panorama outputs and logs.

5. `07_manual_projection_previews.ipynb`  
   Shows manual homography-chain and cylindrical projection previews for geometry analysis.

## Reproducible Commands

Run these commands from the project root. The project environment used for notebooks is `image_recognition`.

### Validate Dataset

```powershell
conda run -n image_recognition python scripts\validate_dataset.py
conda run -n image_recognition python scripts\validate_dataset.py --verify-images
```

### Apply Preprocessing

```powershell
conda run -n image_recognition python scripts\apply_preprocessing.py --split development --ordered-only
conda run -n image_recognition python scripts\apply_preprocessing.py --split test --ordered-only
conda run -n image_recognition python scripts\apply_preprocessing.py --split failure --ordered-only
```

Preprocessed feature images are written to:

```text
data/preprocessing/<split>/feature_gray/<scene>/
```

### Extract Features

Run one descriptor on one scene:

```powershell
conda run -n image_recognition python scripts\extract_features.py --split development --scene scene_03 --descriptor SIFT
```

Run one descriptor on an entire split:

```powershell
conda run -n image_recognition python scripts\extract_features.py --split development --descriptor SIFT
```

Run all supported comparison descriptors on a split:

```powershell
conda run -n image_recognition python scripts\extract_features.py --split development --descriptor all
```

Feature outputs are written to:

```text
data/feature_extract/<split>/<scene>/<descriptor>/
```

## Report Focus

The feature/descriptor case study should be interpreted as a controlled single-scene comparison. It uses already-preprocessed images and does not claim full-dataset performance. The later batch experiments should use the same output structure to summarize performance across development, test, and failure-analysis splits.

Key metrics for stitching quality:

- median Lowe ratio
- Lowe pass rate
- RANSAC inliers
- inlier ratio
- inlier Lowe ratio
- spatial coverage
- reprojection error
- overlap similarity
- homography sanity
- visual alignment
- ghosting and seam quality
- runtime

## Reference Documents

- [Project pipeline report](./panorama_pipeline_report.md)
- [Method comparison report plan](./panorama_method_comparison_report.md)
- [Project guild PDF](./CV_20_Project_2026_English.pdf)
- [Local features lab manual](./local_features_lab_manual.pdf)

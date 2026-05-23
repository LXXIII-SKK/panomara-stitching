# Panorama Stitching Project - Final Exam Report

This repository contains the implementation and final-exam report materials for a classical panorama stitching pipeline. The project studies how local feature detectors, descriptors, matching, geometric verification, warping, and blending affect panorama quality across easy, difficult, and failure-oriented scenes.

## Project Scope

The current project stage covers:

- dataset organization and scene metadata
- image quality and overlap audit
- preprocessing for feature extraction
- single-scene feature/descriptor case study
- batch feature matching and RANSAC evaluation
- manual homography-chain stitching from cached features
- OpenCV Stitcher baseline
- manual-vs-OpenCV stitching comparison
- manual projection previews for geometry analysis

The default notebook/script path now focuses on `test` and `failure_analysis` because these are the scenes that need manual judgment for the showcase. The `development` split is still kept for internal tuning and can be run explicitly with `--split development` or `--split all`.

## Repository Structure

```text
data/
  split/                       downloaded dataset: development, test, and failure-analysis splits
  preprocessing/               saved preprocessing outputs
  feature_extract/             saved feature/descriptor outputs

notebooks/
  PhamHungSon_15_01_data_audit.ipynb
  PhamHungSon_15_02_preprocessing.ipynb
  PhamHungSon_15_03_feature_extractor.ipynb
  PhamHungSon_15_04_batch_feature_matching.ipynb
  PhamHungSon_15_05_manual_homography_stitcher.ipynb
  PhamHungSon_15_06_opencv_scene_stitcher.ipynb
  PhamHungSon_15_07_manual_projection_previews.ipynb

project_utils/
  panorama_dataset.py          dataset and metadata helpers
  preprocessing.py             reusable preprocessing functions

scripts/
  PhamHungSon_15_apply_preprocessing.py       batch preprocessing
  PhamHungSon_15_extract_features.py          batch feature/descriptor extraction
  PhamHungSon_15_manual_homography_stitcher.py project manual stitcher batch runner
  PhamHungSon_15_compare_manual_opencv_stitching.py manual-vs-OpenCV comparison
  PhamHungSon_15_portable_panorama_pipeline.py portable one-scene panorama pipeline
  PhamHungSon_15_generate_opencv_overlay.py   visualization for OpenCV panorama placement
  PhamHungSon_15_validate_dataset.py          dataset consistency checks
```

`data/` and `outputs/` are ignored by Git because they contain local datasets and generated experiment artifacts. The shared dataset contains the split folders only; `data/raw` is not required for running the notebooks.

Dataset link:

```text
https://drive.google.com/drive/folders/1MLb_hScN_5qY6bSoVM65Q9w8KLP_sLU9?usp=drive_link
```

After downloading, the expected layout is:

```text
data/split/development/scene_XX/
data/split/test/scene_XX/
data/split/failure_analysis/scene_XX/
```

## Notebook Order

1. `PhamHungSon_15_01_data_audit.ipynb`  
   Audits scene quality, overlap, image statistics, keypoints, pair matches, and failure indicators.

2. `PhamHungSon_15_02_preprocessing.ipynb`  
   Documents preprocessing concepts and the saved preprocessing pipeline used before feature extraction.

3. `PhamHungSon_15_03_feature_extractor.ipynb`  
   Compares ORB, AKAZE, Harris + HOG, and SIFT on one preprocessed scene pair.

4. `PhamHungSon_15_04_batch_feature_matching.ipynb`  
   Runs the same method pipelines across selected or full dataset scene pairs, then saves pair, scene, method-level RANSAC summaries, and visualization dashboards.

5. `PhamHungSon_15_05_manual_homography_stitcher.ipynb`  
   Builds a from-scratch homography-chain panorama using cached features and Notebook 04 pair-method scores, then compares available manual outputs against the OpenCV Stitcher baseline.

6. `PhamHungSon_15_06_opencv_scene_stitcher.ipynb`  
   Runs the OpenCV Stitcher baseline and records panorama outputs and logs.

7. `PhamHungSon_15_07_manual_projection_previews.ipynb`  
   Shows manual homography-chain and cylindrical projection previews for geometry analysis.

## Reproducible Commands

Run these commands from the project root. The project environment used for notebooks is `image_recognition`.

### Validate Dataset

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_validate_dataset.py
conda run -n image_recognition python scripts\PhamHungSon_15_validate_dataset.py --verify-images
```

### Apply Preprocessing

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_apply_preprocessing.py --split test --ordered-only
conda run -n image_recognition python scripts\PhamHungSon_15_apply_preprocessing.py --split failure_analysis --ordered-only
```

Preprocessed feature images are written to:

```text
data/preprocessing/<split>/feature_gray/<scene>/
```

### Extract Features

Run one descriptor on one scene:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_extract_features.py --split failure_analysis --scene scene_14 --descriptor SIFT
```

Run one descriptor on an entire split:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_extract_features.py --split test --descriptor SIFT
```

Run all supported comparison descriptors on a split:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_extract_features.py --split test --descriptor all
conda run -n image_recognition python scripts\PhamHungSon_15_extract_features.py --split failure_analysis --descriptor all
```

Feature outputs are written to:

```text
data/feature_extract/<split>/<scene>/<descriptor>/
```

### Manual Stitching and OpenCV Comparison

Run the manual homography stitcher over the showcase splits:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_manual_homography_stitcher.py --split test --split failure_analysis --profile balanced --method auto --feature-source cache --use-batch-metrics --allow-partial --overwrite
```

Compare manual outputs with OpenCV Stitcher outputs:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_compare_manual_opencv_stitching.py --side-by-side-limit 16
```

The comparison table and plots are written to:

```text
outputs/manual_homography_stitcher/comparison/
```

## Report Focus

The feature/descriptor case study should be interpreted as a controlled single-scene comparison. It uses already-preprocessed images and does not claim full-dataset performance. The showcase notebooks summarize `test` and `failure_analysis`; development remains available for further tuning.

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

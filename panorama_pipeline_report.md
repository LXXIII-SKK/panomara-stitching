# Panorama Stitching Pipeline Report

## Objective

The project builds and evaluates a classical panorama stitching workflow for ordered image sequences with overlap. The main goal is to compare how feature extraction, descriptor quality, geometric verification, and stitching choices affect panorama quality.

The system is designed for:

- indoor and outdoor scenes
- short core sequences of approximately 2 to 6 images
- harder scenes with blur, exposure change, repeated texture, parallax, or weak overlap
- larger scenes used as stress tests after the core pipeline is stable

## Current Pipeline

```text
Scene images
-> dataset audit and metadata
-> preprocessing
-> feature detection and descriptor extraction
-> descriptor matching
-> RANSAC inlier filtering
-> homography estimation
-> image warping
-> overlap blending
-> panorama output and quality evaluation
```

## Dataset Organization

```text
data/raw/
  scene_XX/
    img_01.jpg
    img_02.jpg
    ...
    meta.json

data/split/
  development/
  test/
  failure_analysis/

data/preprocessing/
  <split>/
    feature_gray/
      scene_XX/
        img_01.png
        img_02.png
```

The `raw` folder stores original images. The `split` folder stores the experimental partitions. The `preprocessing` folder stores feature-ready grayscale images produced before descriptor extraction.

## Stage 1: Dataset Audit

Notebook: `notebooks/01_data_audit.ipynb`  
Script support: `scripts/regenerate_scene_meta.py`, `scripts/validate_dataset.py`

This stage checks whether scenes are suitable for panorama stitching. It records:

- number of images
- ordered files and reference files
- brightness and contrast
- blur score
- feature counts
- adjacent-pair matches
- RANSAC inliers
- inlier ratio
- reprojection error
- scene difficulty labels

The audit separates scenes into development, test, and failure-analysis groups.

## Stage 2: Preprocessing

Notebook: `notebooks/02_preprocessing.ipynb`  
Script support: `scripts/apply_preprocessing.py`, `project_utils/preprocessing.py`

Preprocessing is performed before the feature/descriptor comparison notebook. The current feature input is:

```text
data/preprocessing/<split>/feature_gray/<scene>/
```

The preprocessing stage can include:

- resize to a controlled maximum width
- grayscale conversion
- optional brightness normalization
- optional Gaussian smoothing
- optional CLAHE

The case-study notebook does not run preprocessing again. It consumes the saved preprocessing outputs.

## Stage 3: Feature and Descriptor Comparison

Notebook: `notebooks/03_feature_extractor.ipynb`  
Batch script: `scripts/extract_features.py`

The current single-scene case study compares:

- SIFT
- ORB
- Harris + ORB
- Harris + HOG
- AKAZE
- BRISK

For one adjacent image pair, the report records:

- keypoints in image A and image B
- raw matches
- good matches after descriptor filtering
- RANSAC inliers
- inlier ratio
- reprojection error
- runtime
- match visualization
- pairwise panorama output

This is a controlled case study, not the full-dataset benchmark.

## Stage 4: OpenCV Stitcher Baseline

Notebook: `notebooks/06_opencv_scene_stitcher.ipynb`

This stage uses OpenCV's built-in Stitcher as a baseline panorama system. It is useful for comparing project-specific methods against a mature implementation.

The baseline output includes:

- stitcher status
- panorama image
- pair diagnostics
- scene summary log
- optional placement overlay

The OpenCV Stitcher stage uses raw scene images with resize-only input control. It should not be mixed with the controlled feature/descriptor case study, which uses preprocessed feature images.

## Stage 5: Manual Projection Previews

Notebook: `notebooks/07_manual_projection_previews.ipynb`

Manual projection previews are used for geometry analysis. They show how adjacent homographies compose across an image chain and how the canvas grows under different projection assumptions.

The manual previews are not presented as the final panorama method. They are evidence for understanding:

- accumulated drift
- weak adjacent links
- excessive canvas growth
- parallax sensitivity
- wide-scene behavior

## Evaluation Criteria

The final report should judge stitching quality using both numbers and images.

Important numeric metrics:

- median Lowe ratio
- Lowe pass rate
- RANSAC inliers
- inlier ratio
- inlier Lowe ratio
- spatial coverage
- mean or median reprojection error
- overlap similarity
- homography sanity
- runtime
- success or failure status

Important visual criteria:

- straight-line alignment
- duplicated objects
- ghosting
- seam visibility
- perspective distortion
- missing content

## Current Experimental Status

Completed:

- dataset organization and metadata structure
- preprocessing pipeline
- single-scene feature/descriptor case study
- OpenCV baseline notebook
- manual projection preview notebook
- batch script for feature extraction

Next:

- run feature extraction across full splits
- add matching and RANSAC batch summaries
- add full panorama generation for selected method combinations
- compare blending methods
- prepare final tables and figures for the report

# Panorama Stitching Pipeline Report - Final Exam Submission

## Objective

This final-exam report builds and evaluates a classical panorama stitching workflow for ordered image sequences with overlap. The main goal is to compare how feature extraction, descriptor quality, geometric verification, and stitching choices affect panorama quality.

The system is designed for:

- indoor and outdoor scenes
- short core sequences of approximately 2 to 6 images
- harder scenes with blur, exposure change, repeated texture, parallax, or weak overlap
- larger scenes used as stress tests after the core pipeline is stable

For showcase runs, the notebooks and scripts now focus on `test` and `failure_analysis`. The `development` split remains in the project as a tuning/development reserve and can still be included explicitly with `--split development` or `--split all`.

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
data/split/
  development/
    scene_XX/
      img_01.jpg
      img_02.jpg
      ...
      meta.json
  test/
    scene_XX/
  failure_analysis/
    scene_XX/

data/preprocessing/
  <split>/
    feature_gray/
      scene_XX/
        img_01.png
        img_02.png
```

The shared dataset is distributed as `data/split` only. Each split contains complete scene folders with images and `meta.json`, so the notebooks do not require `data/raw`. The `preprocessing` folder stores feature-ready grayscale images produced before descriptor extraction.

## Stage 1: Dataset Audit

Notebook: `notebooks/PhamHungSon_15_01_data_audit.ipynb`  
Script support: `scripts/PhamHungSon_15_validate_dataset.py`

This stage audits the scene folders under `data/split/test` and `data/split/failure_analysis` by default. `data/split/development` can still be added for internal experiments. It records:

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

The showcase audit separates scenes into test and failure-analysis groups.

## Stage 2: Preprocessing

Notebook: `notebooks/PhamHungSon_15_02_preprocessing.ipynb`  
Script support: `scripts/PhamHungSon_15_apply_preprocessing.py`, `project_utils/preprocessing.py`

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

Preprocessing mainly helps image-quality problems. It is less useful when the failure comes from scene geometry, missing shared content, or unstable moving objects.

| Failure case | Example scene | Can preprocessing help? | Preprocessing to try | Better response when preprocessing is not enough |
| --- | ---: | --- | --- | --- |
| Low texture | `scene_11` | Partially | CLAHE, gamma normalization, mild sharpening | Use stronger descriptors, add more textured views, or recapture |
| Exposure change | `scene_30` | Often yes | Gamma correction, brightness normalization, CLAHE, histogram matching | Keep exposure more consistent during capture |
| Motion blur | `scene_35` | Slightly | Mild sharpening or unsharp mask, careful denoising | Drop blurry frames or choose sharper captures |
| Moving objects | `scene_35` | Usually no | Mask dynamic regions if available | Remove affected frames or rely on static background matches |
| Repeated patterns | `scene_13` | Usually no | Mild contrast only; avoid over-enhancement | Use stricter RANSAC/geometric checks or inspect matches manually |
| Parallax | `scene_14` | No | Not a preprocessing issue | Use local warping, split the scene, or recapture with rotation around one viewpoint |
| Insufficient overlap | `scene_04` | No | Not fixable by contrast or filtering | Recapture with more overlap, drop the broken transition, or split the scene |
| Sideways scan / translated capture | `scene_14`, `scene_15` | Mostly no | Crop or split only as a diagnostic step | Use a different stitching model or avoid treating it as a pure panorama |
| Global stitch failure | `scene_08` | Sometimes | Standardized resize, CLAHE, brightness normalization | Split the chain or inspect whether OpenCV rejects global camera estimation |
| Wide sweep / long chain | `scene_21` | Partially | Exposure normalization and consistent resize | Reduce chain length, select fewer frames, or stitch in smaller groups |
| Output variation / instability | `scene_30` | Partially | Fixed input order, fixed resize, consistent preprocessing | Repeat runs, compare logs, and keep the most stable configuration |

The case-study notebook does not run preprocessing again. It consumes the saved preprocessing outputs.

## Stage 3: Feature and Descriptor Method Selection

Notebook: `notebooks/PhamHungSon_15_03_feature_extractor.ipynb`  
Batch script: `scripts/PhamHungSon_15_extract_features.py`

The single-scene case study now compares a focused four-method set:

- ORB: fast binary baseline
- AKAZE: stronger binary alternative
- Harris + HOG: interpretable classical gradient baseline
- SIFT: stronger scale-invariant gradient descriptor

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

This is a controlled method-selection case study and also the feature-artifact generation stage. The saved `.npz` files under `data/feature_extract/<split>/<scene>/<descriptor>/features/` become the feature-cache input for the batch matching notebook.

## Stage 4: Batch Feature Matching and RANSAC Evaluation

Notebook: `notebooks/PhamHungSon_15_04_batch_feature_matching.ipynb`

This stage extends the single-pair feature comparison to the showcase scene pairs in `test` and `failure_analysis`. It reads the saved `.npz` keypoint/descriptor artifacts from `data/feature_extract`, validates that their metadata matches the current preprocessing path and descriptor parameters, then runs matching and RANSAC. It reuses the same implemented method pipelines:

- ORB
- AKAZE
- Harris + HOG
- SIFT

For each adjacent pair and method, the notebook records:

- keypoints in both images
- raw KNN matches
- Lowe-ratio good matches
- RANSAC inliers
- inlier ratio
- mean and median reprojection error
- inlier spatial coverage
- runtime
- feature source columns showing whether each side came from the saved cache
- project-level `success`, `hard_valid`, or `failure` status
- status distributions, usable-rate heatmaps, per-scene pair timelines, metric boxplots, and match-preview montage
- failure-analysis pair-status overview across all failure scenes
- weakest adjacent-pair ranking across methods
- method-quality bubble chart by split

The output is saved under:

```text
outputs/batch_feature_matching/
```

The `success`, `hard_valid`, and `failure` flags are project heuristics based on inlier count, inlier ratio, and reprojection error. They are not OpenCV status codes.

## Stage 5: Manual Geometric Stitcher

Notebook: `notebooks/PhamHungSon_15_05_manual_homography_stitcher.ipynb`  
Batch script: `scripts/PhamHungSon_15_manual_homography_stitcher.py`  
Comparison script: `scripts/PhamHungSon_15_compare_manual_opencv_stitching.py`  
Portable script: `scripts/PhamHungSon_15_portable_panorama_pipeline.py`

This stage turns the pair-level matching/RANSAC results into an explicit panorama pipeline. It uses notebook `04` metrics to select pair methods when available, reloads cached features from `data/feature_extract`, recomputes the needed adjacent transforms, chains them to a common anchor image, warps all images into one canvas, and blends overlaps. The current default manual geometry is `affine`, because full projective homographies often over-warp phone panoramas into very large tunnel/hourglass canvases. Full `homography` remains available as a study option.

The portable script can also run on a standalone scene folder without project metadata. It exposes fast, balanced, and quality profiles so a weak device can choose ORB at lower resolution while a stronger device can use higher resolution, more features, SIFT/AKAZE candidates, and feather blending.

The manual-vs-OpenCV comparison reads manual stitcher logs and the OpenCV batch summary, then writes a scene-level table and visual side-by-side panels. In the latest affine-manual showcase run, the comparison contains 15 scenes from `test` and `failure_analysis`: 6 `both_ok`, 2 `both_ok_manual_partial`, 5 `manual_only`, and 2 `manual_partial_only`. The `manual_only` label means OpenCV returned an error but the manual chain saved an output; it does not automatically mean the manual panorama is visually better, because the manual method has no global bundle adjustment, seam optimization, or exposure compensation. The partial labels mean the manual method only used a valid contiguous sub-chain.

Outputs are saved under:

```text
outputs/manual_homography_stitcher/
outputs/manual_homography_stitcher/comparison/
```

## Stage 6: OpenCV Stitcher Baseline

Notebook: `notebooks/PhamHungSon_15_06_opencv_scene_stitcher.ipynb`

This stage uses OpenCV's built-in Stitcher as a baseline panorama system. It is useful for comparing project-specific methods against a mature implementation.

The baseline output includes:

- stitcher status
- panorama image
- pair diagnostics
- scene summary log
- optional placement overlay

The OpenCV Stitcher stage uses the original scene images from `data/split` with resize-only input control. It should not be mixed with the controlled feature/descriptor case study, which uses preprocessed feature images.

### Stage 7: Manual Projection Previews

Notebook: `notebooks/PhamHungSon_15_07_manual_projection_previews.ipynb`

Manual projection previews are used for geometry analysis. They show how adjacent transforms compose across an image chain and how the canvas grows under different projection assumptions. Homography previews are kept specifically as a learning and failure-analysis view.

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

- Dataset organization and metadata structure
- Preprocessing pipeline
- Single-scene feature/descriptor case study
- Batch feature matching/RANSAC notebook
- OpenCV baseline notebook
- Manual projection preview notebook
- Batch script for feature extraction and matching
- Full panorama generation for all splits (Manual Geometry Stitcher with affine default)
- Direct comparative manual-vs-OpenCV status evaluation
- Integration with Android mobile application running offline via Chaquopy
- Built-in real-time Python-to-JS Progress Bridge (FileSystem progress polling)
- Multi-pipeline execution and quantitative CV metric tables in Student Debug mode
- Compilation and verification of debug and standalone Release APKs

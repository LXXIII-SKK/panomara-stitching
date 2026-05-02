# Method Comparison Plan

This document defines the comparison design for the panorama stitching report. It is a planning appendix for experiments; measured results should come from notebooks and batch outputs.

## Comparison Unit

The smallest experimental unit is one adjacent image pair from a scene:

```text
Image A -> Image B
```

For each method, the same preprocessed image pair should be used. This keeps the comparison focused on detector and descriptor behavior rather than differences in preprocessing.

## Feature and Descriptor Pipelines

The current comparison set is:

| Pipeline | Detector | Descriptor | Distance |
| --- | --- | --- | --- |
| SIFT | SIFT | SIFT | L2 |
| ORB | ORB | ORB | Hamming |
| Harris + ORB | Harris corners | ORB | Hamming |
| Harris + HOG | Harris corners | HOG-style patch descriptor | L2 |
| AKAZE | AKAZE | AKAZE binary descriptor | Hamming |
| BRISK | BRISK | BRISK | Hamming |

SURF is excluded from the main comparison because the current OpenCV environment does not provide `cv2.xfeatures2d.SURF_create`.

## Metrics

Each pipeline should report:

| Metric | Purpose |
| --- | --- |
| Keypoints in image A and image B | Detector coverage |
| Raw matches | Descriptor matching candidates |
| Good matches | Matches after ratio filtering |
| RANSAC inliers | Geometrically consistent correspondences |
| Inlier ratio | Fraction of good matches that support the homography |
| Reprojection error | Alignment error after homography estimation |
| Runtime | Computational cost |
| Visual result | Final usability of the panorama |

The most important stitching metrics are RANSAC inliers, inlier ratio, reprojection error, and final visual quality. A method with many keypoints can still fail if its matches are geometrically inconsistent.

## Expected Method Behavior

| Pipeline | Expected Strength | Expected Limitation |
| --- | --- | --- |
| SIFT | Stable gradient descriptor, often strong on texture and scale variation | Slower than binary descriptors |
| ORB | Fast and dense keypoints | More false matches in repeated or low-texture regions |
| Harris + ORB | Clear corner baseline with binary descriptor | Sensitive to weak corners and scale changes |
| Harris + HOG | Interpretable classical baseline | Less distinctive than SIFT/AKAZE for stitching |
| AKAZE | Good balance between stability and speed | Can produce fewer useful points in some scenes |
| BRISK | Binary descriptor with scale handling | Usually less stable than SIFT/AKAZE in hard scenes |

## Case-Study Report Layout

For one selected scene, the report page should show:

1. preprocessed input image pair
2. scene difficulty notes
3. quantitative comparison table
4. RANSAC inlier match visualizations
5. pairwise panorama outputs
6. interpretation of best and weakest methods

The single-scene case study is not the final benchmark. It demonstrates how the comparison will be reported and explains why some methods perform better or worse on a specific scene.

## Full-Dataset Extension

The later batch experiment should run the same descriptor set across:

- development split
- test split
- failure-analysis split

The output structure should remain:

```text
data/feature_extract/<split>/<scene>/<descriptor>/
```

The final report should aggregate:

- average keypoints per image
- average good matches per adjacent pair
- average RANSAC inliers
- average inlier ratio
- average reprojection error
- runtime
- failure rate by scene difficulty

## Report Interpretation

The final explanation should connect numbers to scene conditions. For example:

- low texture can reduce keypoints
- repeated patterns can increase false matches
- blur can make descriptors unstable
- parallax can increase reprojection error
- moving objects can create outliers

The report should avoid ranking methods by keypoint count alone. The stronger conclusion is based on geometric consistency and panorama quality.

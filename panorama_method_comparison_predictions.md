# Panorama Method Comparison Predictions

This note summarizes the **predicted comparison results** for the Panorama Stitching project, based on:
- the suggested comparison directions in [CV_20_Project_2026_English.pdf](./CV_20_Project_2026_English.pdf)
- the current project scope in [panorama_project_pipeline_guide.md](./panorama_project_pipeline_guide.md)
- the characteristics already observed in the current dataset and metadata

Important note:
- this file contains **expected results / hypotheses**
- it is meant to guide the report and experiment design
- it should not replace the final measured comparison tables

## 1. Recommended Comparisons from the Project Suggestion

For the Panorama Stitching project, the suggested comparisons are:
- **Features / descriptors**: `SIFT-like` vs `ORB`
- **Matching**: `Brute Force` vs `FLANN-like`
- **Blending**: `simple overlay` vs `feathering` or `multi-band style approximation`

## 2. Pipeline-Based Explanation from Two Consecutive Images

To explain the comparison more naturally, it is better to start with the smallest unit in the project:

- take **two consecutive images** from one supposed scene
- call them `I_k` and `I_{k+1}`
- assume they have overlap and should be stitchable

The pipeline for this pair is:

1. detect feature points
2. compute descriptors
3. match descriptors
4. reject outliers with RANSAC
5. estimate homography
6. warp one image into the coordinate system of the other
7. blend the overlap region

After this pairwise process works, we extend it to:

```text
I_1 -> I_2 -> I_3 -> ... -> I_n
```

for the full scene.

### 2.1. Start with Two Adjacent Images

Let:

```text
I_k(u, v), I_{k+1}(u, v)
```

be two images from the same scene.

We assume there exists a set of corresponding points:

```text
x_i <-> x'_i
```

with homogeneous coordinates:

```text
x_i  = [u_i,  v_i,  1]^T
x'_i = [u'_i, v'_i, 1]^T
```

If the scene is approximately planar, or the camera rotates around a roughly fixed center, then we model the relation between the two images using:

```text
x'_i ~ H x_i
```

where `H` is a `3 x 3` homography matrix:

```text
      [h11 h12 h13]
H   = [h21 h22 h23]
      [h31 h32 h33]
```

So from the start, the stitching problem is:
- find correct correspondences
- estimate a stable `H`
- warp and blend without obvious artifacts

### 2.2. Step 1: Detect Feature Points

We first find salient points:

```text
K_k     = {p_1, p_2, ..., p_m}
K_{k+1} = {q_1, q_2, ..., q_n}
```

These are locations where local appearance is distinctive enough to be re-found in another image.

Why this matters:
- if the overlap region has too few good keypoints, matching becomes weak
- if the scene is low-texture, repeated, or blurred, the keypoint set becomes less reliable

At this stage:
- `ORB` is expected to be faster
- `SIFT-like` is expected to be more stable in hard scenes

So the first expected difference between methods already appears before matching:
- how many useful keypoints are detected
- how repeatable those keypoints are between the two images

### 2.3. Step 2: Compute Descriptors

For each keypoint, we compute a descriptor vector.

For a `SIFT-like` descriptor:

```text
d_i in R^128
```

For `ORB`:

```text
d_i in {0, 1}^B
```

The goal is:
- descriptors from the same physical scene point should be close
- descriptors from different points should be far apart

If we define:

```text
dist_1 = distance to nearest neighbor
dist_2 = distance to second-nearest neighbor
```

then the ratio test accepts a match if:

```text
dist_1 / dist_2 < tau
```

with `tau` usually around `0.7 - 0.8`.

This is the first place where `SIFT-like` is expected to help more:
- in hard scenes, descriptor ambiguity is higher
- `SIFT-like` often produces a larger gap between the best and second-best candidate

That means:

```text
margin = dist_2 - dist_1
```

is often larger for `SIFT-like` than for `ORB` in difficult conditions.

### 2.4. Step 3: Match Descriptors

Now we create a candidate correspondence set:

```text
M = {(x_i, x'_i)}
```

This set still contains:
- correct matches
- wrong matches
- ambiguous matches

Here the comparison is:
- `Brute Force`: exact nearest-neighbor search
- `FLANN-like`: approximate nearest-neighbor search

For brute force, the best match is:

```text
j* = argmin_j dist(a_i, b_j)
```

computed exactly.

For FLANN-like search, we get:

```text
j_hat ~= argmin_j dist(a_i, b_j)
```

approximately.

So at the matching stage:
- `Brute Force` is expected to be more transparent and stable
- `FLANN-like` is expected to be faster, especially as descriptor count grows

But this stage alone cannot rescue bad scene geometry. It only changes how we search for potential correspondences.

### 2.5. Step 4: Remove Outliers with RANSAC

After matching, we do not trust all correspondences equally.

RANSAC repeatedly samples a minimal subset, estimates a candidate homography, and measures how many matches agree with it.

The inlier set is:

```text
I = {i : ||x'_i - H x_i|| < epsilon}
```

Important derived quantities are:
- number of inliers
- inlier ratio
- reprojection error

If

```text
r = |I| / |M|
```

is the inlier ratio, then the approximate number of RANSAC trials needed for confidence `p` and minimal sample size `s` is:

```text
N = log(1 - p) / log(1 - r^s)
```

This is why descriptor quality matters so much:
- better descriptors often increase `r`
- larger `r` makes RANSAC more stable and efficient

So in the pipeline:
- `SIFT-like` is expected to help **before** RANSAC by giving a better candidate set
- `RANSAC` then converts that better match set into a cleaner geometric model

### 2.6. Step 5: Estimate the Homography

Once outliers are removed, we estimate:

```text
x' ~ Hx
```

for the inlier correspondences.

What we want:
- small reprojection error
- stable `H`
- enough inliers distributed across the overlap region

What can still go wrong:
- if all inliers are concentrated in one small area, `H` may be numerically unstable
- if the scene has parallax, one `H` may not explain all points

This is why some scenes fail no matter which descriptor is used.

If the camera moves translationally and the scene has different depths, then points at different depths may obey different apparent transforms:

```text
x'_near != H x_near
x'_far  != H x_far
```

for a single global `H`.

That is a geometry failure, not a descriptor failure.

### 2.7. Step 6: Warp into a Common Coordinate System

Once `H` is estimated, we warp one image toward the other:

```text
I_{k+1}^{warp}(x) = I_{k+1}(H^{-1}x)
```

or equivalently warp `I_k` into the coordinate system of `I_{k+1}`.

This step determines:
- where the overlap appears on the panorama canvas
- whether straight structures remain aligned
- whether the panorama footprint expands correctly

If `H` is wrong:
- boundaries drift
- objects duplicate
- ghosting appears

This means all later stages depend heavily on the quality of the previous geometric estimation.

### 2.8. Step 7: Blend the Overlap Region

Now suppose both warped images contribute intensity values at the same panorama pixel `p`.

With **overlay**:

```text
P(p) = I_k(p)
```

or whichever image is chosen to overwrite the other.

This creates hard seams when:

```text
I_k(p) != I_{k+1}^{warp}(p)
```

With **feathering**:

```text
              w_k(p) I_k(p) + w_{k+1}(p) I_{k+1}^{warp}(p)
P(p) = ----------------------------------------------------------
                    w_k(p) + w_{k+1}(p)
```

This makes the transition smooth instead of abrupt.

With **multi-band style blending**, we blend multiple frequency bands instead of one single weighted average, which is expected to handle exposure drift more gracefully.

So at the blending stage:
- `overlay` is simpler and better as a diagnostic baseline
- `feathering` is expected to improve seam visibility
- `multi-band style approximation` is expected to improve appearance further, especially under exposure change

### 2.9. Extend from One Pair to a Full Scene

Now we move from:

```text
I_k, I_{k+1}
```

to:

```text
I_1, I_2, ..., I_n
```

For a chain, we estimate pairwise transforms:

```text
H_{1->2}, H_{2->3}, ..., H_{n-1->n}
```

and compose them:

```text
H_{1->3} = H_{2->3} H_{1->2}
H_{1->4} = H_{3->4} H_{2->3} H_{1->2}
```

This is exactly why long scenes become difficult:
- even small pairwise errors accumulate
- drift grows with chain length
- scenes with weak bridge pairs become unstable

So when evaluating methods, the real question is not just:
- which method gives the best pairwise match

but rather:
- which method keeps the full chain stable enough for the final panorama

### 2.10. Where Each Comparison Fits in the Pipeline

This gives a very clean pipeline-based interpretation:

- `ORB vs SIFT-like`
  - mainly affects **Step 1-3**
  - keypoints, descriptors, and match quality

- `Brute Force vs FLANN-like`
  - mainly affects **Step 3**
  - nearest-neighbor retrieval and runtime

- `Overlay vs Feathering vs Multi-band style approximation`
  - mainly affects **Step 7**
  - seam appearance and visual smoothness

So if a scene fails because:
- overlap is too small
- parallax is too large
- camera translation breaks the homography assumption

then changing the descriptor or matcher may help a little, but it will not fully solve the problem.

That is why the project should always interpret method comparison **along the pipeline**, not as isolated black-box switches.

## 3. Predicted Overall Outcome

### Expected winners by criterion

| Comparison | Predicted winner | Main reason |
|---|---|---|
| `ORB` vs `SIFT-like` | `SIFT-like` for quality and robustness | better descriptor distinctiveness and stronger tolerance to scale / illumination / hard texture conditions |
| `Brute Force` vs `FLANN-like` | `Brute Force` for reliability on smaller scenes, `FLANN-like` for speed on larger scenes | BF is exact and deterministic; FLANN-like mainly helps efficiency |
| `Overlay` vs `Feathering` | `Feathering` | reduces seam visibility and makes brightness transitions smoother |
| `Feathering` vs `Multi-band style approximation` | `Multi-band style approximation` for visual quality, `Feathering` for simplicity | multi-band style blending usually handles exposure difference better, but is heavier and more complex |

### Expected overall pipeline ranking

For **best visual quality**:
1. `SIFT-like + BF/FLANN-like + RANSAC + feathering or multi-band style approximation`
2. `ORB + BF + RANSAC + feathering`
3. `ORB + FLANN-like + RANSAC + feathering`
4. `ORB + BF + RANSAC + overlay`

For **best speed-quality tradeoff**:
1. `ORB + BF + RANSAC + feathering`
2. `ORB + FLANN-like + RANSAC + feathering`
3. `SIFT-like + BF/FLANN-like + RANSAC + feathering`
4. `ORB + BF + RANSAC + overlay`

## 3.1. Mathematical Framing of the Panorama Pipeline

To explain why one method is expected to outperform another, it helps to write the panorama problem in mathematical form.

For each matched keypoint pair, we observe:

```text
x_i <-> x'_i
```

where each point is written in homogeneous coordinates:

```text
x_i  = [u_i,  v_i,  1]^T
x'_i = [u'_i, v'_i, 1]^T
```

Under the planar-scene or pure-rotation assumption, the relation between the two images is approximated by a homography:

```text
x'_i ~ H x_i
```

with

```text
      [h11 h12 h13]
H   = [h21 h22 h23]
      [h31 h32 h33]
```

and `~` meaning equality up to scale.

The whole classical stitching pipeline is trying to improve three things:

1. **Descriptor reliability**
   - the correct correspondence should have a smaller descriptor distance than wrong correspondences

2. **Geometric consistency**
   - enough matches should agree with a single `H`

3. **Visual compositing quality**
   - after warping by `H`, overlap regions should blend smoothly

So the comparison between methods is really a comparison of how well they improve:
- the quality of the correspondence set
- the stability of the estimated homography
- the quality of the blended overlap

This gives a clean interpretation:
- descriptor choice mainly affects the **input match set**
- matcher choice mainly affects **search efficiency and match retrieval**
- blending choice mainly affects the **final panorama appearance**

## 3.2. Why Geometry Dominates the Final Outcome

Even if matching is good, stitching can still fail if the scene violates the homography assumption.

If the camera undergoes translation and the scene contains points at different depths, then two points with depths `Z1` and `Z2` will generally not move according to the same 2D projective transform. In practice this means:

```text
x'_near  != H x_near
x'_far   != H x_far
```

for one single global `H`.

That is why some scenes fail for **all** classical variants:
- the descriptor is not the main problem
- the model itself is wrong for the scene geometry

This is also why your report should emphasize:
- method comparisons can improve robustness
- but they cannot fully rescue scenes with strong parallax or viewpoint translation

## 4. Feature / Descriptor Comparison

### 3.1. ORB vs SIFT-like

### Predicted result
- `SIFT-like` is expected to produce **better matching quality and more stable stitching**
- `ORB` is expected to be **faster** and still strong on clean, easy scenes

### Why SIFT-like is expected to be better

`SIFT-like` should outperform `ORB` in harder scenes because:
- it usually creates **more distinctive descriptors**
- it is more tolerant to **scale change**, **rotation**, and moderate **illumination variation**
- it tends to remain more stable when the overlap region has:
  - weak texture
  - repeated structures
  - non-uniform lighting
  - lower local contrast

In panorama stitching, this matters because the pipeline does not only need "some matches". It needs:
- enough good matches
- enough geometrically consistent matches
- stable homography estimation after RANSAC

`SIFT-like` usually helps more in exactly those situations.

### Mathematical intuition: descriptor distinctiveness

Suppose each keypoint is mapped to a descriptor vector.

For a `SIFT-like` descriptor, we can think of:

```text
d_sift in R^128
```

and matching is usually based on Euclidean distance:

```text
dist(d_i, d_j) = ||d_i - d_j||_2
```

For `ORB`, the descriptor is binary:

```text
d_orb in {0,1}^B
```

and matching is based on Hamming distance:

```text
dist(d_i, d_j) = Hamming(d_i, d_j)
```

The ratio test accepts a match if:

```text
dist_1 / dist_2 < tau
```

where:
- `dist_1` is the nearest-neighbor distance
- `dist_2` is the second-nearest-neighbor distance
- `tau` is usually around `0.7 - 0.8`

This means a descriptor is useful not just when it finds a nearest neighbor, but when the nearest neighbor is **significantly better** than the second best.

In repeated textures or weak-texture scenes:
- many descriptors look similar
- `dist_1` and `dist_2` become close
- the ratio test becomes less selective

`SIFT-like` descriptors are expected to perform better because their descriptor space is usually more expressive. Informally:

```text
correct-match margin = dist_2 - dist_1
```

is often larger for `SIFT-like` than for `ORB` in difficult scenes.

That larger margin leads to:
- more accepted good matches
- fewer ambiguous matches
- better input for RANSAC

### Mathematical intuition: why this helps homography estimation

RANSAC estimates `H` from minimal subsets of correspondences and keeps the model with the largest consensus set.

If the inlier set is:

```text
I = {i : ||x'_i - H x_i|| < epsilon}
```

then a better descriptor usually gives:
- larger `|I|`
- higher inlier ratio

Let

```text
r = number_of_inliers / number_of_candidate_matches
```

Then the number of RANSAC iterations needed to reach confidence `p` with sample size `s` is roughly:

```text
N = log(1 - p) / log(1 - r^s)
```

If `r` increases even moderately, `N` can drop a lot, and the estimated model becomes much more stable.

So `SIFT-like` is not just "better matching" in a vague sense. It is expected to:
- increase the inlier ratio `r`
- reduce the chance that RANSAC locks onto a bad model
- improve the stability of `H`

### Why ORB is still important

`ORB` is still a strong baseline because:
- it is much faster than `SIFT-like`
- it is lightweight and practical for notebook-scale experiments
- on easy scenes with:
  - clear overlap
  - sufficient texture
  - little exposure change
  - limited geometric difficulty
  it often gives results close to `SIFT-like`

### Mathematical intuition: why ORB is faster

The main speed advantage of `ORB` comes from two places:

1. **Binary descriptors**
   - Hamming distance can be computed very efficiently with bit operations

2. **Cheaper detection/description**
   - `ORB` was designed as a lightweight alternative to heavier floating-point descriptors

Very roughly, matching cost behaves like:

```text
cost ~ number_of_descriptors x descriptor_dimension
```

but with a much smaller constant for binary bitwise comparison than for floating-point `L2` computation.

So even if both methods produce usable panoramas on easy scenes, `ORB` is expected to win clearly in runtime.

### Predicted report conclusion

> `SIFT-like` is expected to outperform `ORB` in robustness and final stitch quality, especially on hard scenes with illumination variation, lower texture, or repeated patterns. `ORB`, however, is expected to provide the best efficiency and a strong baseline on clean scenes, making it the most practical method when runtime matters more than maximum robustness.

## 5. Matcher Comparison

### 4.1. Brute Force vs FLANN-like

### Predicted result
- `Brute Force` is expected to be **slightly more reliable** on small and medium scenes
- `FLANN-like` is expected to be **faster**, especially for larger scenes and heavier descriptors
- the quality difference is expected to be **smaller than the feature-descriptor difference**

### Why Brute Force may be better in reliability

`Brute Force` matching compares descriptors directly and exactly:
- it is simpler
- it is more deterministic
- it is easier to debug
- it tends to behave predictably in small and medium panorama datasets

For a course project, this matters because:
- easier tuning helps reproducibility
- debugging failed scenes is more straightforward
- the accuracy benefit of approximate indexing is often limited when the scene count is not huge

### Mathematical intuition: exact nearest neighbor vs approximate nearest neighbor

Suppose image A has descriptors:

```text
D_A = {a_1, a_2, ..., a_n}
```

and image B has descriptors:

```text
D_B = {b_1, b_2, ..., b_m}
```

For each `a_i`, brute-force matching computes:

```text
j* = argmin_j dist(a_i, b_j)
```

exactly by checking all `m` candidates.

This makes the search complexity roughly:

```text
O(nm)
```

per pair, ignoring descriptor dimension constants.

`FLANN-like` methods approximate this nearest-neighbor search:

```text
j_hat ~= argmin_j dist(a_i, b_j)
```

The gain is speed, but the tradeoff is that:
- the returned nearest neighbor may not be the exact best one
- the returned second neighbor may also shift
- therefore the ratio test outcome can change

That is why `FLANN-like` is expected to change runtime more strongly than final accuracy:
- when descriptor space is well separated, approximate search still works fine
- when descriptor space is ambiguous, geometry is already difficult and approximate search alone cannot solve it

### Why FLANN-like may still be useful

`FLANN-like` is expected to be better when:
- the descriptor set becomes large
- the scene contains many images
- runtime becomes a visible bottleneck

Its advantage is mainly:
- faster nearest-neighbor search
- better scalability on larger matching workloads

However, `FLANN-like` is **not expected to fix fundamentally bad scenes**, such as scenes with:
- severe parallax
- insufficient overlap
- inconsistent viewpoint motion
- globally unstable geometry

Those are geometry problems, not nearest-neighbor search problems.

### Complexity interpretation

The practical comparison can be summarized like this:

- `Brute Force`
  - exact
  - simpler
  - usually more reproducible
  - higher computational cost

- `FLANN-like`
  - approximate
  - often faster for large descriptor sets
  - less transparent when debugging edge cases

So the predicted advantage of `FLANN-like` grows when:

```text
number_of_images x descriptors_per_image
```

becomes large enough that matching dominates runtime.

### Predicted report conclusion

> The difference between `Brute Force` and `FLANN-like` is expected to appear mainly in runtime rather than final stitching quality. `Brute Force` should remain a reliable and transparent baseline for small-to-medium panorama scenes, while `FLANN-like` should become more attractive as the number of images or descriptor volume increases.

## 6. Blending Comparison

### 5.1. Overlay vs Feathering

### Predicted result
- `Feathering` is expected to produce **better-looking panoramas**
- `Overlay` is expected to be the **fastest and simplest**, but usually with worse seam quality

### Why feathering is expected to be better

`Overlay` simply places warped images on top of each other. This tends to create:
- visible seams
- abrupt exposure transitions
- harder boundaries between image regions

`Feathering` improves this by:
- smoothly weighting pixels across overlap regions
- reducing sharp transitions at boundaries
- producing a more coherent panorama appearance

This is especially useful when:
- exposure changes slightly between images
- overlap regions are large enough for smooth weighting
- alignment is already reasonably correct

### Mathematical intuition: overlay vs weighted blending

Let `I_k` be the `k`-th warped image on the panorama canvas, and let `p` be a pixel in the panorama domain.

With **simple overlay**, the panorama is effectively:

```text
P(p) = I_k(p)
```

for whichever image is written last or selected by a simple rule.

This creates discontinuities because at a seam:

```text
P_left(p)  = I_a(p)
P_right(p) = I_b(p)
```

and if `I_a(p)` and `I_b(p)` differ in brightness or color, the boundary becomes visible.

With **feathering**, we use weighted averaging:

```text
            sum_k w_k(p) I_k(p)
P(p) = ------------------------------
          sum_k w_k(p)
```

where `w_k(p)` is a spatial weight, often larger near the center of an image and smaller near its boundary.

This reduces seam visibility because the transition between images becomes continuous rather than abrupt.

If two overlapping images differ slightly:

```text
I_b(p) = I_a(p) + delta(p)
```

then overlay gives a hard jump of size about `delta(p)` at the boundary, while feathering spreads that transition over a region.

So feathering is expected to reduce the *magnitude of visible seam discontinuity*, even if it does not change the geometry.

### Why overlay is still useful

`Overlay` is still useful as a baseline because:
- it is simple to implement
- it makes alignment errors easier to see
- it shows whether blending is hiding or not hiding geometric problems

In fact, `overlay` is often a good **diagnostic baseline**, even if it is not the best final output method.

### Important limitation

Neither `overlay` nor `feathering` can fix bad alignment. If homography is wrong:
- ghosting remains
- duplicated edges remain
- object mismatch remains

So blending improves appearance only when geometry is already acceptable.

### Mathematical intuition: why blending cannot fix ghosting

Suppose an edge from image A lands at pixel `p`, while the corresponding edge from image B lands at `p + Delta`.

Then even with perfect blending weights:

```text
P(p) ~ alpha I_a(p) + (1 - alpha) I_b(p)
```

the two structures are still spatially misaligned because the problem is not intensity mismatch, but position mismatch.

In other words:
- blending solves a **photometric transition** problem
- it does not solve a **geometric registration** problem

### Predicted report conclusion

> `Feathering` is expected to outperform direct `overlay` visually by reducing seam visibility and producing smoother transitions between images. However, it should be emphasized that blending improves appearance but does not correct geometric alignment errors. Therefore, blending quality must be interpreted together with alignment quality rather than in isolation.

## 7. Optional Blending Extension

### 6.1. Feathering vs Multi-band Style Approximation

### Predicted result
- `Multi-band style approximation` is expected to produce the best visual blending
- `Feathering` is expected to remain the best compromise between quality and implementation complexity

### Why multi-band style blending may be better

Multi-band style blending is usually stronger when:
- exposure differs noticeably between images
- there are both low-frequency brightness changes and high-frequency texture details
- seam reduction must remain good across multiple spatial scales

It often gives:
- softer seams
- better brightness transitions
- less obvious region boundaries

### Mathematical intuition: why multi-band blending can outperform feathering

Feathering uses one weight field over the whole image. That means:
- low-frequency brightness differences and high-frequency texture details are handled in the same way

Multi-band blending separates the image into frequency bands.

If we write a Laplacian pyramid of image `I_k` as:

```text
L_k^0, L_k^1, ..., L_k^L
```

and a Gaussian pyramid of the blending weights as:

```text
G_k^0, G_k^1, ..., G_k^L
```

then blended reconstruction is roughly:

```text
P = sum_{ell=0}^L Reconstruct( sum_k G_k^ell * L_k^ell )
```

This helps because:
- low-frequency brightness changes can be blended smoothly
- high-frequency edges can stay sharper

So multi-band style blending is expected to handle:
- exposure shift
- broad illumination drift
- large smooth seam transitions

better than simple feathering.

### Why feathering may still be the better course-project choice

Even if multi-band style blending looks better, `feathering` may remain the more practical choice because:
- it is easier to implement and explain
- it is lighter computationally
- it is easier to debug
- it is often sufficient for classical panorama baselines

### Predicted report conclusion

> If implemented, `multi-band style approximation` is expected to provide the strongest visual blending quality, especially under exposure changes, but `feathering` is likely to remain the more practical method for a classical course project due to its lower complexity and clearer explainability.

## 8. Expected Behavior by Scene Type

### 7.1. Success / core scenes

Expected outcome:
- both `ORB` and `SIFT-like` should usually succeed
- the most visible difference will likely come from `overlay` vs `feathering`
- matcher choice will likely affect runtime more than panorama quality

### 7.2. Hard-valid scenes

Expected outcome:
- `SIFT-like` should outperform `ORB` more clearly
- `feathering` will improve seam appearance, but not rescue poor alignment
- `FLANN-like` may help speed, but not enough to change hard scenes into easy scenes

### 7.3. Failure scenes

Expected outcome:
- some scenes will remain failures for all classical combinations
- this is especially true when failure is caused by:
  - severe parallax
  - insufficient overlap
  - large viewpoint translation
  - unstable long chains

This is an important expected finding:
- not all failures come from a weak descriptor
- many failures come from **violated geometric assumptions**

### Mathematical interpretation of failure scenes

The most important failure mode is when there is no single homography `H` that explains the overlap well enough.

For a valid scene, we want the residual:

```text
e_i = ||x'_i - H x_i||
```

to remain small for many correspondences.

For parallax-heavy scenes, the residual pattern becomes depth-dependent:

```text
e_near >> e_far
```

or the opposite, depending on viewpoint motion.

This leads to:
- lower inlier ratio
- more unstable RANSAC
- stronger ghosting after warping

No descriptor or matcher can fully fix this, because the violation happens after matching, at the level of the geometric model itself.

### 7.4. Large scenes

Expected outcome:
- `FLANN-like` may become more attractive because matching cost grows
- `SIFT-like` may still improve robustness, but runtime will increase
- stability and drift will likely become bigger problems than local pair quality alone

## 9. Predicted Final Discussion for the Report

The predicted final discussion can be written in this direction:

> Among classical feature-based panorama pipelines, the descriptor choice is expected to have the largest effect on robustness, the matcher choice is expected to have the largest effect on efficiency, and the blending choice is expected to have the largest effect on perceived visual quality. `SIFT-like` should perform best on difficult scenes because it offers more stable and distinctive matching, while `ORB` should remain the fastest and most practical baseline for clean scenes. `Brute Force` and `FLANN-like` are expected to differ mainly in runtime, with limited impact on the final geometry when the scene itself is already weak. In blending, `feathering` should outperform simple overlay visually, while multi-band style blending may further improve seam quality at the cost of added complexity. However, across all combinations, scenes with severe parallax, weak overlap, or unstable viewpoint motion are still expected to fail because the main limitation is geometric rather than purely descriptor-based.

## 9.1. Predicted Metric-Level Effects

If the experiments are run correctly, the predicted changes should appear in metrics like this:

| Method change | Expected metric effect |
|---|---|
| `ORB -> SIFT-like` | higher `good_matches`, higher `inliers`, higher `inlier_ratio`, lower reprojection error on hard scenes |
| `BF -> FLANN-like` | similar `inlier_ratio`, similar reprojection error, lower runtime on large scenes |
| `Overlay -> Feathering` | similar geometric metrics, lower seam visibility, better qualitative score |
| `Feathering -> Multi-band style approximation` | similar geometric metrics, lower seam visibility under exposure differences, slightly higher runtime |

This table is important because it clarifies **where** each method is expected to help:
- descriptor improvements should show up before and during homography estimation
- matcher improvements should show up mainly in runtime
- blending improvements should show up after warping, in visual quality

## 10. Short Version for a Report Table

| Stage | Compared methods | Predicted better method | Why |
|---|---|---|---|
| Feature / descriptor | ORB vs SIFT-like | SIFT-like | more distinctive and robust descriptors, better on hard scenes |
| Matching | Brute Force vs FLANN-like | BF for reliability, FLANN-like for speed | BF is exact and easier to debug; FLANN-like scales better |
| Blending | Overlay vs Feathering | Feathering | smoother transitions, lower seam visibility |
| Optional blending | Feathering vs Multi-band style approximation | Multi-band style approximation | better seam handling across different frequency scales |

## 11. Practical Recommendation for This Project

If the goal is a fair, strong classical comparison, the recommended experiment set is:
- `ORB + BF + RANSAC + overlay`
- `ORB + BF + RANSAC + feathering`
- `ORB + FLANN-like + RANSAC + feathering`
- `SIFT-like + BF/FLANN-like + RANSAC + feathering`

This set is good because it separates:
- descriptor effect
- matcher effect
- blending effect

and matches the course-project suggestion closely.

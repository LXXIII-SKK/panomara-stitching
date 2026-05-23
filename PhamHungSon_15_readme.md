# PhamHungSon_15 - Huong dan chay pipeline Panorama Stitching cho bao cao thi cuoi ky

File nay huong dan cach chay goi nop bai thi cuoi ky trong thu muc `final_report`. Du lieu anh khong nam san trong goi nop bai; nguoi cham tai du lieu tu link Google Drive, giai nen vao `final_report/data`, sau do chay script va notebook theo thu tu ben duoi.

## 1. Thong tin thanh vien

| Thanh vien | STT | Phan viec |
| --- | ---: | --- |
| Pham Hung Son | 15 | To chuc du lieu, audit dataset, tien xu ly anh, trich xuat dac trung, so sanh descriptor, chay OpenCV Stitcher baseline, tao preview projection va tong hop bao cao |

## 2. Tai va dat du lieu

Link du lieu:

```text
https://drive.google.com/drive/folders/1MLb_hScN_5qY6bSoVM65Q9w8KLP_sLU9?usp=drive_link
```

Sau khi tai, giai nen/copy thu muc `data` vao trong `final_report` sao cho cau truc co dang:

```text
final_report/
  data/
    split/
      development/
        scene_02/
          img_01.jpg
          img_02.jpg
          ...
          meta.json
      test/
        scene_01/
          img_01.jpg
          ...
          meta.json
      failure_analysis/
        scene_04/
          img_01.jpg
          ...
          meta.json
```

Luu y: dataset nop bai chi can `data/split`. Khong can `data/raw`.

Quy mo du lieu hien tai:

```text
Tong cong:          29 scenes, 173 images
development split: 14 scenes, 93 images
test split:         6 scenes, 36 images
failure_analysis:   9 scenes, 44 images
```

## 3. Cau truc goi nop bai

```text
final_report/
  PhamHungSon_15_readme.md
  requirements.txt
  panorama_pipeline_report.md
  panorama_method_comparison_report.md
  Computer_Vision___Final_Report/
    main.tex
    main.pdf
    chapter/

  notebooks/
    PhamHungSon_15_01_data_audit.ipynb
    PhamHungSon_15_02_preprocessing.ipynb
    PhamHungSon_15_03_feature_extractor.ipynb
    PhamHungSon_15_04_batch_feature_matching.ipynb
    PhamHungSon_15_05_manual_homography_stitcher.ipynb
    PhamHungSon_15_06_opencv_scene_stitcher.ipynb
    PhamHungSon_15_07_manual_projection_previews.ipynb

  scripts/
    PhamHungSon_15_validate_dataset.py
    PhamHungSon_15_apply_preprocessing.py
    PhamHungSon_15_extract_features.py
    PhamHungSon_15_manual_homography_stitcher.py
    PhamHungSon_15_compare_manual_opencv_stitching.py
    PhamHungSon_15_portable_panorama_pipeline.py
    PhamHungSon_15_generate_opencv_overlay.py

  project_utils/
    panorama_dataset.py
    preprocessing.py
```

Thu muc `data/preprocessing`, `data/feature_extract`, va `outputs` se duoc sinh ra khi chay script/notebook.

## 4. Moi truong chay

Project duoc chay voi Conda environment:

```text
image_recognition
```

Tu thu muc `final_report`, co the cai dat cac goi can thiet bang:

```powershell
conda activate image_recognition
pip install -r requirements.txt
```

Neu khong activate environment, co the dung `conda run` nhu cac lenh ben duoi.

Di chuyen vao thu muc nop bai:

```powershell
cd C:\Users\PC\Downloads\Project\final_report
```

Kiem tra nhanh:

```powershell
conda run -n image_recognition python --version
conda run -n image_recognition python -c "import cv2; print(cv2.__version__)"
```

## 5. Kich ban chay script

### Buoc 1: Kiem tra dataset

Lenh mac dinh se kiem tra toan bo `data/split`:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_validate_dataset.py
```

Neu muon kiem tra tung split:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_validate_dataset.py --root data\split\development
conda run -n image_recognition python scripts\PhamHungSon_15_validate_dataset.py --root data\split\test
conda run -n image_recognition python scripts\PhamHungSon_15_validate_dataset.py --root data\split\failure_analysis
```

Neu muon kiem tra file anh co doc duoc hay khong:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_validate_dataset.py --verify-images
```

### Buoc 2: Tao anh tien xu ly cho feature extraction

Pipeline tien xu ly chinh:

```text
resize -> grayscale -> Gaussian blur -> CLAHE
```

Chay cho cac split showcase:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_apply_preprocessing.py --split test --ordered-only --profile baseline --output-kind gray --overwrite
conda run -n image_recognition python scripts\PhamHungSon_15_apply_preprocessing.py --split failure_analysis --ordered-only --profile baseline --output-kind gray --overwrite
```

Output:

```text
data/preprocessing/<split>/feature_gray/<scene>/img_XX.png
data/preprocessing/<split>/preprocess_manifest.csv
```

### Buoc 3: Trich xuat descriptor bang script

Chay tat ca descriptor cho scene case-study:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_extract_features.py --split failure_analysis --scene scene_14 --descriptor all --overwrite
```

Chay mot descriptor cho ca split:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_extract_features.py --split test --descriptor SIFT --overwrite
```

Output:

```text
data/feature_extract/<split>/<scene>/<descriptor>/
```

Descriptor ho tro trong project:

```text
ORB
AKAZE
HARRIS_HOG
SIFT
```

Nhom da rut gon thuc nghiem chinh con 4 method dai dien: ORB la binary baseline nhanh, AKAZE la binary descriptor manh hon, Harris + HOG la baseline gradient de giai thich, va SIFT la descriptor gradient manh hon.

## 6. Thu tu chay notebook

Nen chay theo thu tu sau:

1. `notebooks/PhamHungSon_15_01_data_audit.ipynb`

   Audit dataset trong `data/split`: so luong anh, metadata, blur, contrast, keypoints, matches, RANSAC inliers, reprojection error, overlap va cac dau hieu failure.

2. `notebooks/PhamHungSon_15_02_preprocessing.ipynb`

   Trinh bay va minh hoa preprocessing. Notebook co cell demo tung buoc: resize, grayscale, Gaussian blur, CLAHE, global histogram equalization, denoising va sharpening. Notebook nay mang tinh giai thich; de sinh file preprocessing cho pipeline, dung script o Buoc 2.

3. `notebooks/PhamHungSon_15_03_feature_extractor.ipynb`

   Case-study so sanh feature detector/descriptor tren mot scene cu the. Notebook nay doc anh da tien xu ly tu `data/preprocessing/...`, vi vay can chay Buoc 2 truoc.

   Notebook cung co phan minh hoa cach cac method hoat dong: gradient/DoG/Harris/FAST cues, keypoint overlay cho tung pipeline, descriptor-vector heatmap/bit map, va HOG patch orientation.

   Cau hinh khuyen nghi trong notebook:

   ```python
   SPLIT_NAME = "failure_analysis"
   INPUT_KIND = "feature_gray"
   SCENE_ID = "scene_14"
   PAIR_INDEX = 1
   ```

   Notebook so sanh cac pipeline:

   ```text
   ORB
   AKAZE
   Harris + HOG
   SIFT
   ```

4. `notebooks/PhamHungSon_15_04_batch_feature_matching.ipynb`

   Doc feature da luu tu `data/feature_extract/<split>/<scene>/<descriptor>/`, sau do chay descriptor matching, Lowe ratio test, homography/RANSAC va tinh reprojection error tren cac cap anh lien ke. Notebook luu `pair_metrics.csv`, `scene_method_summary.csv`, `method_summary.csv`, `weak_pair_summary.csv`, match preview, heatmap, timeline, boxplot, failure-analysis overview, weak-pair chart, method-quality bubble chart va montage vao `outputs/batch_feature_matching/`.

   Mac dinh notebook chay cac split showcase (`test` va `failure_analysis`):

   ```python
   SPLITS_TO_RUN = ["test", "failure_analysis"]
   RUN_FULL_BATCH = True
   ```

   Neu muon notebook 04 chay ca development, them `development` vao `SPLITS_TO_RUN` hoac dung script voi `--split all`. Neu chi chay nhanh mot vai scene, doi `RUN_FULL_BATCH = False` va chinh `SELECTED_SPLIT`, `SELECTED_SCENES`.

5. `notebooks/PhamHungSon_15_05_manual_homography_stitcher.ipynb`

   Tao panorama bang pipeline thu cong: doc feature cache tu `data/feature_extract`, dung ket qua Notebook 04 de chon method cho tung cap anh, uoc luong homography bang RANSAC, chain homography, warp anh len canvas chung va blend overlap. Notebook nay giai thich ro cach stitch anh thay vi chi goi OpenCV Stitcher, dong thoi co phan so sanh output manual voi OpenCV Stitcher.

6. `notebooks/PhamHungSon_15_06_opencv_scene_stitcher.ipynb`

   Chay baseline OpenCV Stitcher tren anh goc trong `data/split`. Notebook dung `cv2.Stitcher` nguyen ban, chi resize input de kiem soat kich thuoc anh, khong them preprocessing/enhancement ngoai pipeline noi bo cua OpenCV.

7. `notebooks/PhamHungSon_15_07_manual_projection_previews.ipynb`

   Minh hoa homography chain, full-canvas projection va cylindrical projection de phan tich drift, canvas growth va loi hinh hoc. Notebook nay dung de giai thich, khong thay the OpenCV Stitcher baseline.

## 7. Output quan trong

Sau khi chay pipeline, cac ket qua se nam trong:

```text
outputs/audit/
outputs/case_study/
outputs/batch_feature_matching/
outputs/manual_homography_stitcher/
outputs/manual_homography_stitcher/comparison/
outputs/openCV/
outputs/manual/
data/preprocessing/
data/feature_extract/
```

Nhung thu muc nay la output sinh ra khi chay, khong can co san trong goi nop bai.

## 8. Loi thuong gap

Neu notebook 03 bao khong tim thay `data/preprocessing/...`, hay chay lai Buoc 2.

Chay manual geometry stitcher cho mot scene:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_manual_homography_stitcher.py --split test --scene scene_01 --profile balanced --method auto --feature-source cache --use-batch-metrics --overwrite
```

Chay manual geometry stitcher cho toan bo split database:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_manual_homography_stitcher.py --split test --split failure_analysis --profile balanced --method auto --feature-source cache --use-batch-metrics --allow-partial --overwrite
```

So sanh manual stitcher voi OpenCV Stitcher:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_compare_manual_opencv_stitching.py --side-by-side-limit 16
```

Chay portable one-scene pipeline, phu hop de dua logic vao ung dung Android/Python bridge:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_portable_panorama_pipeline.py --scene-folder data\split\test\scene_01 --output outputs\manual_homography_stitcher\portable_scene_01_fast.jpg --profile fast --method ORB --blend-mode overwrite --work-width 960
```

Neu notebook 06 khong tao overlay duoc, hay chay cell stitch scene truoc de tao panorama trong `outputs/openCV/panoramas`, sau do moi chay cell overlay.

Neu `cv2.SIFT_create` khong co, can cai `opencv-contrib-python` trong environment. Project van co cac descriptor khac nhu ORB, AKAZE va Harris + HOG.

Neu duong dan du lieu sai, kiem tra lai cau truc:

```text
final_report/data/split/development
final_report/data/split/test
final_report/data/split/failure_analysis
```

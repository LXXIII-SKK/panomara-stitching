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
Tong cong:          31 scenes, 167 images
development split: 12 scenes, 63 images
test split:         6 scenes, 36 images
failure_analysis:  13 scenes, 68 images
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

### Buoc 2: Chay data audit truoc khi preprocessing

Mo notebook 01 va chay tat ca cell:

```text
notebooks/PhamHungSon_15_01_data_audit.ipynb
```

Notebook nay sinh cac file audit sau:

```text
outputs/audit/image_metrics.csv
outputs/audit/pair_metrics.csv
outputs/audit/scene_metrics.csv
```

File `outputs/audit/image_metrics.csv` co cot `preprocess_recommendations`. Buoc preprocessing ben duoi dung `--profile audit_auto` de doc cac khuyen nghi nay. Neu bo qua notebook 01, `audit_auto` se fallback ve pipeline baseline.

### Buoc 3: Tao anh tien xu ly cho toan bo split

Pipeline tien xu ly chinh:

```text
resize -> grayscale -> optional auto-gamma -> Gaussian blur -> CLAHE -> optional sharpening
```

Chay rieng tung split de output dung layout ma cac buoc sau mong doi:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_apply_preprocessing.py --split development --ordered-only --profile audit_auto --output-kind both --overwrite
conda run -n image_recognition python scripts\PhamHungSon_15_apply_preprocessing.py --split test --ordered-only --profile audit_auto --output-kind both --overwrite
conda run -n image_recognition python scripts\PhamHungSon_15_apply_preprocessing.py --split failure_analysis --ordered-only --profile audit_auto --output-kind both --overwrite
```

Khong dung `--split split` cho flow chinh neu cac buoc sau doc `data/preprocessing/<split>/...`, vi cach do ghi vao scope `data/preprocessing/split/...`.

Rieng `failure_analysis/scene_18` la scene toi. De tao ban de nhin ro hon, co the chay them mot pass manh hon sau lenh failure_analysis:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_apply_preprocessing.py --split failure_analysis --scene scene_18 --ordered-only --profile audit_auto --output-kind both --gamma-min 0.35 --target-brightness 155 --clahe-clip-limit 3.0 --overwrite
```

Output:

```text
data/preprocessing/<split>/feature_gray/<scene>/img_XX.png
data/preprocessing/<split>/color_enhanced/<scene>/img_XX.png
data/preprocessing/<split>/preprocess_manifest.csv
```

`feature_gray` la input cho feature extraction. `color_enhanced` chi de xem/debug/report, khong bat buoc cho matching.

### Buoc 4: Trich xuat descriptor cho toan bo split

Chay tat ca descriptor cho tat ca split hien tai:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_extract_features.py --split all --descriptor all --overwrite
```

Neu chi muon test nhanh mot scene:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_extract_features.py --split failure_analysis --scene scene_32 --descriptor all --overwrite
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

### Buoc 5: Chay batch matching/RANSAC

Mo notebook 04 va chay tat ca cell:

```text
notebooks/PhamHungSon_15_04_batch_feature_matching.ipynb
```

Notebook 04 doc feature cache tu `data/feature_extract/...` va sinh:

```text
outputs/batch_feature_matching/pair_metrics.csv
outputs/batch_feature_matching/scene_method_summary.csv
outputs/batch_feature_matching/method_summary.csv
outputs/batch_feature_matching/plots/
```

`pair_metrics.csv` duoc manual stitcher dung khi chay voi `--use-batch-metrics`.

### Buoc 6: Chay OpenCV Stitcher baseline

Mo notebook 06 va chay tat ca cell:

```text
notebooks/PhamHungSon_15_06_opencv_scene_stitcher.ipynb
```

Notebook 06 sinh:

```text
outputs/openCV/logs/batch_summary_panorama.csv
outputs/openCV/logs/scene_XX_opencv_panorama_summary.json
outputs/openCV/panoramas/
```

Nen chay OpenCV baseline truoc khi chay comparison trong Buoc 7, vi comparison can `batch_summary_panorama.csv`.

### Buoc 7: Chay manual geometry stitcher va comparison

Chay manual stitcher cho cac split showcase:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_manual_homography_stitcher.py --split test --split failure_analysis --profile balanced --method auto --feature-source cache --use-batch-metrics --allow-partial --overwrite
```

Sau do so sanh manual voi OpenCV:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_compare_manual_opencv_stitching.py --split test --split failure_analysis --side-by-side-limit 30
```

Tong hop failure-analysis theo problem labels:

```powershell
conda run -n image_recognition python scripts\PhamHungSon_15_summarize_failure_problem_stitching.py
```

Output:

```text
outputs/manual_homography_stitcher/manual_homography_manifest.csv
outputs/manual_homography_stitcher/logs/
outputs/manual_homography_stitcher/panoramas/
outputs/manual_homography_stitcher/comparison/
outputs/failure_problem_analysis/
```

### Buoc 8: Notebooks giai thich/optional

Sau khi cac output chinh da co, co the mo cac notebook giai thich:

```text
notebooks/PhamHungSon_15_02_preprocessing.ipynb
notebooks/PhamHungSon_15_03_feature_extractor.ipynb
notebooks/PhamHungSon_15_05_manual_homography_stitcher.ipynb
notebooks/PhamHungSon_15_07_manual_projection_previews.ipynb
```

Notebook 02 va 03 chu yeu minh hoa ly thuyet/preprocessing/feature descriptors. Notebook 05 co the chay manual demo trong notebook, nhung lenh script o Buoc 7 la cach nhanh va day du nhat de sinh output manual cho tat ca scene showcase.

## 6. Tom tat thu tu chay day du

Neu nguoi cham muon chay lai tu dau, thu tu ngan gon la:

```text
1. Validate dataset
2. Run Notebook 01 data audit
3. Run preprocessing commands for development/test/failure_analysis
4. Run feature extraction with --split all --descriptor all
5. Run Notebook 04 batch matching
6. Run Notebook 06 OpenCV baseline
7. Run manual stitcher script
8. Run manual-vs-OpenCV comparison script
9. Run failure-problem summary script
10. Optional: run Notebook 02/03/05/07 for explanation figures
```

## 7. Output quan trong

Sau khi chay pipeline, cac ket qua se nam trong:

```text
outputs/audit/
outputs/case_study/
outputs/batch_feature_matching/
outputs/manual_homography_stitcher/
outputs/manual_homography_stitcher/comparison/
outputs/openCV/
outputs/failure_problem_analysis/
data/preprocessing/
data/feature_extract/
```

Nhung thu muc nay la output sinh ra khi chay, khong can co san trong goi nop bai.

## 8. Loi thuong gap

Neu notebook 03 bao khong tim thay `data/preprocessing/...`, hay chay lai Buoc 3.

Neu `--profile audit_auto` khong ap dung khuyen nghi preprocessing, kiem tra xem da chay notebook 01 va co file nay chua:

```text
outputs/audit/image_metrics.csv
```

Neu comparison bao `manual_missing` hoac `opencv_only_or_manual_missing`, nghia la chua chay manual stitcher script cho day du scene showcase. Chay lai lenh manual stitcher o Buoc 7.

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
conda run -n image_recognition python scripts\PhamHungSon_15_compare_manual_opencv_stitching.py --split test --split failure_analysis --side-by-side-limit 30
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

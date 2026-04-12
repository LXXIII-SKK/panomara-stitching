# Panorama Stitching Project

Repo này dùng để tổ chức dataset panorama, audit overlap/matching, và thử nghiệm stitching theo 2 hướng:
- `OpenCV Stitcher` để có baseline nhanh
- `manual projection previews` để debug footprint, coverage, và các scene quét rộng

## Current Structure

```text
data/raw/
  scene_XX/
    img_001.jpg
    img_002.jpg
    ...
    meta.json

notebooks/
  01_data_audit.ipynb
  06_opencv_scene_stitcher.ipynb
  07_manual_projection_previews.ipynb

project_utils/
  panorama_dataset.py

scripts/
  rename_scene_images.py
  regenerate_scene_meta.py
  validate_dataset.py
  generate_opencv_overlay.py
```

`outputs/` và `notebooks/outputs/` đang được ignore trong Git. Chúng vẫn được notebook/script tạo ra cục bộ khi cần.

## Recommended Workflow

### 1. Rename images inside a scene

Đổi tên ảnh theo chuẩn `img_xxx` dựa trên suffix số cuối của filename gốc.

```powershell
python scripts/rename_scene_images.py data/raw/scene_22 --dry-run
python scripts/rename_scene_images.py data/raw/scene_22
```

Ví dụ:
- `20260411_140944_001.jpg -> img_001.jpg`
- `20260411_140944_030.jpg -> img_030.jpg`

### 2. Regenerate metadata for all scenes

Script này:
- đọc `ordered_files` theo `meta.json` nếu có
- fallback sang thứ tự `img_001`, `img_002`, ...
- tính lại `image_stats`, `pair_audit`, `audit_summary`
- chạy `cv2.Stitcher` nhiều lần để tạo `stability_check`
- giữ lại các trường thủ công như `category`, `difficulty`, `issues`, `notes`

```powershell
python scripts/regenerate_scene_meta.py
python scripts/regenerate_scene_meta.py --scene scene_16 --stability-runs 12
```

### 3. Validate dataset consistency

Script này kiểm:
- `scene_id`
- `ordered_files`
- `reference_files`
- `num_images`
- `image_stats`
- `pair_audit`
- `pair_label_counts`
- `stability_check`
- cảnh báo các scene có `category` lệch với `stitcher_status`

```powershell
python scripts/validate_dataset.py
python scripts/validate_dataset.py --verify-images
```

### 4. Run notebooks

Audit dataset:

```powershell
jupyter notebook notebooks/01_data_audit.ipynb
```

OpenCV baseline:

```powershell
jupyter notebook notebooks/06_opencv_scene_stitcher.ipynb
```

Manual debug previews:

```powershell
jupyter notebook notebooks/07_manual_projection_previews.ipynb
```

## Notebook Roles

### `01_data_audit.ipynb`
- audit quality của từng scene
- xem brightness, blur, keypoints, matches, inliers
- dùng để tìm scene tốt, scene khó, scene lỗi

### `06_opencv_scene_stitcher.ipynb`
- chạy `cv2.Stitcher`
- xuất panorama vào `outputs/openCV/panoramas/`
- xuất log vào `outputs/openCV/logs/`
- có thêm overlay để xem input image nào đang nằm ở đâu trong panorama

### `07_manual_projection_previews.ipynb`
- `manual full-canvas preview` để debug planar chain
- `manual cylindrical preview` để xem scene quét rộng hợp lý hơn
- xuất preview vào `outputs/manual/panoramas/`
- xuất log vào `outputs/manual/logs/`

## Metadata Notes

Mỗi `scene_XX/meta.json` hiện có hai nhóm thông tin:

- Tự sinh bằng code:
  - `ordered_files`
  - `reference_files`
  - `num_images`
  - `capture_span_seconds`
  - `max_capture_gap_seconds`
  - `image_stats`
  - `pair_audit`
- `audit_summary`

Trong `audit_summary.stability_check` sẽ có:
- `runs`
- `status_counts`
- `ok_rate`
- `dominant_status`
- `dominant_rate`
- `ok_panorama_shape_counts`
- `dominant_ok_panorama_shape`
- `dominant_ok_panorama_shape_rate`
- `ok_panorama_shape_bucket_size`
- `ok_panorama_shape_bucket_counts`
- `dominant_ok_panorama_shape_bucket`
- `dominant_ok_panorama_shape_bucket_rate`
- `is_output_consistent`
- `stability_label`

- Đánh giá thủ công:
  - `type`
  - `capture_group`
  - `category`
  - `difficulty`
  - `recommended_use`
  - `issues`
  - `notes`
  - các cờ `has_*`

Sau khi thêm scene mới, thứ tự nên là:
1. rename ảnh về `img_xxx`
2. chạy `regenerate_scene_meta.py`
3. chạy `validate_dataset.py`
4. review tay `category / difficulty / notes` nếu cần

## Notes

- `scene` có thể có thêm `reference_files` ngoài chain chính.
- `OpenCV Stitcher` có thể trả `OK` nhưng vẫn chỉ giữ một phần scene nếu chain rộng hoặc không ổn định.
- Với wide scene hoặc scene quay quanh người, `manual cylindrical preview` thường hữu ích hơn `manual full-canvas preview`.

Chi tiết định hướng project và pipeline tổng thể nằm trong [panorama_project_pipeline_guide.md](./panorama_project_pipeline_guide.md).

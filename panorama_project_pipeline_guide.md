# Panorama Stitching and Image Registration — Pipeline Guide

## 1) Mục tiêu dự án

Xây dựng một hệ thống có thể:
- nhận đầu vào là **một chuỗi ảnh chụp cùng một cảnh có overlap**
- hỗ trợ **core scenes: 2–5 ảnh/scene**
- hỗ trợ thêm **large scenes: 10–15 ảnh/scene** cho stress-test mở rộng
- các ảnh phải có **overlap**
- tự tìm các vùng tương ứng giữa các ảnh
- căn chỉnh các ảnh về cùng hệ tọa độ
- ghép chúng thành **một ảnh panorama hoàn chỉnh**

## 2) Ứng dụng

Dự án có thể được trình bày như:
- công cụ hỗ trợ **mobile photography**
- công cụ ghép ảnh để tạo **wide-view scene**
- công cụ hỗ trợ **virtual scene capture**

## 3) Đầu ra mong muốn

Hệ thống nên tạo ra:
- ảnh keypoints
- ảnh raw matches
- ảnh inlier matches sau RANSAC
- ảnh warped trung gian
- ảnh panorama cuối
- file log chỉ số: số keypoints, số matches, số inliers, runtime, ghi chú failure

---

## 4) Phạm vi nên chốt

### 4.1. Bắt buộc
- ghép được **2 ảnh**
- mở rộng lên **3–5 ảnh/scene** cho đánh giá chính
- có thể thêm **10–15 ảnh/scene** như bài test mở rộng, nhưng không thay cho bộ core
- so sánh ít nhất:
  - **ORB vs SIFT-like**
  - **Brute-force vs FLANN-like**
  - **Overlay vs Feathering**
- dùng **RANSAC**
- phân tích failure case:
  - low texture
  - repeated patterns
  - parallax
  - moving objects
  - exposure change

### 4.2. Không nên ôm thêm
- 360 panorama phức tạp
- cylindrical/spherical stitching nâng cao
- deep learning panorama synthesis
- auto scene ordering phức tạp
- self-calibration hoặc bundle adjustment nâng cao
- biến toàn bộ bộ đánh giá chính thành chỉ các large scenes

---

## 5) Toàn bộ pipeline của dự án

### Giai đoạn A. Data collection
Thu thập các scene ảnh có overlap đủ để ghép panorama, bao gồm both success cases và failure cases.

### Giai đoạn B. Data audit & organization
Kiểm tra overlap, độ mờ, độ sáng, thứ tự ảnh, tính hợp lệ của scene; sắp xếp dữ liệu thành development scenes, final test scenes, failure scenes.

### Giai đoạn C. Preprocessing
Resize, chuyển grayscale cho feature extraction, chuẩn hóa ảnh, optional blur filtering / brightness normalization.

### Giai đoạn D. Feature detection & descriptor extraction
Dùng ít nhất hai chiến lược, ví dụ:
- ORB
- SIFT-like

### Giai đoạn E. Feature matching
So khớp descriptor giữa các cặp ảnh liền kề:
- brute-force
- FLANN-like

### Giai đoạn F. Outlier rejection
Dùng **RANSAC** để loại matches sai.

### Giai đoạn G. Homography estimation
Ước lượng ma trận homography từ các inlier để ánh xạ ảnh này sang hệ tọa độ của ảnh kia.

### Giai đoạn H. Warping
Biến đổi hình học ảnh theo homography để đặt lên cùng canvas panorama.

### Giai đoạn I. Blending
Ghép vùng overlap bằng:
- simple overlay
- feathering
- optional multi-band approximation

### Giai đoạn J. Panorama quality assessment
Đánh giá:
- alignment quality
- ghosting
- seam visibility
- runtime
- success/failure rate

### Giai đoạn K. Visualization & report
Hiển thị tất cả các kết quả trung gian và phân tích nguyên nhân thành công/thất bại.

---

## 6) Cấu trúc thư mục dự án

```text
panorama_project/
│
├── data/
│   ├── raw/
│   │   ├── scene_01/
│   │   │   ├── img_01.jpg
│   │   │   ├── img_02.jpg
│   │   │   ├── img_03.jpg
│   │   │   └── meta.json
│   │   ├── scene_02/
│   │   └── ...
│   │
│   ├── processed/
│   │   ├── scene_01/
│   │   └── ...
│   │
│   ├── splits/
│   │   ├── dev_scenes.txt
│   │   ├── test_scenes.txt
│   │   └── failure_scenes.txt
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_matching_ransac.ipynb
│   ├── 04_stitching_pipeline.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── io_utils.py
│   ├── preprocess.py
│   ├── features.py
│   ├── matching.py
│   ├── geometry.py
│   ├── blending.py
│   ├── stitching.py
│   └── evaluation.py
│
├── outputs/
│   ├── manual/
│   │   ├── keypoints/
│   │   ├── matches/
│   │   ├── inliers/
│   │   ├── warped/
│   │   ├── panoramas/
│   │   └── logs/
│   └── openCV/
│       ├── panoramas/
│       └── logs/
│
├── report_assets/
│   ├── tables/
│   ├── figures/
│   └── failure_cases/
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 7) Hướng dẫn chi tiết từng bước

# BƯỚC 1. Chốt phương án kỹ thuật

## 7.1. Mục tiêu kỹ thuật
Chốt ngay từ đầu:
- ảnh đầu vào:
  - **core scenes**: 2–5 ảnh/scene
  - **large scenes**: 10–15 ảnh/scene cho stress-test mở rộng
- phương pháp baseline:
  - ORB + BF + RANSAC + overlay
- phương pháp so sánh:
  - SIFT-like + FLANN-like + RANSAC + feathering

## 7.2. Câu hỏi nhóm phải thống nhất
- có dùng ảnh màu ở phần cuối hay chỉ grayscale cho feature extraction?
- ghép theo cặp liên tiếp hay ghép toàn chuỗi?
- có resize tất cả ảnh về cùng max width không?
- có dùng auto rejection cho ảnh mờ không?

# BƯỚC 2. Thu thập dữ liệu

## 7.3. Dữ liệu chính nên gồm gì

### A. Success scenes
- **20–25 scenes**
- 3–5 ảnh/scene
- ảnh liền kề overlap khoảng **30%–60%**
- cùng thiết bị nếu có thể
- cùng orientation
- không zoom quá mạnh

Ví dụ scene tốt:
- tường có poster
- phòng học
- hành lang
- mặt ngoài tòa nhà
- công viên với nhiều chi tiết
- giá sách
- khu văn phòng
- dãy ghế / bàn

### B. Hard-but-valid scenes
- **5–8 scenes**
- ánh sáng không đều
- hơi thiếu texture
- hơi có parallax nhẹ
- một ít người đi qua
- repeated patterns vừa phải

### C. Failure scenes
- **8–12 scenes**
- dùng cho error analysis

### D. Large scenes
- **2–4 scenes**
- **10–15 ảnh/scene**
- dùng cho **extended evaluation / stress-test**
- không thay thế cho bộ core scenes
- chấp nhận độ khó cao hơn: drift tích lũy, weak pair, bridge image, exposure change nhẹ

# BƯỚC 3. Cách chụp ảnh cho dữ liệu chính

## 7.4. Cách chụp success scenes
- đứng tương đối cố định
- xoay camera từ trái sang phải hoặc ngược lại
- mỗi ảnh mới giữ lại 30–60% nội dung chung với ảnh trước
- không thay zoom
- không đi ngang quá nhiều
- tránh vật thể động lớn trong vùng overlap
- chụp nhanh liên tiếp để giảm thay đổi sáng

### Quy tắc overlap
Ví dụ 4 ảnh:
- img_01 overlap img_02
- img_02 overlap img_03
- img_03 overlap img_04

## 7.5. Khi nào cần ảnh cầu nối
Nếu:
- A overlap B tốt
- B với C overlap quá ít

thì chụp thêm:
- D nằm giữa B và C

Khi lưu dữ liệu, sắp lại:
- A → B → D → C

# BƯỚC 4. Thu thập failure cases

## 7.6. Các failure cases nên có và cách chụp

### 1) Overlap quá ít
- quay quá xa giữa hai ảnh
- kỳ vọng: ít match, ít inlier

### 2) Không có overlap
- chụp hai vùng tách rời
- kỳ vọng: không stitch được

### 3) Low texture
- tường trắng, trần trơn, bầu trời
- kỳ vọng: rất ít keypoints

### 4) Repeated patterns
- hàng cửa sổ, gạch nền, kệ nhiều ô
- kỳ vọng: match sai nhiều

### 5) Parallax do translation
- bước ngang rõ rệt giữa các ảnh
- kỳ vọng: foreground và background không thể cùng khớp

### 6) Đi vòng quanh vật thể gần
- xe, ghế, tượng, chai nước gần camera
- kỳ vọng: méo, ghosting

### 7) Moving objects
- có người hoặc xe đi qua vùng overlap
- kỳ vọng: vật thể bị nhân đôi

### 8) Exposure change
- chụp từ vùng tối sang vùng sáng
- kỳ vọng: seam lộ rõ

### 9) Motion blur
- lia nhanh, thiếu sáng
- kỳ vọng: giảm keypoints

### 10) Zoom / scale change mạnh
- 1x rồi 2x hoặc bước lại gần
- kỳ vọng: matching khó hơn

# BƯỚC 5. Chia dữ liệu

## 7.7. Development scenes
- **15–20 scenes**
- dùng thử tham số

## 7.8. Final test scenes
- **10 scenes**
- chỉ dùng cho báo cáo cuối

## 7.9. Failure scenes
- **8–12 scenes**
- dùng cho error analysis

# BƯỚC 6. Gắn metadata cho từng scene

Mỗi scene nên có:
- scene_id
- type: indoor / outdoor / unknown
- capture_group: core / large
- category: success / hard_valid / failure
- number_of_images
- expected_difficulty
- ordered_files
- capture_span_seconds
- max_capture_gap_seconds
- recommended_use
- notes
- has_moving_objects
- has_repeated_patterns
- has_low_texture
- has_parallax
- has_exposure_change
- has_motion_blur
- has_insufficient_overlap

Gợi ý:
- `type` có thể để `unknown` nếu chưa kiểm tra bằng mắt
- các cờ `has_*` có thể là `true / false / null`; `null` nghĩa là cần review thủ công
- nên có thêm `issues` và `audit_summary` để ghi lại kết quả kiểm tra bằng code

Ví dụ:

```json
{
  "scene_id": "scene_07",
  "type": "unknown",
  "capture_group": "core",
  "category": "failure",
  "difficulty": "hard",
  "num_images": 4,
  "ordered_files": ["img_01.jpg", "img_02.jpg", "img_03.jpg", "img_04.jpg"],
  "capture_span_seconds": 12,
  "max_capture_gap_seconds": 5,
  "issues": ["parallax", "moving_objects"],
  "has_moving_objects": true,
  "has_repeated_patterns": false,
  "has_low_texture": false,
  "has_parallax": true,
  "has_exposure_change": false,
  "has_motion_blur": false,
  "has_insufficient_overlap": false,
  "recommended_use": "failure_analysis",
  "notes": "camera moved sideways while capturing",
  "audit_summary": {
    "stitcher_status": "ERR_HOMOGRAPHY_EST_FAIL",
    "avg_keypoints": 912.4,
    "brightness_span": 11.8
  }
}
```

# BƯỚC 7. Data audit

## 7.10. Nên kiểm tra thủ công
- ảnh có overlap không
- có ảnh nào mờ nặng không
- thứ tự ảnh đã đúng theo chuỗi overlap chưa
- có ảnh cầu nối cần chèn không

## 7.11. Nên kiểm tra bằng code
Cho từng ảnh:
- kích thước
- brightness mean
- contrast std
- blur score

Cho từng cặp ảnh:
- số keypoints
- số raw matches
- số good matches
- số inliers
- inlier ratio

# BƯỚC 8. Preprocessing

## 7.12. Preprocessing bắt buộc
- đọc ảnh
- chỉnh orientation đúng nếu cần
- resize về cùng `max_width`
- chuyển grayscale cho feature extraction
- lưu ảnh processed

## 7.13. Preprocessing cân nhắc
- histogram normalization nhẹ
- CLAHE
- Gaussian blur rất nhẹ nếu ảnh quá nhiễu

## 7.14. Điều không nên làm
- chỉnh tay từng ảnh riêng
- crop thủ công để cứu lỗi
- dùng sharpening riêng lẻ từng ảnh

# BƯỚC 9. Feature detection & descriptor extraction

## 7.15. Phương án nên dùng

### Method A: ORB
Ưu điểm:
- nhanh
- nhẹ
- dễ chạy trên CPU

Nhược điểm:
- kém bền hơn khi scale/texture khó

### Method B: SIFT-like
Ưu điểm:
- ổn định hơn với scale/rotation vừa phải

Nhược điểm:
- chậm hơn
- nặng hơn

## 7.16. Kết quả cần lưu
- ảnh keypoints
- số lượng keypoints

# BƯỚC 10. Feature matching

## 7.17. Matching strategies

### A. Brute-force matcher
- đơn giản
- tốt cho baseline

### B. FLANN-like matcher
- nhanh hơn khi nhiều keypoints

## 7.18. Post-processing cho matching
- knn matching
- ratio test
- rồi đưa qua RANSAC

## 7.19. Cần lưu gì
- raw matches image
- good matches image
- match counts

# BƯỚC 11. Outlier rejection bằng RANSAC

## 7.20. Vai trò
RANSAC giữ correspondence hình học đúng và loại match sai.

## 7.21. Đầu ra
- inlier mask
- số inliers
- inlier ratio
- ảnh inlier matches

# BƯỚC 12. Homography estimation

## 7.22. Ý nghĩa
Homography là ma trận biến đổi 2D giúp ánh xạ điểm của ảnh B sang hệ tọa độ ảnh A.

## 7.23. Khi nào hợp lý
- camera chủ yếu quay
- cảnh gần phẳng
- parallax không quá lớn

## 7.24. Khi nào dễ fail
- translation mạnh
- object gần camera
- nhiều lớp độ sâu
- đi vòng quanh vật thể 3D

# BƯỚC 13. Warping

## 7.25. Mục tiêu
Dùng homography để warp ảnh sang cùng canvas.

## 7.26. Cần chú ý
- canvas size phải đủ lớn
- tọa độ sau warp có thể âm, cần offset
- cần xác định vùng bao panorama

# BƯỚC 14. Blending

## 7.27. Overlay
Ưu:
- đơn giản
- nhanh

Nhược:
- seam rất rõ

## 7.28. Feathering
Ưu:
- mềm seam hơn

Nhược:
- alignment sai thì vẫn ghosting

# BƯỚC 15. Multi-image stitching

## 7.29. Cách làm
- ghép tuần tự theo chuỗi overlap
- ví dụ 01 → 02 → 03 → 04

## 7.30. Quy tắc dữ liệu
Thứ tự file nên phản ánh:
- ảnh nào overlap ảnh nào

# BƯỚC 16. Evaluation

## 7.31. Chỉ số định lượng nên có

### Cho từng ảnh
- blur score
- brightness mean
- contrast std
- keypoint count

### Cho từng cặp ảnh
- raw matches
- good matches
- inliers
- inlier ratio
- homography success/fail

### Cho từng scene
- total runtime
- stitch success/fail
- number of images successfully merged
- capture_group: core hay large
- capture_span_seconds
- max_capture_gap_seconds

## 7.32. Đánh giá bán định lượng
Thang 1–5:
- seam visibility
- ghosting severity
- overall visual coherence

## 7.33. Đánh giá định tính
Mỗi phương pháp nên có:
- 2 good examples
- 2 medium examples
- 3–5 failure examples

# BƯỚC 17. So sánh thí nghiệm

## 7.34. Ma trận thí nghiệm tối thiểu

### Experiment 1
- ORB + BF + RANSAC + overlay

### Experiment 2
- ORB + BF + RANSAC + feathering

### Experiment 3
- ORB + FLANN-like + RANSAC + feathering

### Experiment 4
- SIFT-like + BF/FLANN-like + RANSAC + feathering

## 7.35. So sánh chính
### A. Feature/descriptor
- ORB vs SIFT-like

### B. Matcher
- BF vs FLANN-like

### C. Blending
- overlay vs feathering

# BƯỚC 18. Sản phẩm cuối

## 7.36. Nên có
- notebook hoặc script chạy được
- README hướng dẫn chạy
- folder output có kết quả
- bảng kết quả
- ảnh minh họa cho report

## 7.37. Command-line mẫu
```bash
python stitch.py --scene data/raw/scene_01 --feature orb --matcher bf --blend feather
```

## 7.38. Output mẫu
### A. Manual pipeline
- `outputs/manual/keypoints/scene_01_01.png`
- `outputs/manual/matches/scene_01_pair12.png`
- `outputs/manual/inliers/scene_01_pair12_inliers.png`
- `outputs/manual/panoramas/scene_01_final.png`
- `outputs/manual/logs/scene_01_metrics.json`

### B. OpenCV notebook baseline
- `outputs/openCV/panoramas/scene_01_opencv_panorama.jpg`
- `outputs/openCV/logs/scene_01_opencv_panorama_summary.json`

---

## 8) Kế hoạch thực hiện theo tuần

### Tuần 1
- chốt phạm vi
- chốt cấu trúc thư mục
- thu 5–8 scene thử
- dựng baseline ORB + BF + RANSAC + overlay

### Tuần 2
- chuẩn hóa quy tắc chụp ảnh
- thu đủ success scenes
- gắn metadata
- kiểm tra overlap, blur, ordering

### Tuần 3
- hoàn thiện data audit notebook
- chạy keypoint/matching analysis
- loại các scene chết
- bổ sung bridge images nếu thiếu overlap

### Tuần 4
- hoàn thiện pipeline baseline
- xuất keypoints, matches, inliers, panorama

### Tuần 5
- thêm SIFT-like
- thêm FLANN-like
- chạy so sánh feature và matcher

### Tuần 6
- thêm feathering
- chạy full experiment matrix
- bắt đầu lấy bảng kết quả

### Tuần 7
- thu failure scenes còn thiếu
- viết error analysis
- chọn hình success/failure tiêu biểu

### Tuần 8
- hoàn thiện report
- hoàn thiện slides
- đóng gói code + README + output

---

## 9) Checklist thu thập dữ liệu chuẩn

### 9.1. Với scene chính
- [ ] 3–5 ảnh
- [ ] ảnh liền kề overlap 30–60%
- [ ] cùng orientation
- [ ] không zoom lung tung
- [ ] không translation quá mạnh
- [ ] không có blur nặng
- [ ] có đủ chi tiết để match

### 9.2. Với large scene
- [ ] 10–15 ảnh
- [ ] chỉ dùng cho extended evaluation / stress-test
- [ ] chuỗi overlap vẫn liên tục
- [ ] ghi rõ pair nào yếu / fail nếu có
- [ ] không trộn vào bảng kết quả core nếu chưa ghi nhãn rõ

### 9.3. Với failure scene
- [ ] chỉ thay đổi **1 nguyên nhân chính** nếu có thể
- [ ] ghi rõ failure type
- [ ] chụp một version “thành công” và một version “thất bại” nếu có thể

---

## 10) Những sai lầm nên tránh

- dùng ảnh cắt từ **một ảnh lớn duy nhất** làm dữ liệu chính
- không ghi thứ tự overlap thực tế
- trộn lung tung success scenes và failure scenes
- trộn chung core scenes và large scenes trong cùng một bảng đánh giá chính mà không ghi nhãn
- chỉnh tay từng ảnh để “cứu” kết quả
- chỉ đưa panorama cuối mà không show keypoints / matches / inliers
- không lưu số liệu định lượng
- không có failure analysis

---

## 11) Mẫu mô tả pipeline ngắn gọn cho proposal/report

**Pipeline của hệ thống gồm các bước: (1) thu thập bộ ảnh chồng lấp theo từng scene, (2) kiểm tra và tổ chức dữ liệu, (3) tiền xử lý ảnh, (4) phát hiện keypoints và trích xuất descriptor, (5) so khớp đặc trưng giữa các ảnh liền kề, (6) loại outlier bằng RANSAC, (7) ước lượng homography để căn chỉnh hình học, (8) warp ảnh lên cùng canvas, (9) ghép và làm mượt vùng overlap bằng kỹ thuật blending, và (10) đánh giá chất lượng panorama bằng cả chỉ số định lượng lẫn phân tích trực quan. Hệ thống sẽ so sánh ít nhất hai chiến lược feature-description và hai kỹ thuật matching/blending để xác định phương án phù hợp nhất cho từng điều kiện cảnh.**

---

## 12) Kết luận thực tế cho nhóm

- **Dữ liệu chính**: scene thật, ảnh chụp riêng, có overlap
- **Core scenes**: 3–5 ảnh/scene cho đánh giá chính
- **Large scenes**: 10–15 ảnh/scene cho extended evaluation / stress-test
- **Không có training theo nghĩa train model**
- thay vào đó là **development scenes + test scenes + failure scenes**
- **Pipeline lõi**: ORB/SIFT-like → BF/FLANN-like → RANSAC → homography → warp → blend
- **Phải lưu kết quả trung gian**
- **Phải có failure analysis**
- **Phải có số liệu, không chỉ nhìn bằng mắt**

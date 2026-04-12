from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "07_manual_projection_previews.ipynb"


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def split_block(source: str, start_marker: str, end_marker: str | None = None) -> str:
    start = source.index(start_marker)
    end = len(source) if end_marker is None else source.index(end_marker, start)
    return source[start:end].strip() + "\n"


def leading_block(source: str, end_marker: str) -> str:
    end = source.index(end_marker)
    return source[:end].strip() + "\n"


def trailing_block(source: str, start_marker: str) -> str:
    start = source.index(start_marker)
    return source[start:].strip() + "\n"


def replace_slice(cells: list[dict], start_idx: int, end_idx: int, new_cells: list[dict]) -> list[dict]:
    return cells[:start_idx] + new_cells + cells[end_idx:]


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    if any(
        cell["cell_type"] == "markdown" and "### Full-Canvas Part 1: Config And Output Targets" in "".join(cell["source"])
        for cell in cells
    ):
        print(f"{NOTEBOOK_PATH} is already sliced into manual-preview modules.")
        return

    full_md_idx = next(i for i, cell in enumerate(cells) if cell["cell_type"] == "markdown" and "## Manual Full-Canvas Preview" in "".join(cell["source"]))
    full_code_idx = full_md_idx + 1
    cyl_md_idx = next(i for i, cell in enumerate(cells) if cell["cell_type"] == "markdown" and "## Manual Cylindrical Preview" in "".join(cell["source"]))
    cyl_code_idx = cyl_md_idx + 1

    full_source = "".join(cells[full_code_idx]["source"])
    cyl_source = "".join(cells[cyl_code_idx]["source"])

    full_config = leading_block(full_source, "def resize_long_edge(")
    full_helpers = split_block(full_source, "def resize_long_edge(", "def estimate_adjacent_homography(")
    full_pair = split_block(full_source, "def estimate_adjacent_homography(", "def manual_full_canvas_preview(")
    full_runner = split_block(full_source, "def manual_full_canvas_preview(", "\n\ntry:\n")
    full_execute = trailing_block(full_source, "\n\ntry:\n")

    cyl_config = leading_block(cyl_source, "def cyl_resize_long_edge(")
    cyl_helpers = split_block(cyl_source, "def cyl_resize_long_edge(", "def estimate_cylindrical_pair_transform(")
    cyl_pair = split_block(cyl_source, "def estimate_cylindrical_pair_transform(", "def manual_cylindrical_preview(")
    cyl_runner = split_block(cyl_source, "def manual_cylindrical_preview(", "\n\ncylindrical_preview_result =")
    cyl_execute = split_block(cyl_source, "cylindrical_preview_result =")

    full_cells = [
        make_markdown_cell(
            """## Manual Full-Canvas Preview

Phần này mô phỏng một stitcher phẳng tối giản do chúng ta tự ghép bằng tay.

Mục tiêu của preview này không phải là tạo panorama đẹp nhất, mà là để bóc tách hình học:
- ảnh nào nối với ảnh nào
- transform được chuyền dọc chuỗi ra sao
- canvas cuối cùng rộng tới đâu nếu giữ tất cả ảnh
- drift tích lũy sẽ biểu hiện như thế nào khi không để OpenCV tự prune component

Vì thế, full-canvas preview thường hữu ích nhất cho debug hơn là cho kết quả cuối.
"""
        ),
        make_markdown_cell(
            """### Full-Canvas Part 1: Config And Output Targets

Cell này định nghĩa toàn bộ tham số điều khiển preview phẳng:
- `MANUAL_PREVIEW_MAX_LONG_EDGE`: scale ảnh xuống trước khi match và warp để giữ tốc độ
- `MANUAL_PREVIEW_ANCHOR`: ảnh nào được dùng làm hệ tọa độ gốc của chuỗi
- `MANUAL_PREVIEW_*`: ngưỡng cho ORB/SIFT, ratio test, RANSAC
- `MANUAL_PREVIEW_MAX_ESTIMATED_GB`: hàng rào an toàn để tránh cấp phát canvas quá lớn

Nó cũng chuẩn bị thư mục output cho panorama, overlay và log JSON. Đây là nơi notebook ghi các file debug để bạn xem lại ngoài notebook.
"""
        ),
        make_code_cell(full_config),
        make_markdown_cell(
            """### Full-Canvas Part 2: Preprocessing And Feature Helpers

Những helper này xử lý các thao tác cơ bản trước khi ghép:
- `resize_long_edge`: chuẩn hóa kích thước ảnh theo cạnh dài
- `gray_clahe`: chuyển sang grayscale và tăng tương phản cục bộ để dễ phát hiện keypoint hơn
- `detector_and_norm`: chọn detector và khoảng cách matcher phù hợp với ORB hoặc SIFT
- `classify_manual_pair`: biến kết quả matching thành nhãn `strong / ok / weak / fail`

Điểm quan trọng ở đây là preview manual vẫn dùng OpenCV primitives, nhưng logic quyết định được viết tay. Tức là chúng ta đang tự ráp pipeline chứ không gọi `cv2.Stitcher`.
"""
        ),
        make_code_cell(full_helpers),
        make_markdown_cell(
            """### Full-Canvas Part 3: Adjacent Pair Estimation

Đây là khối tìm transform giữa hai ảnh kề nhau:
- detect keypoints trên hai ảnh đã CLAHE
- chạy KNN matching + ratio test
- ước lượng homography bằng RANSAC
- đo số inlier và median reprojection error
- chọn ứng viên tốt nhất giữa ORB và SIFT

Tại bước này, preview chỉ giải bài toán cục bộ giữa `image_a` và `image_b`. Nó chưa hiểu toàn chuỗi, chưa tối ưu global camera, và chưa biết scene nào nên bị loại bỏ.
"""
        ),
        make_code_cell(full_pair),
        make_markdown_cell(
            """### Full-Canvas Part 4: Chain Composition, Canvas Build, And Visualization

`manual_full_canvas_preview()` là phần nối tất cả mảnh lại với nhau:
1. đọc toàn bộ ảnh của scene theo thứ tự đã chốt
2. ước lượng homography cho từng cặp kề nhau
3. chọn một ảnh anchor ở giữa chuỗi
4. nhân dồn transform để đưa mọi ảnh về hệ tọa độ anchor
5. chiếu tất cả corners để ước lượng canvas
6. dùng memory guard để bỏ qua những canvas quá lớn
7. warp từng ảnh lên canvas chung
8. cộng dồn / average các pixel chồng lắp
9. tạo footprint overlay, coverage heatmap, và log JSON

Đây là phần quan trọng nhất để hiểu vì sao preview manual có thể kém hơn OpenCV:
- transform bị cộng dồn nên drift dễ tăng
- blending chỉ là average đơn giản
- không có bước camera adjustment toàn cục như OpenCV Stitcher
- không có seam finder hay component pruning thông minh
"""
        ),
        make_code_cell(full_runner),
        make_markdown_cell(
            """### Full-Canvas Part 5: Execute The Preview

Cell cuối cùng chạy preview phẳng với `SCENE_ID` hiện tại.

Notebook giữ `try/except` ở đây để một scene lỗi không làm hỏng toàn bộ notebook. Nếu fail, bạn vẫn nhận được một record lỗi thay vì notebook dừng hẳn.
"""
        ),
        make_code_cell(full_execute),
    ]

    cells = replace_slice(cells, full_md_idx, full_code_idx + 1, full_cells)

    # Recompute cylindrical indices after replacing the full-canvas block.
    cyl_md_idx = next(i for i, cell in enumerate(cells) if cell["cell_type"] == "markdown" and "## Manual Cylindrical Preview" in "".join(cell["source"]))

    cyl_cells = [
        make_markdown_cell(
            """## Manual Cylindrical Preview

Preview này phù hợp hơn với các scene quét rộng quanh người.

Ý tưởng là:
- trước hết bẻ mỗi ảnh sang mặt trụ
- rồi mới ước lượng transform giữa các ảnh kề nhau trong không gian cylindrical
- sau đó ghép tất cả lên một canvas chung

Nó vẫn là một pipeline thủ công, nhưng projection model hợp với wide panorama hơn full-canvas phẳng.
"""
        ),
        make_markdown_cell(
            """### Cylindrical Part 1: Config

Nhóm tham số này kiểm soát preview cylindrical:
- `CYL_PREVIEW_MAX_LONG_EDGE`: scale ảnh
- `CYL_PREVIEW_FOCAL_SCALE` hoặc `CYL_PREVIEW_FOCAL_PX_OVERRIDE`: giả định focal length cho phép warp cylindrical
- `CYL_PREVIEW_ANCHOR`: anchor ảnh ở giữa hoặc ảnh bạn chọn
- `CYL_PREVIEW_*`: ngưỡng matcher / RANSAC cho không gian cylindrical

Trong preview này, focal length là một giả định gần đúng. Vì vậy kết quả rất hữu ích để debug coverage, nhưng chưa phải calibration chính xác như một panorama engine hoàn chỉnh.
"""
        ),
        make_code_cell(cyl_config),
        make_markdown_cell(
            """### Cylindrical Part 2: Warp And Local Geometry Helpers

Khối helper này làm bốn việc:
- resize ảnh đầu vào
- tăng tương phản bằng CLAHE
- warp ảnh và mask sang cylindrical surface
- định nghĩa detector / matcher / rule gán nhãn pair

Khác với preview phẳng, ở đây transform cục bộ không còn là homography đầy đủ mà được ràng vào mô hình đơn giản hơn sau khi đã warp cylindrical.
"""
        ),
        make_code_cell(cyl_helpers),
        make_markdown_cell(
            """### Cylindrical Part 3: Pair Transform Estimation In Cylindrical Space

`estimate_cylindrical_pair_transform()` match hai ảnh đã warp cylindrical rồi fit `estimateAffinePartial2D`.

Lý do chọn affine partial ở đây:
- sau khi lên mặt trụ, chuyển động giữa hai ảnh kề nhau thường gần với tịnh tiến + xoay + scale nhẹ hơn
- affine partial thường ổn định hơn homography đầy đủ cho wide sweep debug

Cell này vẫn lưu lại số match, số inlier, reprojection error và cả `dx / dy` để bạn nhìn hướng dịch chuyển của từng cặp.
"""
        ),
        make_code_cell(cyl_pair),
        make_markdown_cell(
            """### Cylindrical Part 4: Chain Composition, Canvas Build, And Output

`manual_cylindrical_preview()` là nơi toàn bộ chuỗi cylindrical được ghép lại:
1. đọc ảnh theo thứ tự scene
2. resize và warp từng ảnh sang cylindrical surface
3. ước lượng transform giữa các ảnh kề nhau trong cylindrical space
4. chọn anchor và nhân dồn transform
5. tìm canvas đủ lớn để chứa toàn bộ footprint
6. warp các ảnh cylindrical lên canvas
7. average vùng overlap, tạo heatmap và footprint overlay
8. so sánh trực tiếp với panorama OpenCV nếu file đó đã tồn tại

Preview này thường hợp lý hơn full-canvas khi scene quét rộng, nhưng nó vẫn không có global camera optimization hay seam optimization như OpenCV Stitcher.
"""
        ),
        make_code_cell(cyl_runner),
        make_markdown_cell(
            """### Cylindrical Part 5: Execute The Preview

Cell này chạy preview cylindrical cho `SCENE_ID` hiện tại và hiển thị:
- panorama cylindrical
- overlay footprint
- coverage heatmap
- panorama OpenCV để so sánh
"""
        ),
        make_code_cell(cyl_execute),
    ]

    cells = replace_slice(cells, cyl_md_idx, cyl_md_idx + 2, cyl_cells)

    notebook["cells"] = cells
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()

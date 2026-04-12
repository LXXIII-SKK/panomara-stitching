from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def replace_cell_source(notebook: dict, predicate, new_source: str) -> bool:
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if predicate(cell, source):
            cell["source"] = new_source.splitlines(keepends=True)
            return True
    return False


def patch_opencv_notebook(path: Path) -> None:
    notebook = load_notebook(path)

    replacements = [
        (
            lambda cell, source: cell.get("cell_type") == "markdown" and "## Module 3: Dataset Inventory" in source,
            """## Module 3: Dataset Inventory

Cell này được đặt lên đầu để bạn xem nhanh toàn bộ scene hiện có trước khi chọn `SCENE_ID`.

Bảng sẽ hiển thị:
- `scene_id`
- số lượng ảnh và ảnh reference
- `capture_group`, `category`, `difficulty`
- trạng thái snapshot hiện tại trong `meta.json`
- `ok_rate`, `stability_label`, `output_consistent`
- mục đích sử dụng của scene
""",
        ),
        (
            lambda cell, source: cell.get("cell_type") == "code" and "def build_scene_table()" in source,
            """def build_scene_table() -> pd.DataFrame:
    rows = []
    for scene_dir in list_scene_dirs(RAW_DIR):
        files, reference_files, meta, used_meta_order = ordered_scene_files(scene_dir)
        audit_summary = meta.get("audit_summary", {}) or {}
        stability = audit_summary.get("stability_check", {}) or {}
        rows.append(
            {
                "scene_id": scene_dir.name,
                "num_images": len(files),
                "num_reference_images": len(reference_files),
                "capture_group": meta.get("capture_group"),
                "category": meta.get("category"),
                "difficulty": meta.get("difficulty"),
                "stitcher_status": audit_summary.get("stitcher_status"),
                "ok_rate": stability.get("ok_rate"),
                "stability_label": stability.get("stability_label"),
                "output_consistent": stability.get("is_output_consistent"),
                "recommended_use": meta.get("recommended_use"),
                "used_meta_order": used_meta_order,
            }
        )

    columns = [
        "scene_id",
        "num_images",
        "num_reference_images",
        "capture_group",
        "category",
        "difficulty",
        "stitcher_status",
        "ok_rate",
        "stability_label",
        "output_consistent",
        "recommended_use",
        "used_meta_order",
    ]
    return pd.DataFrame(rows)[columns].sort_values("scene_id").reset_index(drop=True)


scene_table = build_scene_table()
display(scene_table)
""",
        ),
        (
            lambda cell, source: cell.get("cell_type") == "code" and "def run_scene(" in source,
            """def run_scene(scene_id: str, max_input_width: int = 1600, mode_name: str = "PANORAMA", save_outputs: bool = True):
    scene_dir = RAW_DIR / scene_id
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene not found: {scene_dir}")

    files, reference_files, meta, used_meta_order = ordered_scene_files(scene_dir)
    if len(files) < 2:
        raise ValueError(f"Scene {scene_id} must have at least 2 images")

    images_for_stitcher = []
    file_names = [path.name for path in files]

    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        images_for_stitcher.append(resize_keep_aspect(image, max_input_width))

    stitcher = cv2.Stitcher_create(stitcher_mode_value(mode_name))
    status_code, panorama = stitcher.stitch(images_for_stitcher)
    status_code = int(status_code)
    status_name = STATUS_NAMES.get(status_code, f"CODE_{status_code}")

    image_metrics = [image_stats(image, file_name) for image, file_name in zip(images_for_stitcher, file_names)]
    pair_metrics = compute_pair_diagnostics(images_for_stitcher, file_names)

    panorama_path = None
    panorama_shape = None
    panorama_rgb = None
    if status_code == int(cv2.Stitcher_OK) and panorama is not None:
        panorama_shape = {
            "width": int(panorama.shape[1]),
            "height": int(panorama.shape[0]),
        }
        panorama_rgb = bgr_to_rgb(panorama)
        if save_outputs:
            panorama_path = OUTPUT_PANORAMA_DIR / f"{scene_id}_opencv_{mode_name.lower()}.jpg"
            cv2.imwrite(str(panorama_path), panorama)

    meta_audit = meta.get("audit_summary", {}) or {}
    meta_stability = meta_audit.get("stability_check", {}) or {}

    summary = {
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "num_images": len(file_names),
        "num_reference_images": len(reference_files),
        "reference_files": [path.name for path in reference_files],
        "ordered_files": file_names,
        "used_meta_order": used_meta_order,
        "meta_capture_group": meta.get("capture_group"),
        "meta_category": meta.get("category"),
        "meta_difficulty": meta.get("difficulty"),
        "meta_recommended_use": meta.get("recommended_use"),
        "meta_notes": meta.get("notes"),
        "meta_stitcher_status": meta_audit.get("stitcher_status"),
        "meta_ok_rate": meta_stability.get("ok_rate"),
        "meta_stability_label": meta_stability.get("stability_label"),
        "meta_output_consistent": meta_stability.get("is_output_consistent"),
        "stitcher_mode": mode_name.upper(),
        "max_input_width": int(max_input_width),
        "status_code": status_code,
        "status_name": status_name,
        "panorama_shape": panorama_shape,
        "panorama_path": None if panorama_path is None else str(panorama_path),
        "image_stats": image_metrics,
        "pair_diagnostics": pair_metrics,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    log_path = OUTPUT_LOG_DIR / f"{scene_id}_opencv_panorama_summary.json"
    if save_outputs:
        log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    display(Markdown(f"### {scene_id} | `{status_name}`"))
    display(pd.DataFrame(pair_metrics))

    if panorama_rgb is not None:
        plt.figure(figsize=(16, 7))
        plt.imshow(panorama_rgb)
        plt.title(f"{scene_id} panorama")
        plt.axis("off")
        plt.show()
    else:
        print("Stitcher did not return a panorama image.")

    if reference_files:
        print("Reference files excluded from stitch chain:", [path.name for path in reference_files])

    summary["log_path"] = str(log_path)
    return summary
""",
        ),
        (
            lambda cell, source: cell.get("cell_type") == "code" and "single_result = run_scene(" in source,
            """single_result = run_scene(
    scene_id=SCENE_ID,
    max_input_width=MAX_INPUT_WIDTH,
    mode_name=STITCHER_MODE,
    save_outputs=SAVE_OUTPUTS,
)

display(pd.DataFrame([
    {
        "scene_id": single_result["scene_id"],
        "num_images": single_result["num_images"],
        "meta_capture_group": single_result["meta_capture_group"],
        "meta_category": single_result["meta_category"],
        "meta_difficulty": single_result["meta_difficulty"],
        "meta_stitcher_status": single_result["meta_stitcher_status"],
        "meta_ok_rate": single_result["meta_ok_rate"],
        "meta_stability_label": single_result["meta_stability_label"],
        "meta_output_consistent": single_result["meta_output_consistent"],
        "status_name": single_result["status_name"],
        "panorama_shape": single_result["panorama_shape"],
        "panorama_path": single_result["panorama_path"],
        "log_path": single_result["log_path"],
    }
]))

display(Markdown(f"**Scene notes:** {single_result['meta_notes']}" if single_result["meta_notes"] else "**Scene notes:** none"))
display(
    Markdown(
        f"**Metadata stability:** {single_result['meta_stability_label']} | "
        f"ok_rate={single_result['meta_ok_rate']} | "
        f"output_consistent={single_result['meta_output_consistent']}"
        if single_result["meta_stability_label"] is not None
        else "**Metadata stability:** none"
    )
)
display(pd.DataFrame(single_result["pair_diagnostics"]))
""",
        ),
        (
            lambda cell, source: cell.get("cell_type") == "code" and "if RUN_ALL_SCENES:" in source and "batch_rows" in source,
            """if RUN_ALL_SCENES:
    batch_rows = []
    for scene_id in scene_table["scene_id"]:
        print(f"Running {scene_id} ...")
        result = run_scene(
            scene_id=scene_id,
            max_input_width=MAX_INPUT_WIDTH,
            mode_name=STITCHER_MODE,
            save_outputs=SAVE_OUTPUTS,
        )
        batch_rows.append(
            {
                "scene_id": result["scene_id"],
                "num_images": result["num_images"],
                "meta_capture_group": result["meta_capture_group"],
                "meta_category": result["meta_category"],
                "meta_difficulty": result["meta_difficulty"],
                "meta_stitcher_status": result["meta_stitcher_status"],
                "meta_ok_rate": result["meta_ok_rate"],
                "meta_stability_label": result["meta_stability_label"],
                "meta_output_consistent": result["meta_output_consistent"],
                "status_code": result["status_code"],
                "status_name": result["status_name"],
                "panorama_path": result["panorama_path"],
                "log_path": result["log_path"],
            }
        )

    batch_df = pd.DataFrame(batch_rows).sort_values(
        ["status_code", "meta_ok_rate", "scene_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    batch_csv_path = OUTPUT_LOG_DIR / f"batch_summary_{STITCHER_MODE.lower()}.csv"
    batch_df.to_csv(batch_csv_path, index=False)
    display(batch_df)
    print("Saved batch summary to:", batch_csv_path)
else:
    print("Set RUN_ALL_SCENES = True and rerun this cell to stitch every scene.")
""",
        ),
    ]

    for predicate, new_source in replacements:
        if not replace_cell_source(notebook, predicate, new_source):
            raise RuntimeError(f"Could not find target cell in {path.name}")

    save_notebook(path, notebook)


def patch_manual_notebook(path: Path) -> None:
    notebook = load_notebook(path)

    if not replace_cell_source(
        notebook,
        lambda cell, source: cell.get("cell_type") == "code" and "def preview_scene(scene_id: str, max_width: int = 420):" in source,
        """def preview_scene(scene_id: str, max_width: int = 420):
    scene_dir = RAW_DIR / scene_id
    files, reference_files, meta, _ = ordered_scene_files(scene_dir)
    audit_summary = meta.get("audit_summary", {}) or {}
    stability = audit_summary.get("stability_check", {}) or {}

    display(
        pd.DataFrame(
            [
                {
                    "scene_id": scene_id,
                    "num_images": len(files),
                    "num_reference_images": len(reference_files),
                    "capture_group": meta.get("capture_group"),
                    "category": meta.get("category"),
                    "difficulty": meta.get("difficulty"),
                    "stitcher_status": audit_summary.get("stitcher_status"),
                    "ok_rate": stability.get("ok_rate"),
                    "stability_label": stability.get("stability_label"),
                    "output_consistent": stability.get("is_output_consistent"),
                }
            ]
        )
    )

    figure, axes = plt.subplots(1, len(files), figsize=(max(4, 3 * len(files)), 4))
    if len(files) == 1:
        axes = [axes]

    for axis, file_path in zip(axes, files):
        image = cv2.imread(str(file_path))
        image = resize_keep_aspect(image, max_width)
        axis.imshow(bgr_to_rgb(image))
        axis.set_title(file_path.name, fontsize=8)
        axis.axis("off")

    figure.suptitle(scene_id)
    plt.tight_layout()
    plt.show()

    if reference_files:
        print("Reference files excluded from stitch chain:", [path.name for path in reference_files])
""",
    ):
        raise RuntimeError(f"Could not find preview_scene cell in {path.name}")

    save_notebook(path, notebook)


def main() -> None:
    patch_opencv_notebook(PROJECT_ROOT / "notebooks" / "06_opencv_scene_stitcher.ipynb")
    patch_manual_notebook(PROJECT_ROOT / "notebooks" / "07_manual_projection_previews.ipynb")
    print("Synced notebooks with stability metadata.")


if __name__ == "__main__":
    main()

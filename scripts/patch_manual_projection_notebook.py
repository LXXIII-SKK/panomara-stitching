from pathlib import Path
import json


CONFIG_OLD = """MANUAL_PREVIEW_MAX_LONG_EDGE = 1600
MANUAL_PREVIEW_ANCHOR = None  # 1-based image index, or None to use the middle image
MANUAL_PREVIEW_SAVE_OUTPUTS = True
MANUAL_PREVIEW_RANSAC_THRESH = 4.0
MANUAL_PREVIEW_RATIO_TEST = 0.75
MANUAL_PREVIEW_ORB_FEATURES = 6000
"""

CONFIG_NEW = """MANUAL_PREVIEW_MAX_LONG_EDGE = 1600
MANUAL_PREVIEW_ANCHOR = None  # 1-based image index, or None to use the middle image
MANUAL_PREVIEW_SAVE_OUTPUTS = True
MANUAL_PREVIEW_RANSAC_THRESH = 4.0
MANUAL_PREVIEW_RATIO_TEST = 0.75
MANUAL_PREVIEW_ORB_FEATURES = 6000
MANUAL_PREVIEW_MAX_ESTIMATED_GB = 4.0
"""

CANVAS_OLD = """    canvas_width = int(max(1, np.ceil(max_xy[0] - min_xy[0])))
    canvas_height = int(max(1, np.ceil(max_xy[1] - min_xy[1])))

    accum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)
    coverage = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
"""

CANVAS_NEW = """    canvas_width = int(max(1, np.ceil(max_xy[0] - min_xy[0])))
    canvas_height = int(max(1, np.ceil(max_xy[1] - min_xy[1])))

    estimated_alloc_bytes = canvas_width * canvas_height * (3 * 4 + 1 * 4 + 2)
    estimated_alloc_gb = estimated_alloc_bytes / (1024 ** 3)
    if estimated_alloc_gb > MANUAL_PREVIEW_MAX_ESTIMATED_GB:
        manual_log_path = MANUAL_OUTPUT_LOG_DIR / f"{scene_id}_manual_full_canvas_preview.json"
        summary = {
            "scene_id": scene_id,
            "scene_dir": str(scene_dir),
            "status": "skipped_large_canvas",
            "reason": "estimated_memory_guard",
            "estimated_alloc_gb": round(float(estimated_alloc_gb), 2),
            "num_images": len(files),
            "num_reference_images": len(reference_files),
            "reference_files": [path.name for path in reference_files],
            "ordered_files": [path.name for path in files],
            "used_meta_order": used_meta_order,
            "anchor_index_1based": int(anchor_index + 1),
            "anchor_file": files[anchor_index].name,
            "max_long_edge": int(max_long_edge),
            "canvas_shape": {"width": int(canvas_width), "height": int(canvas_height)},
            "manual_panorama_path": None,
            "manual_overlay_path": None,
            "manual_coverage_path": None,
            "pair_records": pair_records,
            "image_footprints": image_footprints,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "manual_log_path": str(manual_log_path),
        }
        if save_outputs:
            manual_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        display(Markdown(f"### {scene_id} | manual full-canvas preview skipped"))
        display(pd.DataFrame([
            {
                "scene_id": scene_id,
                "status": summary["status"],
                "canvas_shape": summary["canvas_shape"],
                "estimated_alloc_gb": summary["estimated_alloc_gb"],
                "max_long_edge": max_long_edge,
                "manual_log_path": summary["manual_log_path"],
            }
        ]))
        return summary

    accum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)
    coverage = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
"""

CALL_OLD = """manual_preview_result = manual_full_canvas_preview(
    scene_id=SCENE_ID,
    max_long_edge=MANUAL_PREVIEW_MAX_LONG_EDGE,
    anchor_1based=MANUAL_PREVIEW_ANCHOR,
    save_outputs=MANUAL_PREVIEW_SAVE_OUTPUTS,
)
"""

CALL_NEW = """try:
    manual_preview_result = manual_full_canvas_preview(
        scene_id=SCENE_ID,
        max_long_edge=MANUAL_PREVIEW_MAX_LONG_EDGE,
        anchor_1based=MANUAL_PREVIEW_ANCHOR,
        save_outputs=MANUAL_PREVIEW_SAVE_OUTPUTS,
    )
except Exception as exc:
    manual_preview_result = {
        "scene_id": SCENE_ID,
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    display(Markdown(f"### {SCENE_ID} | manual full-canvas preview failed"))
    display(pd.DataFrame([manual_preview_result]))
"""


def patch_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def manual_full_canvas_preview" not in src:
            continue

        updated = src
        updated = updated.replace(CONFIG_OLD, CONFIG_NEW)
        updated = updated.replace(CANVAS_OLD, CANVAS_NEW)
        updated = updated.replace(CALL_OLD, CALL_NEW)

        if updated != src:
            cell["source"] = updated.splitlines(keepends=True)
            changed = True

    if changed:
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return changed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "notebooks" / "07_manual_projection_previews.ipynb",
    ]

    patched = []
    for target in targets:
        if target.exists() and patch_notebook(target):
            patched.append(str(target))

    if patched:
        print("Patched:")
        for item in patched:
            print(item)
    else:
        print("No notebook changes were needed.")


if __name__ == "__main__":
    main()

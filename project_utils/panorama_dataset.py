from __future__ import annotations

import json
import re
from pathlib import Path

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMG_SEQUENCE_RE = re.compile(r"^img_(\d+)$", re.IGNORECASE)


def list_scene_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)


def load_scene_meta(scene_dir: Path) -> dict:
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def list_all_image_files(scene_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in scene_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS
        ],
        key=lambda path: path.name.lower(),
    )


def list_numbered_sequence_files(scene_dir: Path) -> list[Path]:
    numbered = []
    for path in list_all_image_files(scene_dir):
        match = IMG_SEQUENCE_RE.match(path.stem)
        if match:
            numbered.append((int(match.group(1)), path))
    return [path for _, path in sorted(numbered, key=lambda item: item[0])]


def ordered_scene_files(scene_dir: Path) -> tuple[list[Path], list[Path], dict, bool]:
    image_files = list_all_image_files(scene_dir)
    meta = load_scene_meta(scene_dir)
    ordered_names = meta.get("ordered_files", []) or []
    name_to_path = {path.name: path for path in image_files}

    if ordered_names and all(name in name_to_path for name in ordered_names):
        ordered = [name_to_path[name] for name in ordered_names]
        ordered_name_set = {path.name for path in ordered}
        reference = [path for path in image_files if path.name not in ordered_name_set]
        return ordered, reference, meta, True

    numbered = list_numbered_sequence_files(scene_dir)
    if numbered:
        numbered_name_set = {path.name for path in numbered}
        reference = [path for path in image_files if path.name not in numbered_name_set]
        return numbered, reference, meta, False

    return image_files, [], meta, False


def scene_file_summary(scene_dir: Path) -> dict:
    ordered, reference, meta, used_meta_order = ordered_scene_files(scene_dir)
    return {
        "scene_id": scene_dir.name,
        "ordered_files": [path.name for path in ordered],
        "reference_files": [path.name for path in reference],
        "num_images": len(ordered),
        "used_meta_order": used_meta_order,
        "meta": meta,
    }

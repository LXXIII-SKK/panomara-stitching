from pathlib import Path
import re
import uuid

ROOT_DIR = Path(r"../data/raw")  # contains scene_01, scene_02, ...
PREFIX = "img"
START_INDEX = 1
PAD_WIDTH = 2
DRY_RUN = False
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

number_pattern = re.compile(r"(\d+)")

def extract_first_number(name: str):
    match = number_pattern.search(name)
    return int(match.group(1)) if match else float("inf")

def list_images(scene_dir: Path):
    files = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: (extract_first_number(p.stem), p.name.lower()))
    return files

def renumber_scene(scene_dir: Path):
    files = list_images(scene_dir)
    if not files:
        return

    temp_pairs = []
    for f in files:
        temp_path = scene_dir / f"__tmp__{uuid.uuid4().hex}{f.suffix.lower()}"
        temp_pairs.append((f, temp_path))

    final_pairs = []
    for i, (_, temp_path) in enumerate(temp_pairs, start=START_INDEX):
        final_path = scene_dir / f"{PREFIX}_{i:0{PAD_WIDTH}d}{temp_path.suffix.lower()}"
        final_pairs.append((temp_path, final_path))

    print(f"\nScene: {scene_dir.name}")
    for (orig, _), (_, final) in zip(temp_pairs, final_pairs):
        print(f"  {orig.name} -> {final.name}")

    if DRY_RUN:
        return

    for orig, temp in temp_pairs:
        orig.rename(temp)
    for temp, final in final_pairs:
        temp.rename(final)

if __name__ == "__main__":
    scene_dirs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])
    for scene_dir in scene_dirs:
        renumber_scene(scene_dir)

    if DRY_RUN:
        print("\nDRY_RUN=True, no files were renamed.")
    else:
        print("\nAll scenes processed.")
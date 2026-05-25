import cv2
import numpy as np
import pandas as pd
import os
import random
import shutil
import argparse
from pathlib import Path

random.seed(42)
np.random.seed(42)

# =========================================================
# STEP 1: SPLIT DATASET
# =========================================================

def find_data_root():
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "data" / "cnn" / "val2017").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_data_root()
SOURCE_DIR = PROJECT_ROOT / "data" / "cnn" / "val2017"
BASE_OUTPUT = PROJECT_ROOT / "data" / "cnn" / "split"

TRAIN_DIR = BASE_OUTPUT / "Train"
VAL_DIR = BASE_OUTPUT / "Val"
TEST_DIR = BASE_OUTPUT / "Test"


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_dataset(train_ratio=0.8, val_ratio=0.1, seed=42):

    images = [
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print("Total images:", len(images))

    rng = random.Random(seed)
    rng.shuffle(images)

    train_count = max(1, int(len(images) * train_ratio))
    val_count = max(1, int(len(images) * val_ratio))

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    print("Train:", len(train_images))
    print("Val:", len(val_images))
    print("Test:", len(test_images))

    for target_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        reset_dir(target_dir)

    copy_images(train_images, TRAIN_DIR)
    copy_images(val_images, VAL_DIR)
    copy_images(test_images, TEST_DIR)

    print("Dataset split complete.")


def copy_images(image_list, target_dir):

    for idx, image_name in enumerate(image_list, start=1):

        src = SOURCE_DIR / image_name

        dst = target_dir / f"{idx}.jpg"

        shutil.copy(src, dst)


# =========================================================
# STEP 2: SYNTHETIC PATCH GENERATION
# =========================================================

def count_images_in_folder(folder_path):
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    return len(image_files)


def GetPatches(image, patch_size=128, perturbation=32,
               border=42, translation=10):

    h, w = image.shape[:2]
    min_size = patch_size + 2 * border + 1

    if w > min_size and h > min_size:

        end_margin = patch_size + border

        x = np.random.randint(border, w - end_margin)
        y = np.random.randint(border, h - end_margin)

        translation = np.random.randint(-translation, translation)

        pts1 = np.array([
            [x, y],
            [x, patch_size + y],
            [patch_size + x, y],
            [patch_size + x, patch_size + y]
        ])

        pts2 = np.zeros_like(pts1)

        for i, pt in enumerate(pts1):

            pts2[i][0] = (
                pt[0]
                + np.random.randint(-perturbation, perturbation)
                + translation
            )

            pts2[i][1] = (
                pt[1]
                + np.random.randint(-perturbation, perturbation)
                + translation
            )

        H_inv = np.linalg.inv(
            cv2.getPerspectiveTransform(
                np.float32(pts1),
                np.float32(pts2)
            )
        )

        imageB = cv2.warpPerspective(image, H_inv, (w, h))

        Patch_a = image[y:y + patch_size, x:x + patch_size]
        Patch_b = imageB[y:y + patch_size, x:x + patch_size]

        H4 = (pts2 - pts1).astype(np.float32)

        return Patch_a, Patch_b, H4, imageB, pts1, pts2

    else:
        return None, None, None, None, None, None


def generate_data_set(
    option,
    path,
    save_path,
    patches_per_image=12,
    patch_size=128,
    perturbation=32,
    border=42,
    translation=10,
    resize_width=320,
    resize_height=240,
):

    im_count = count_images_in_folder(path)

    save_path = Path(save_path)

    reset_dir(save_path)

    H4_list = []
    Ca_list = []
    Cb_list = []

    print(f"Generating {option} data with {im_count} images ......")
    print(f"Patches per image: {patches_per_image}")
    print("Begin Data Generation .... ")

    for i in range(1, im_count + 1):

        image_path = Path(path) / f"{i}.jpg"

        image_a = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image_a is None:
            print(f"Could not read image: {image_path}")
            continue

        image_a = cv2.resize(
            image_a,
            (resize_width, resize_height),
            interpolation=cv2.INTER_AREA
        )

        for patch_idx in range(int(patches_per_image)):

            patch_a, patch_b, H4, _, Ca, Cb = GetPatches(
                image_a,
                patch_size=patch_size,
                perturbation=perturbation,
                border=border,
                translation=translation
            )

            if patch_a is None:

                print(f"Encountered invalid image, skipping: {image_path}")
                continue

            sub_directories = ['PA', 'PB', 'IA']

            for sub_dir in sub_directories:

                sub_path = save_path / sub_dir

                os.makedirs(sub_path, exist_ok=True)

            image_name = f"{i:06d}_{patch_idx:03d}.jpg"

            path_a = save_path / 'PA' / image_name
            path_b = save_path / 'PB' / image_name
            im_path_a = save_path / 'IA' / image_name

            cv2.imwrite(str(path_a), patch_a)
            cv2.imwrite(str(path_b), patch_b)
            cv2.imwrite(str(im_path_a), image_a)

            H4_values = list(np.hstack((H4[:, 0], H4[:, 1])))
            H4_list.append([
                image_name,
                H4_values[0], H4_values[4],
                H4_values[1], H4_values[5],
                H4_values[2], H4_values[6],
                H4_values[3], H4_values[7]
            ])

            CA_values = list(np.hstack((Ca[:, 0], Ca[:, 1])))
            Ca_list.append([
                image_name,
                CA_values[0], CA_values[4],
                CA_values[1], CA_values[5],
                CA_values[2], CA_values[6],
                CA_values[3], CA_values[7]
            ])

            CB_values = list(np.hstack((Cb[:, 0], Cb[:, 1])))
            Cb_list.append([
                image_name,
                CB_values[0], CB_values[4],
                CB_values[1], CB_values[5],
                CB_values[2], CB_values[6],
                CB_values[3], CB_values[7]
            ])

    pd.DataFrame(H4_list).to_csv(
        save_path / "H4.csv",
        header=False,
        index=False
    )

    pd.DataFrame(Ca_list).to_csv(
        save_path / "Ca.csv",
        header=False,
        index=False
    )

    pd.DataFrame(Cb_list).to_csv(
        save_path / "Cb.csv",
        header=False,
        index=False
    )

    print(f"Saved synthetic dataset to: {save_path}")


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Seed", type=int, default=42)
    parser.add_argument("--TrainRatio", type=float, default=0.8)
    parser.add_argument("--ValRatio", type=float, default=0.1)
    parser.add_argument("--PatchesPerImage", type=int, default=12)
    parser.add_argument("--PatchSize", type=int, default=128)
    parser.add_argument("--Perturbation", type=int, default=32)
    parser.add_argument("--Border", type=int, default=42)
    parser.add_argument("--Translation", type=int, default=10)
    parser.add_argument("--ResizeWidth", type=int, default=320)
    parser.add_argument("--ResizeHeight", type=int, default=240)
    args = parser.parse_args()

    random.seed(args.Seed)
    np.random.seed(args.Seed)

    # Step 1: Split dataset
    split_dataset(
        train_ratio=args.TrainRatio,
        val_ratio=args.ValRatio,
        seed=args.Seed,
    )

    # Step 2: Generate synthetic training data
    generate_data_set(
        'Train',
        TRAIN_DIR,
        BASE_OUTPUT / 'Train_synthetic',
        patches_per_image=args.PatchesPerImage,
        patch_size=args.PatchSize,
        perturbation=args.Perturbation,
        border=args.Border,
        translation=args.Translation,
        resize_width=args.ResizeWidth,
        resize_height=args.ResizeHeight,
    )

    generate_data_set(
        'Val',
        VAL_DIR,
        BASE_OUTPUT / 'Val_synthetic',
        patches_per_image=args.PatchesPerImage,
        patch_size=args.PatchSize,
        perturbation=args.Perturbation,
        border=args.Border,
        translation=args.Translation,
        resize_width=args.ResizeWidth,
        resize_height=args.ResizeHeight,
    )

    generate_data_set(
        'Test',
        TEST_DIR,
        BASE_OUTPUT / 'Test_synthetic',
        patches_per_image=args.PatchesPerImage,
        patch_size=args.PatchSize,
        perturbation=args.Perturbation,
        border=args.Border,
        translation=args.Translation,
        resize_width=args.ResizeWidth,
        resize_height=args.ResizeHeight,
    )


if __name__ == '__main__':
    main()

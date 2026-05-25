#!/usr/bin/env python

import cv2
import os
import sys
import numpy as np
import torch
import torchvision.io as io
import csv
from pathlib import Path

from tqdm import tqdm
from torchvision import transforms

from Network.PhamHungSon_15_CNN_Network import HomographyModel, denormalize_prediction, DEFAULT_LABEL_SCALE


# =========================================================
# DON'T GENERATE .PYC FILES
# =========================================================

sys.dont_write_bytecode = True


def find_data_root(root_dir):
    for candidate in (root_dir, *root_dir.parents):
        if (candidate / "data" / "cnn").exists():
            return candidate
    return root_dir.parents[1]


# =========================================================
# STANDARDIZE INPUTS
# =========================================================

def StandardizeInputs(Img):
    # Img is a tensor in [0, 255]
    Img = Img / 255.0
    Img = (Img - 0.5) / 0.5
    return Img


# =========================================================
# READ LABELS
# =========================================================

def ReadLabelsTest(LabelsPathTest):

    if not os.path.isfile(LabelsPathTest):

        print(
            "ERROR: Labels do not exist in",
            LabelsPathTest
        )

        sys.exit()

    labels_test = {}

    with open(LabelsPathTest, 'r') as labels_file:

        csv_reader = csv.reader(labels_file)

        for row in csv_reader:

            image_name = row[0]

            labels = [
                float(label)
                for label in row[1:]
            ]

            labels_test[image_name] = labels

    return labels_test


# =========================================================
# TEST OPERATION
# =========================================================

def TestOperation(
    H4pt_batch,
    Patched_batch_torch,
    ModelPath,
    LabelsPathPred,
    Backbone="auto",
    LabelScale=None,
    InputMode="auto",
    OutputActivation="auto",
):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    CheckPoint = torch.load(
        ModelPath,
        map_location=device
    )

    checkpoint_args = CheckPoint.get("args", {}) or {}
    resolved_backbone = Backbone if Backbone != "auto" else checkpoint_args.get("Backbone", "ConvNet")
    resolved_label_scale = (
        float(LabelScale)
        if LabelScale is not None
        else float(checkpoint_args.get("LabelScale", DEFAULT_LABEL_SCALE))
    )
    # Older checkpoints were trained before these args existed.
    resolved_input_mode = (
        InputMode
        if InputMode != "auto"
        else checkpoint_args.get("InputMode", "basic")
    )
    resolved_output_activation = (
        OutputActivation
        if OutputActivation != "auto"
        else checkpoint_args.get("OutputActivation", "tanh")
    )

    print("Backbone:", resolved_backbone)
    print("InputMode:", resolved_input_mode)
    print("OutputActivation:", resolved_output_activation)
    print("LabelScale:", resolved_label_scale)

    model = HomographyModel(
        backbone=resolved_backbone,
        input_mode=resolved_input_mode,
        output_activation=resolved_output_activation,
    ).to(device)

    state_dict = CheckPoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "model.fc1.weight" and "model.fc1.0.weight" in model.state_dict():
            new_state_dict["model.fc1.0.weight"] = v
        elif k == "model.fc1.bias" and "model.fc1.0.bias" in model.state_dict():
            new_state_dict["model.fc1.0.bias"] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    print(
        "Number of parameters:",
        len(model.state_dict().items())
    )

    model.eval()

    OutSaveT = open(LabelsPathPred, "w")

    H4ptPred_list = []

    batch_size = 1

    for j in tqdm(
        range(
            0,
            len(Patched_batch_torch),
            batch_size
        )
    ):

        Img = Patched_batch_torch[
            j:j+batch_size
        ].to(device)

        PredH_norm = model(Img)
        PredH = denormalize_prediction(PredH_norm, label_scale=resolved_label_scale)

        PredH = (
            PredH
            .detach()
            .cpu()
            .numpy()
        )

        OutSaveT.write(
            str(PredH) + "\n"
        )

        H4ptPred_list.append(PredH)

        del Img

    OutSaveT.close()

    # =====================================================
    # COMPUTE ERROR AND CRUCIAL REGRESSION METRICS
    # =====================================================

    pred_arr = np.array([p[0] for p in H4ptPred_list])  # [N, 8]
    gt_arr = np.array(H4pt_batch)  # [N, 8]

    # Calculate L2 error (8D vector difference norm)
    ESE_list = [np.linalg.norm(x - y, ord=2) for x, y in zip(pred_arr, gt_arr)]
    Avg_L2_error = np.mean(ESE_list)

    # Calculate Average Corner Error (MCE) in pixels
    dx = pred_arr - gt_arr
    dx_reshaped = dx.reshape(-1, 4, 2)
    corner_errors = np.linalg.norm(dx_reshaped, axis=2)  # [N, 4]
    sample_mce = np.mean(corner_errors, axis=1)  # [N]

    mean_corner_error = np.mean(sample_mce)
    mae = np.mean(np.abs(dx))
    max_corner_error = np.max(corner_errors)

    accuracy_1px = np.mean(sample_mce < 1.0) * 100
    accuracy_3px = np.mean(sample_mce < 3.0) * 100
    accuracy_5px = np.mean(sample_mce < 5.0) * 100
    outlier_rate = np.mean(sample_mce > 10.0) * 100

    # Print out results with high precision
    print("\n================ EVALUATION METRICS ================")
    print(f"Number of Test Samples: {len(H4pt_batch)}")
    print(f"Average L2 Displacement Error: {Avg_L2_error:.4f} pixels")
    print(f"Mean Corner Error (MCE): {mean_corner_error:.4f} pixels")
    print(f"Mean Absolute Error (MAE): {mae:.4f} pixels")
    print(f"Maximum Corner Error: {max_corner_error:.4f} pixels")
    print(f"Accuracy @ 1.0 pixel threshold: {accuracy_1px:.2f}%")
    print(f"Accuracy @ 3.0 pixel threshold: {accuracy_3px:.2f}%")
    print(f"Accuracy @ 5.0 pixel threshold: {accuracy_5px:.2f}%")
    print(f"Outlier Rate (> 10.0 pixels): {outlier_rate:.2f}%")
    print("====================================================\n")

    return H4ptPred_list


# =========================================================
# VISUALIZATION
# =========================================================

def VisuaizePatch(
    PredH4pt_list,
    Ca_list,
    Cb_list,
    Ia_list,
    ResultsPath
):

    os.makedirs(
        ResultsPath,
        exist_ok=True
    )

    for i in range(len(PredH4pt_list)):

        PredH4pt = PredH4pt_list[i][0]

        Ca_corner = np.array(
            Ca_list[i]
        ).reshape(4, 2)

        Cb_corner = np.array(
            Cb_list[i],
            dtype=np.int32
        ).reshape((-1, 4, 2))

        Ia_img = Ia_list[i]

        Cb_Pred = (
            PredH4pt + Ca_corner.flatten()
        )

        Cb_Pred = (
            Cb_Pred.astype(np.int32)
            .reshape((-1, 4, 2))
        )

        # =================================================
        # GROUND TRUTH
        # =================================================

        for center in Cb_corner:

            for k in center:

                cv2.circle(
                    Ia_img,
                    (int(k[0]), int(k[1])),
                    2,
                    (0, 255, 0),
                    2
                )

        # =================================================
        # PREDICTION
        # =================================================

        for center in Cb_Pred:

            for l in center:

                cv2.circle(
                    Ia_img,
                    (int(l[0]), int(l[1])),
                    2,
                    (255, 0, 0),
                    2
                )

        # =================================================
        # DRAW POLYGONS
        # =================================================

        connections = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0)
        ]

        for corners in Cb_corner:

            for start_idx, end_idx in connections:

                start_point = tuple(
                    corners[start_idx]
                )

                end_point = tuple(
                    corners[end_idx]
                )

                cv2.line(
                    Ia_img,
                    start_point,
                    end_point,
                    (0, 255, 0),
                    2
                )

        for corners in Cb_Pred:

            for start_idx, end_idx in connections:

                start_point = tuple(
                    corners[start_idx]
                )

                end_point = tuple(
                    corners[end_idx]
                )

                cv2.line(
                    Ia_img,
                    start_point,
                    end_point,
                    (255, 0, 0),
                    2
                )

        output_path = os.path.join(
            ResultsPath,
            f"PhamHungSon_15_CNN_Image{i}.png"
        )

        cv2.imwrite(
            output_path,
            Ia_img
        )


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # CONFIGURATION
    # =====================================================

    import argparse
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--Backbone", type=str, default="auto", help="auto, ConvNet, or ResNet18")
    Parser.add_argument("--InputMode", type=str, default="auto", help="auto, basic, coord, pairdiff, or pairdiff_coord")
    Parser.add_argument("--OutputActivation", type=str, default="auto", help="auto, linear, or tanh")
    Parser.add_argument("--ModelPath", type=str, default="", help="Path to checkpoint")
    Parser.add_argument("--LabelScale", type=float, default=None, help="Scale used during supervised training; auto reads checkpoint args")
    Args = Parser.parse_args()

    ROOT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = find_data_root(ROOT_DIR)

    if Args.ModelPath:
        ModelPath = Args.ModelPath
    else:
        checkpoint_dir = ROOT_DIR / "PhamHungSon_15_CNN_CheckpointsSupRegularized"
        checkpoints = sorted(
            checkpoint_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not checkpoints:
            print("ERROR: no supervised checkpoints found in", checkpoint_dir)
            sys.exit()

        ModelPath = str(checkpoints[0])

    BasePath = PROJECT_ROOT / "data" / "cnn" / "split" / "Test_synthetic"

    LabelsPath = os.path.join(
        BasePath,
        "H4.csv"
    )

    ResultsPath = ROOT_DIR / "PhamHungSon_15_CNN_Results" / "PhamHungSon_15_CNN_Supervised_test"

    LabelsPathPred = ROOT_DIR / "PhamHungSon_15_CNN_TxtFiles" / "PhamHungSon_15_CNN_PredOutSup.txt"
    LabelsPathPred.parent.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # PATHS
    # =====================================================

    Ia = os.path.join(BasePath, "IA/")
    Patch_a = os.path.join(BasePath, "PA/")
    Patch_b = os.path.join(BasePath, "PB/")

    Corner_a = os.path.join(
        BasePath,
        "Ca.csv"
    )

    Corner_b = os.path.join(
        BasePath,
        "Cb.csv"
    )

    # =====================================================
    # LOAD LABELS
    # =====================================================

    H4_labels = ReadLabelsTest(
        LabelsPath
    )

    Ca_lables = ReadLabelsTest(
        Corner_a
    )

    Cb_lables = ReadLabelsTest(
        Corner_b
    )

    # =====================================================
    # LOAD DATA
    # =====================================================

    H4pt_list = []

    Patched_batch = []

    Ia_list = []

    Ca_list = []

    Cb_list = []

    for filename in tqdm(os.listdir(Patch_a)):

        Image1 = os.path.join(
            Patch_a,
            filename
        )

        Image2 = os.path.join(
            Patch_b,
            filename
        )

        Image_Ia = os.path.join(
            Ia,
            filename
        )

        patch_1 = cv2.imread(
            Image1,
            cv2.IMREAD_GRAYSCALE
        )

        patch_2 = cv2.imread(
            Image2,
            cv2.IMREAD_GRAYSCALE
        )

        Ia_img = cv2.imread(
            Image_Ia
        )

        if (
            patch_1 is None
            or patch_2 is None
            or Ia_img is None
        ):
            continue

        stacked_image = torch.cat([
            torch.from_numpy(patch_1),
            torch.from_numpy(patch_2)
        ], axis=0)

        stacked_image = stacked_image.view(
            2,
            128,
            128
        ).float()

        stacked_image = StandardizeInputs(
            stacked_image
        )

        H4pt = H4_labels[filename]

        Ca = Ca_lables[filename]

        Cb = Cb_lables[filename]

        H4pt_list.append(H4pt)

        Ia_list.append(Ia_img)

        Ca_list.append(Ca)

        Cb_list.append(Cb)

        Patched_batch.append(
            stacked_image.clone().detach()
        )

    Patched_batch_torch = torch.stack(
        Patched_batch
    )

    # =====================================================
    # TEST
    # =====================================================

    H4ptPred_list = TestOperation(
        H4pt_list,
        Patched_batch_torch,
        ModelPath,
        LabelsPathPred,
        Backbone=Args.Backbone,
        LabelScale=Args.LabelScale,
        InputMode=Args.InputMode,
        OutputActivation=Args.OutputActivation,
    )

    # =====================================================
    # VISUALIZE
    # =====================================================

    VisuaizePatch(
        H4ptPred_list,
        Ca_list,
        Cb_list,
        Ia_list,
        ResultsPath
    )

    print("\nTesting complete.")


if __name__ == "__main__":
    main()

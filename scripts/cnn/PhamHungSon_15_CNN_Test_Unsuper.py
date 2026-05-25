#!/usr/bin/env python

import cv2
import os
import sys
import numpy as np
import argparse
from Network.PhamHungSon_15_CNN_Network_unsuper import HomographyModel
from tqdm import tqdm
import torch
from torchvision import transforms
import csv
from pathlib import Path


# Don't generate pyc codes
sys.dont_write_bytecode = True


def find_data_root(root_dir):
    for candidate in (root_dir, *root_dir.parents):
        if (candidate / "data" / "cnn").exists():
            return candidate
    return root_dir.parents[1]


def StandardizeInputs(Img):
    # Img is a tensor in [0, 255]
    Img = Img / 255.0
    Img = (Img - 0.5) / 0.5
    return Img

def TestOperation(Patched_batch_torch, H4pt_batch_torch, Ca_batch_torch, Cb_batch_torch, Pa_batch_torch, LabelsPathPred, ModelPath, H4pt_list):

    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions are written to the local CNN PhamHungSon_15_CNN_TxtFiles folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = HomographyModel().to(device)

    CheckPoint = torch.load(ModelPath, map_location=device)

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
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )
    model.eval()
    OutSaveT = open(LabelsPathPred, "w")
    PredH4pt_list = []
    PredPb_list = []
    batch_size = 256
    for j in tqdm(range(0,len(Patched_batch_torch), batch_size)):
        
        Patched_batch_j = Patched_batch_torch[j:j+batch_size].to(device).float()
        H4pt_batch_j = H4pt_batch_torch[j:j+batch_size].to(device).float()
        Ca_batch_j = Ca_batch_torch[j:j+batch_size].to(device).float()
        Cb_batch_j = Cb_batch_torch[j:j+batch_size].to(device).float()
        Pa_batch_j = Pa_batch_torch[j:j+batch_size].to(device).float()
        PredPb, PredH = model(Patched_batch_j, H4pt_batch_j, Ca_batch_j, Cb_batch_j, Pa_batch_j)
        PredH = PredH.detach().cpu().numpy()
        PredPb = PredPb.detach().cpu().numpy()
        OutSaveT.write(str(PredH) + "\n")
        PredH4pt_list.extend(PredH)
        PredPb_list.extend(PredPb)
        del Patched_batch_j
        del H4pt_batch_j
        del Ca_batch_j
        del Cb_batch_j
        del Pa_batch_j
        
    OutSaveT.close()
    # =====================================================
    # COMPUTE ERROR AND CRUCIAL REGRESSION METRICS
    # =====================================================

    pred_arr = np.array(PredH4pt_list)  # [N, 8]
    gt_arr = np.array(H4pt_list)  # [N, 8]

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
    print(f"Number of Test Samples: {len(H4pt_list)}")
    print(f"Average L2 Displacement Error: {Avg_L2_error:.4f} pixels")
    print(f"Mean Corner Error (MCE): {mean_corner_error:.4f} pixels")
    print(f"Mean Absolute Error (MAE): {mae:.4f} pixels")
    print(f"Maximum Corner Error: {max_corner_error:.4f} pixels")
    print(f"Accuracy @ 1.0 pixel threshold: {accuracy_1px:.2f}%")
    print(f"Accuracy @ 3.0 pixel threshold: {accuracy_3px:.2f}%")
    print(f"Accuracy @ 5.0 pixel threshold: {accuracy_5px:.2f}%")
    print(f"Outlier Rate (> 10.0 pixels): {outlier_rate:.2f}%")
    print("====================================================\n")
    
    return PredH4pt_list, PredPb_list

def VisuaizePatch(PredH4pt_list, Ca_list, Cb_list, Ia_list, ResultsPath):

    os.makedirs(ResultsPath, exist_ok=True)
    
    for i in range(len(PredH4pt_list)):
        PredH4pt = PredH4pt_list[i]
        Ca_corner = Ca_list[i]
        Cb_corner = np.array(Cb_list[i], dtype=np.int32).reshape((-1, 4, 2))
        Ia_img = Ia_list[i]
        Cb_Pred = (PredH4pt + Ca_corner).astype(np.int32).reshape((-1, 4, 2))

                
        for center in Cb_corner:
            for k in center:
                cv2.circle(Ia_img, (int(k[0]), int(k[1])), 2, (0, 255, 0), 2) 
                
        for center in Cb_Pred:
            for l in center:
                cv2.circle(Ia_img, (int(l[0]), int(l[1])), 2, (255, 0, 0), 2)
                
        for corners in Cb_corner:
            connections = [(0, 1), (1, 3), (3, 2), (2, 0)]

            for start_idx, end_idx in connections:
                start_point = tuple(corners[start_idx])
                end_point = tuple(corners[end_idx])
                cv2.line(Ia_img, start_point, end_point, (0, 255, 0), 2)
                
        for corners in Cb_Pred:
            connections = [(0, 1), (1, 3), (3, 2), (2, 0)]

            for start_idx, end_idx in connections:
                start_point = tuple(corners[start_idx])
                end_point = tuple(corners[end_idx])
                cv2.line(Ia_img, start_point, end_point, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(ResultsPath, f"PhamHungSon_15_CNN_Image{i}.png"), Ia_img)

def ReadLabelsTest(LabelsPathTest):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTest)
        sys.exit()
    labels_test = {}

    with open(LabelsPathTest, 'r') as labels_file:
        csv_reader = csv.reader(labels_file)
        for row in csv_reader:
            image_name = row[0]
            labels = [float(label) for label in row[1:]]
            labels_test[image_name] = labels

    return labels_test

def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    ROOT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = find_data_root(ROOT_DIR)
    default_base = PROJECT_ROOT / "data" / "cnn" / "split" / "Test_synthetic"
    checkpoint_dir = ROOT_DIR / "PhamHungSon_15_CNN_CheckpointsUnsup5000"
    checkpoints = sorted(
        checkpoint_dir.glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    default_model = str(checkpoints[0]) if checkpoints else ""

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default=default_model,
        help="Path to load latest model from",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default=str(default_base),
        help="Path to load images from",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default=str(default_base / "H4.csv"),
        help="Path of labels file",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    if not ModelPath:
        print("ERROR: no unsupervised checkpoints found in", checkpoint_dir)
        sys.exit()
    
    Ia = os.path.join(BasePath, "IA/")
    Patch_a = os.path.join(BasePath, "PA/")
    Patch_b = os.path.join(BasePath, "PB/")
    Corner_a = os.path.join(BasePath, "Ca.csv")
    Corner_b = os.path.join(BasePath, "Cb.csv")
    H4_labels = ReadLabelsTest(LabelsPath)
    Ca_lables = ReadLabelsTest(Corner_a)
    Cb_lables = ReadLabelsTest(Corner_b)
    
    H4pt_list = []
    H4pt_batch = []
    Patched_batch = []
    Ca_batch = []
    Cb_batch = []
    Pa_batch = []
    Pb_list = []
    Ia_list = []
    Ca_list = []
    Cb_list = []
    
    for filename in os.listdir(Patch_a):
        Image1 = os.path.join(Patch_a, filename)
        Image2 = os.path.join(Patch_b, filename)
        Image_Ia = os.path.join(Ia, filename)
        patch_1 = cv2.imread(Image1, cv2.IMREAD_GRAYSCALE)
        patch_2 = cv2.imread(Image2, cv2.IMREAD_GRAYSCALE)
        Ia_img = cv2.imread(Image_Ia)
        stacked_image = torch.cat([torch.from_numpy(patch_1), torch.from_numpy(patch_2)], axis=0)
        stacked_image = stacked_image.view(2, 128, 128).float()
        stacked_image = StandardizeInputs(stacked_image)
        H4pt = H4_labels[filename]
        Ca = Ca_lables[filename]
        Cb = Cb_lables[filename]
        
        H4pt_list.append(H4pt)
        Pb_list.append(patch_2)
        Ia_list.append(Ia_img)
        Ca_list.append(Ca)
        Cb_list.append(Cb)
        
        Patched_batch.append(stacked_image.clone().detach())
        H4pt_batch.append(torch.tensor(H4pt, dtype=torch.float32).clone().detach())
        Ca_batch.append(torch.tensor(Ca, dtype=torch.float32).clone().detach())
        Cb_batch.append(torch.tensor(Cb, dtype=torch.float32).clone().detach())
        Pa_batch.append(torch.tensor(patch_1, dtype=torch.double).clone().detach())

    Patched_batch_torch = torch.stack(Patched_batch)
    H4pt_batch_torch = torch.stack(H4pt_batch)
    Ca_batch_torch = torch.stack(Ca_batch)
    Cb_batch_torch = torch.stack(Cb_batch)
    Pa_batch_torch = torch.stack(Pa_batch)
    
    LabelsPathPred = ROOT_DIR / "PhamHungSon_15_CNN_TxtFiles" / "PhamHungSon_15_CNN_PredOutUnsup.txt"
    LabelsPathPred.parent.mkdir(parents=True, exist_ok=True)
    ResultsPath = ROOT_DIR / "PhamHungSon_15_CNN_Results" / "PhamHungSon_15_CNN_Unsupervised_test"
    PredH4pt_list, PredPb_list = TestOperation(Patched_batch_torch, H4pt_batch_torch, Ca_batch_torch, Cb_batch_torch, Pa_batch_torch, LabelsPathPred, ModelPath, H4pt_list)
    
    VisuaizePatch(PredH4pt_list, Ca_list, Cb_list, Ia_list, ResultsPath)
    
if __name__ == "__main__":
    main()

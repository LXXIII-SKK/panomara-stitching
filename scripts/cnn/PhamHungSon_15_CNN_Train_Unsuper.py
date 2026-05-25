#!/usr/bin/env python

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print("TensorBoard is not installed; continuing without event logs.")

        def add_scalar(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass
from torchvision import transforms
from Network.PhamHungSon_15_CNN_Network_unsuper import HomographyModel, LossFn
import os
import random
from Misc.PhamHungSon_15_CNN_MiscUtils import *
from Misc.PhamHungSon_15_CNN_DataUtils import *
import argparse
import torchvision.io as io
from pathlib import Path


def find_data_root(root_dir):
    for candidate in (root_dir, *root_dir.parents):
        if (candidate / "data" / "cnn").exists():
            return candidate
    return root_dir.parents[1]


def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, Process):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []
    C_a = []
    P_a = []
    C_b = []
    P_b = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        if Process == "Validation":
            LabelsPath = os.path.join(BasePath, "Val_synthetic/H4.csv")
            TrainCoordinates = ReadLabels(LabelsPath)
            DirNamesTrain = os.path.join(BasePath, "Val_synthetic/PA/")
            original_warped_image_path = os.path.join(BasePath, "Val_synthetic/PB/")
            point_patch_2_path = os.path.join(BasePath, "Val_synthetic/Cb.csv")
            point_patch_1_path = os.path.join(BasePath, "Val_synthetic/Ca.csv")
            point_patch_2 = ReadLabels(point_patch_2_path)
            point_patch_1 = ReadLabels(point_patch_1_path)
        
        elif Process == "Train":
            LabelsPath = os.path.join(BasePath, "Train_synthetic/H4.csv")
            TrainCoordinates = ReadLabels(LabelsPath)
            DirNamesTrain = os.path.join(BasePath, "Train_synthetic/PA/")
            original_warped_image_path = os.path.join(BasePath, "Train_synthetic/PB/")
            point_patch_2_path = os.path.join(BasePath, "Train_synthetic/Cb.csv")
            point_patch_1_path = os.path.join(BasePath, "Train_synthetic/Ca.csv")
            point_patch_2 = ReadLabels(point_patch_2_path)
            point_patch_1 = ReadLabels(point_patch_1_path)
            
        else:
            raise ValueError(f"Invalid value for 'Process': {Process}. It should be 'Train' or 'Validation'.")
            
        selected_Image = random.choice(os.listdir(DirNamesTrain))
        
        if selected_Image in TrainCoordinates:
            original_image_path = os.path.join(DirNamesTrain, selected_Image)
            patched_image = io.read_image(original_image_path)
            original_warped_image_path = os.path.join(original_warped_image_path, selected_Image)
            warped_image = io.read_image(original_warped_image_path)
            h4pt = TrainCoordinates[selected_Image]
            C4pt_2 = point_patch_2[selected_Image]
            C4pt_1 = point_patch_1[selected_Image]
        else:
            continue
        stacked_image = torch.cat([patched_image, warped_image], axis=0)
        stacked_image = stacked_image.view(2, 128, 128).float()
        
        ImageNum += 1

        # Normalize to [-1, 1]
        normalized_image = (stacked_image / 255.0 - 0.5) / 0.5

        if Process == "Train":
            # Real-time online brightness and contrast augmentation to combat overfitting
            brightness_shift = random.uniform(-0.1, 0.1)
            contrast_scale = random.uniform(0.9, 1.1)
            normalized_image = torch.clamp(normalized_image * contrast_scale + brightness_shift, -1.0, 1.0)

        I1Batch.append(normalized_image.clone().detach())
        CoordinatesBatch.append(torch.tensor(h4pt).clone().detach())
        C_a.append(torch.tensor(C4pt_1).clone().detach())
        C_b.append(torch.tensor(C4pt_2).clone().detach())
        P_a.append(torch.tensor(patched_image, dtype=torch.float64).clone().detach())
        P_b.append(torch.tensor(warped_image, dtype=torch.float64).clone().detach())

    return torch.stack(I1Batch), torch.stack(CoordinatesBatch), torch.stack(C_a), torch.stack(C_b), torch.stack(P_a), torch.stack(P_b)

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


# =====================================================
# SAFE SAVE & PRUNING UTILITIES
# =====================================================

def PruneCheckpoints(CheckPointPath, max_keep=3):
    try:
        import glob
        files = glob.glob(os.path.join(CheckPointPath, "*.ckpt"))
        files.sort(key=os.path.getmtime)
        if len(files) > max_keep:
            to_delete = files[:-max_keep]
            for f in to_delete:
                try:
                    os.remove(f)
                    print(f"Pruned old checkpoint to save disk space: {f}")
                except Exception as e:
                    print(f"Warning: could not prune {f}: {e}")
    except Exception as e:
        print(f"Warning: error during checkpoint pruning: {e}")

def SafeSaveCheckpoint(state, SaveName, CheckPointPath):
    TempSaveName = SaveName + ".tmp"
    try:
        torch.save(state, TempSaveName)
        if os.path.exists(SaveName):
            try:
                os.remove(SaveName)
            except:
                pass
        os.rename(TempSaveName, SaveName)
        print("Saved checkpoint successfully:", SaveName)
        PruneCheckpoints(CheckPointPath, max_keep=3)
    except (IOError, RuntimeError, OSError) as e:
        print(f"\n[CRITICAL WARNING] Checkpoint save failed: {e}")
        print("This is likely due to low disk space (C: drive has limited free space) or a file lock.")
        print("Skipping checkpoint save to prevent interrupting your training run.\n")
        if os.path.exists(TempSaveName):
            try:
                os.remove(TempSaveName)
            except:
                pass

def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Predict output with forward pass
    model = HomographyModel().to(device)

    Optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(os.path.join(CheckPointPath, LatestFile + ".ckpt"), map_location=device)
        StartEpoch = int(CheckPoint.get("epoch", -1)) + 1
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")
    
    loss_vs_epoch = []
    loss_vs_iteration = []
    
    for Epochs in range(StartEpoch, NumEpochs):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in range(NumIterationsPerEpoch):
            model.train()

            I1Batch, CoordinatesBatch, Ca, Cb, Pa, Pb = GenerateBatch(
                BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, "Train")

            I1Batch = I1Batch.to(device).float()
            CoordinatesBatch = CoordinatesBatch.to(device).float()
            Ca = Ca.to(device).float()
            Cb = Cb.to(device).float()
            Pa = Pa.to(device).float()
            Pb = Pb.to(device).float()

            # Predict output with forward pass
            PbPredicted, H4ptPredicted = model(I1Batch, CoordinatesBatch, Ca, Cb, Pa)

            PbPredicted =  PbPredicted
            LossThisBatch = LossFn(PbPredicted, Pb)

            loss_vs_iteration.append(LossThisBatch.item())
            
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = os.path.join(
                    CheckPointPath,
                    f"PhamHungSon_15_CNN_epoch_{Epochs:03d}_iter_{PerEpochCounter:04d}_model.ckpt",
                )

                SafeSaveCheckpoint(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                    CheckPointPath,
                )


            model.eval()
            with torch.no_grad():
                validation_batch, validation_labels, VCa, VCb, VPa, VPb = GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, "Validation")
                validation_batch = validation_batch.to(device).float()
                validation_labels = validation_labels.to(device).float()
                VCa = VCa.to(device).float()
                VCb = VCb.to(device).float()
                VPa = VPa.to(device).float()
                VPb = VPb.to(device).float()
                result = model.validation_step(validation_batch, validation_labels, VCa, VCb, VPa, VPb)
            
            # Tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                result["val_loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        average_epoch_loss = sum(loss_vs_iteration[-NumIterationsPerEpoch:]) / NumIterationsPerEpoch
        loss_vs_epoch.append(average_epoch_loss)
        Writer.add_scalar("LossEveryEpoch", average_epoch_loss, Epochs,)

        # Save model every epoch
        SaveName = os.path.join(
            CheckPointPath,
            f"PhamHungSon_15_CNN_epoch_{Epochs:03d}_model.ckpt",
        )
        SafeSaveCheckpoint(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
            CheckPointPath,
        )

    
def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    ROOT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = find_data_root(ROOT_DIR)

    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default=str(PROJECT_ROOT / "data" / "cnn" / "split"),
        help="Base path of generated CNN split data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default=str(ROOT_DIR / "PhamHungSon_15_CNN_CheckpointsUnsup5000"),
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=20,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=256,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default=str(ROOT_DIR / "PhamHungSon_15_CNN_LogsUnsup5000"),
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = os.path.join(Args.CheckPointPath, "")
    LogsPath = Args.LogsPath

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath)
  
if __name__ == "__main__":
    main()

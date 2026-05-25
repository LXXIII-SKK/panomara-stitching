
import os
import sys
import csv
import torch

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath, CheckPointPath):

    TrainPath = os.path.join(BasePath, "Train_synthetic/PA")

    DirNamesTrain = os.listdir(TrainPath)

    LabelsPathTrain = os.path.join(BasePath, "Train_synthetic/H4.csv")

    TrainLabels = ReadLabels(LabelsPathTrain)

    if not os.path.isdir(CheckPointPath):
        os.makedirs(CheckPointPath)

    SaveCheckPoint = 100
    NumTestRunsPerEpoch = 5
    ImageSize = [128, 128, 1]

    NumTrainSamples = len(DirNamesTrain)

    print("\n[DEBUG SetupAll]")
    print("TrainPath:", TrainPath)
    print("NumTrainSamples:", NumTrainSamples)

    return (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainLabels,
        10,
    )

def ReadLabels(LabelsPathTrain):
    if not (os.path.isfile(LabelsPathTrain)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
        sys.exit()
    labels_images = {}

    with open(LabelsPathTrain, 'r') as labels_file:
        csv_reader = csv.reader(labels_file)
        for row in csv_reader:
            image_name = row[0]
            labels = [float(label) for label in row[1:]]
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            labels_images[image_name] = labels_tensor

    return labels_images

def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./PhamHungSon_15_CNN_TxtFiles/PhamHungSon_15_CNN_DirNames.txt with full path to all image files without extension
    """
    DirNamesTrain = os.listdir(BasePath)

    return DirNamesTrain


def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./PhamHungSon_15_CNN_TxtFiles/PhamHungSon_15_CNN_DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames

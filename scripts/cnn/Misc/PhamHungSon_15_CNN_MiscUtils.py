
import time
import glob
import os
import sys
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True


def tic():
    StartTime = time.time()
    return StartTime


def toc(StartTime):
    return time.time() - StartTime


def remap(x, oMin, oMax, iMin, iMax):
    # Taken from https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratios
    # range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if iMin == iMax:
        print("Warning: Zero output range")
        return None

    # portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    result = np.add(np.divide(np.multiply(x - iMin, oMax - oMin), iMax - iMin), oMin)

    return result


def FindLatestModel(CheckPointPath):
    path_str = os.path.join(str(CheckPointPath), "")
    FileList = glob.glob(path_str + "*.ckpt")
    if not FileList:
        return None
    LatestFile = max(FileList, key=os.path.getmtime)
    basename = os.path.basename(LatestFile)
    return basename.replace(".ckpt", "")



def convertToOneHot(vector, n_labels):
    return np.equal.outer(vector, np.arange(n_labels)).astype(np.float)

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "PhamHungSon_15_08_CNN.ipynb"

def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    # Fix Cell 16 (supervised training command)
    nb['cells'][16]['source'] = [
        "if IN_COLAB:\n",
        "    !python /content/Code/Train_Sup.py \\\n",
        "      --BasePath=/content/data/synthetic \\\n",
        "      --CheckPointPath=/content/checkpoints/ \\\n",
        "      --LogsPath=/content/logs/ \\\n",
        "      --NumEpochs=25 \\\n",
        "      --MiniBatchSize=8 \\\n",
        "      --DivTrain=5 \\\n",
        "      --LoadCheckPoint=0\n",
        "else:\n",
        "    # Run local train\n",
        "    !python {CODE_ROOT}/Train_Sup.py \\\n",
        "      --NumEpochs=1 \\\n",
        "      --MiniBatchSize=4 \\\n",
        "      --DivTrain=1 \\\n",
        "      --LoadCheckPoint=0\n"
    ]
    
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print("Cleaned up Cell 16 successfully!")

if __name__ == "__main__":
    main()

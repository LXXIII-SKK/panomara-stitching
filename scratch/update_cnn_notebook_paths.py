import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "PhamHungSon_15_08_CNN.ipynb"

def main():
    if not NOTEBOOK_PATH.exists():
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return
        
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb['cells']
    
    # Let's add a cell at the very beginning (or index 1) that defines the environment detection
    env_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Detect environment (Google Colab vs. Local Environment)\n",
            "import os\n",
            "import sys\n",
            "\n",
            "IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')\n",
            "\n",
            "if IN_COLAB:\n",
            "    print(\"Running in Google Colab environment.\")\n",
            "    DATA_ROOT = \"/content/data\"\n",
            "    CODE_ROOT = \"/content/Code\"\n",
            "    CHECKPOINT_ROOT = \"/content/checkpoints\"\n",
            "    LOGS_ROOT = \"/content/logs\"\n",
            "else:\n",
            "    print(\"Running in Local Environment (fits system architecture).\")\n",
            "    # Aligns to your local data and scripts directories\n",
            "    DATA_ROOT = \"../data/cnn\"\n",
            "    CODE_ROOT = \"../scripts/cnn\"\n",
            "    CHECKPOINT_ROOT = \"../scripts/cnn/CheckpointsSupP2\"\n",
            "    LOGS_ROOT = \"../scripts/cnn/LogsSupP2\"\n"
        ]
    }
    
    # We will insert it at index 1
    cells.insert(1, env_cell)
    
    # Now let's scan other cells and replace hardcoded /content/ paths with dynamic paths
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_lines = []
            for line in source_lines:
                # Replace train_path / val_path setup in cell 9 (which is now cell 10)
                if "'/content/data/" in line or "\"/content/data/" in line:
                    line = line.replace("'/content/data/split/Train'", "os.path.join(DATA_ROOT, 'split/Train')")
                    line = line.replace("'/content/data/synthetic/Train_synthetic'", "os.path.join(DATA_ROOT, 'split/Train_synthetic')")
                    line = line.replace("'/content/data/split/Val'", "os.path.join(DATA_ROOT, 'split/Val')")
                    line = line.replace("'/content/data/synthetic/Val_synthetic'", "os.path.join(DATA_ROOT, 'split/Val_synthetic')")
                    line = line.replace("'/content/data/split/Test'", "os.path.join(DATA_ROOT, 'split/Test')")
                    line = line.replace("'/content/data/synthetic/Test_synthetic'", "os.path.join(DATA_ROOT, 'split/Test_synthetic')")
                
                # Replace specific cv2.imread path
                if "'/content/data/synthetic/Train_synthetic/PA/1a.jpg'" in line:
                    line = line.replace("'/content/data/synthetic/Train_synthetic/PA/1a.jpg'", "os.path.join(DATA_ROOT, 'split/Train_synthetic/PA/1a.jpg')")
                    
                # For shell commands, we will write them to check if IN_COLAB first, or run them locally if needed
                if line.startswith("!python /content/Code/Train_Sup.py"):
                    line = (
                        "if IN_COLAB:\n"
                        "    !python /content/Code/Train_Sup.py --BasePath=/content/data/synthetic --CheckPointPath=/content/checkpoints/ --LogsPath=/content/logs/ --NumEpochs=25 --MiniBatchSize=8 --DivTrain=5 --LoadCheckPoint=0\n"
                        "else:\n"
                        "    # Run local train\n"
                        "    !python {CODE_ROOT}/Train_Sup.py --NumEpochs=1 --MiniBatchSize=4 --DivTrain=1 --LoadCheckPoint=0\n"
                    )
                elif line.startswith("!python /content/Code/Test_Sup.py"):
                    line = (
                        "if IN_COLAB:\n"
                        "    !python /content/Code/Test_Sup.py --ModelPath '/content/checkpoints/0a0model.ckpt' --BasePath /content/data/synthetic/Test_synthetic/ --LabelsPath '/content/data/synthetic/Test_synthetic/H4.csv'\n"
                        "else:\n"
                        "    # Run local test\n"
                        "    !python {CODE_ROOT}/Test_Sup.py\n"
                    )
                elif line.startswith("!python /content/Code/Wrapper_Sup.py"):
                    line = (
                        "if IN_COLAB:\n"
                        "    !python /content/Code/Wrapper_Sup.py\n"
                        "else:\n"
                        "    # Run local wrapper\n"
                        "    !python {CODE_ROOT}/Wrapper_Sup.py\n"
                    )
                elif line.startswith("!mkdir -p /content/data/"):
                    line = (
                        "if IN_COLAB:\n"
                        "    " + line + "\n"
                        "else:\n"
                        "    print('Local directories already verified in data/cnn/')\n"
                    )
                elif line.startswith("!pip install"):
                    line = (
                        "if IN_COLAB:\n"
                        "    " + line + "\n"
                        "else:\n"
                        "    print('Local packages already verified')\n"
                    )
                    
                new_lines.append(line)
            cell['source'] = new_lines
            
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully updated CNN notebook paths: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    main()

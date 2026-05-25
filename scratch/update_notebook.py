import json
import os

notebook_path = r"c:\Users\PC\Downloads\Project\notebooks\PhamHungSon_15_08_CNN.ipynb"

if not os.path.exists(notebook_path):
    print("Notebook path not found:", notebook_path)
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

updated = False
for idx, cell in enumerate(nb.get("cells", [])):
    source_text = "".join(cell.get("source", []))
    if "Train_Sup.py" in source_text and "MiniBatchSize" in source_text:
        cell["source"] = [
            "if IN_COLAB:\n",
            "    # Standard supervised train in Google Colab\n",
            "    !python /content/Code/Train_Sup.py \\\n",
            "      --BasePath=/content/data/synthetic \\\n",
            "      --CheckPointPath=/content/checkpoints/ \\\n",
            "      --LogsPath=/content/logs/ \\\n",
            "      --NumEpochs=25 \\\n",
            "      --MiniBatchSize=64 \\\n",
            "      --DivTrain=1 \\\n",
            "      --LoadCheckPoint=0 \\\n",
            "      --Optimizer=Adam \\\n",
            "      --LR=0.001 \\\n",
            "      --Backbone=ConvNet\n",
            "else:\n",
            "    # Run local train with upgraded Adam optimizer and stable parameters\n",
            "    # You can set --Backbone=ResNet18 to run the state-of-the-art ResNet-18 model!\n",
            "    !python {CODE_ROOT}/Train_Sup.py \\\n",
            "      --NumEpochs=5 \\\n",
            "      --MiniBatchSize=64 \\\n",
            "      --DivTrain=1 \\\n",
            "      --LoadCheckPoint=0 \\\n",
            "      --Optimizer=Adam \\\n",
            "      --LR=0.0005 \\\n",
            "      --Backbone=ConvNet\n"
        ]
        updated = True
        print(f"Successfully updated cell index {idx} with Backbone argument.")
        break

if updated:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook successfully saved.")
else:
    print("Error: Cell containing 'Train_Sup.py' was not found.")

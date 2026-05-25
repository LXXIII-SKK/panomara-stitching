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
    if "Test_Sup.py" in source_text and "ModelPath" in source_text:
        cell["source"] = [
            "if IN_COLAB:\n",
            "    !python /content/Code/Test_Sup.py \\\n",
            "      --ModelPath='/content/checkpoints/0model.ckpt' \\\n",
            "      --BasePath=/content/data/synthetic/Test_synthetic/ \\\n",
            "      --LabelsPath='/content/data/synthetic/Test_synthetic/H4.csv' \\\n",
            "      --Backbone=ConvNet\n",
            "else:\n",
            "    # Run local test. Set --Backbone=ResNet18 if you trained the ResNet-18 model!\n",
            "    !python {CODE_ROOT}/Test_Sup.py \\\n",
            "      --Backbone=ConvNet\n"
        ]
        updated = True
        print(f"Successfully updated cell index {idx} containing 'Test_Sup.py'.")
        break

if updated:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook successfully saved.")
else:
    print("Error: Cell containing 'Test_Sup.py' was not found.")

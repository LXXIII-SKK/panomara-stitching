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
        
    print(f"Notebook: {NOTEBOOK_PATH.name}")
    print(f"Total cells: {len(nb['cells'])}")
    
    markdown_cells = 0
    code_cells = 0
    
    for idx, cell in enumerate(nb['cells']):
        cell_type = cell.get('cell_type', '')
        source = "".join(cell.get('source', []))
        
        if cell_type == 'markdown':
            markdown_cells += 1
            # Print headers (lines starting with #)
            for line in source.split('\n'):
                if line.startswith('#'):
                    print(f"  Cell {idx} [MD]: {line}")
        elif cell_type == 'code':
            code_cells += 1
            # Print first few lines of code if it imports or defines models
            lines = [l for l in source.split('\n') if l.strip()]
            if any(term in "".join(lines).lower() for term in ['import ', 'class ', 'def ', 'train', 'net']):
                summary = "; ".join(lines[:3])
                print(f"  Cell {idx} [Code]: {summary[:120]}...")

if __name__ == "__main__":
    main()

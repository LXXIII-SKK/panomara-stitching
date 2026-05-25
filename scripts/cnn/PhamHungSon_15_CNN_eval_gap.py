import sys, torch
from pathlib import Path
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parents[1]

sys.path.insert(0, str(ROOT_DIR / 'Network'))
from PhamHungSon_15_CNN_Network import HomographyModel, denormalize_prediction

# We need HomographyPairDataset and MetricAccumulator from Train_Sup
sys.path.insert(0, str(ROOT_DIR))
import importlib.util
spec = importlib.util.spec_from_file_location('train_sup', str(Path(__file__).resolve().parent / 'PhamHungSon_15_CNN_Train_Sup.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = ROOT_DIR / 'PhamHungSon_15_CNN_CheckpointsSupRegularized' / 'PhamHungSon_15_CNN_best_model.ckpt'
ck = torch.load(ckpt_path, map_location=device)
epoch = ck.get('epoch', -1) + 1
best_mce = ck.get('best_mce', 999)

model = HomographyModel('ConvNet', dropout=0.5).to(device)
model.load_state_dict(ck['model_state_dict'])
model.eval()

print(f"Loaded best checkpoint: {ckpt_path}")
print(f"Epoch {epoch}, reported best MCE: {best_mce:.4f}px")
print()

base = PROJECT_ROOT / 'data' / 'cnn' / 'split'

for split, name in [('Train_synthetic', 'TRAIN'), ('Val_synthetic', 'VAL')]:
    ds = mod.HomographyPairDataset(base, split, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    acc = mod.MetricAccumulator()
    with torch.no_grad():
        for x, y, _ in loader:
            pred = model(x.to(device))
            pred_px = denormalize_prediction(pred, 42.0)
            acc.update(pred_px.cpu(), y)
    m = acc.compute()
    print(f"{name} ({len(ds)} samples):")
    print(f"  MCE     = {m['mce_px']:.4f} px")
    print(f"  MAE     = {m['mae_px']:.4f} px")
    print(f"  Acc@1px = {m['acc_1px']:.2f}%")
    print(f"  Acc@3px = {m['acc_3px']:.2f}%")
    print(f"  Acc@5px = {m['acc_5px']:.2f}%")
    print(f"  Outlier >10px = {m['outlier_10px']:.2f}%")
    print()

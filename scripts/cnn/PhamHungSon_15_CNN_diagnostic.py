import sys, torch, torch.nn.functional as F, csv, os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR / 'Network'))
from PhamHungSon_15_CNN_Network import HomographyModel, train_loss_fn, denormalize_prediction
import torchvision.io as io
from torchvision.io import ImageReadMode

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

base = str(PROJECT_ROOT / 'data' / 'cnn' / 'split' / 'Train_synthetic')
with open(os.path.join(base, 'H4.csv')) as f:
    rows = list(csv.reader(f))[:9]

xs, ys = [], []
for row in rows:
    fname = row[0]
    pa = io.read_image(os.path.join(base, 'PA', fname), mode=ImageReadMode.GRAY).float()
    pb = io.read_image(os.path.join(base, 'PB', fname), mode=ImageReadMode.GRAY).float()
    x = torch.cat([pa, pb], 0)
    x = (x / 255.0 - 0.5) / 0.5
    xs.append(x)
    ys.append(torch.tensor([float(v) for v in row[1:9]]))

x = torch.stack(xs).to(device)
y = torch.stack(ys).to(device)
y_norm = y / 42.0
print(f'y pixel range: [{float(y.min()):.1f}, {float(y.max()):.1f}]')
print(f'y norm range:  [{float(y_norm.min()):.3f}, {float(y_norm.max()):.3f}]')
print()

model = HomographyModel('ConvNet', dropout=0.5).to(device)

# Show architecture
total_params = sum(p.numel() for p in model.parameters())
print(f'Total params: {total_params:,}')
print()

# Eval mode (Dropout off) - sanity: can it overfit?
model.eval()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

print('=== OVERFIT TEST (eval/no-dropout) ===')
for step in range(301):
    pred = model(x)
    loss = train_loss_fn(pred, y, label_scale=42.0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        with torch.no_grad():
            pred_px = denormalize_prediction(pred, 42.0)
            mce = torch.linalg.norm((pred_px-y).view(-1,4,2), dim=2).mean().item()
            print(f'  Step {step:03d}: loss={loss.item():.6f}  pred_range=[{pred.min().item():.3f},{pred.max().item():.3f}]  MCE={mce:.3f}px')

print()
# Now train mode
model2 = HomographyModel('ConvNet', dropout=0.5).to(device)
model2.train()
opt2 = torch.optim.AdamW(model2.parameters(), lr=5e-4, weight_decay=1e-4)

print('=== FULL TRAIN MODE (Dropout ON) - 200 steps ===')
for step in range(201):
    pred = model2(x)
    loss = train_loss_fn(pred, y, label_scale=42.0)
    opt2.zero_grad(); loss.backward(); opt2.step()
    if step % 40 == 0:
        model2.eval()
        with torch.no_grad():
            pred_px = denormalize_prediction(model2(x), 42.0)
            mce = torch.linalg.norm((pred_px-y).view(-1,4,2), dim=2).mean().item()
        model2.train()
        print(f'  Step {step:03d}: loss={loss.item():.6f}  MCE={mce:.3f}px')

print('\nDone.')

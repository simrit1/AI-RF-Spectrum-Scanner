"""Train the spectrogram CNN from labeled IQ .npz files."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from cnn_rf import build_cnn, iq_to_spectrogram, load_npz_example, CNN_CLASS_NAMES, DEFAULT_SPEC_SIZE

p = argparse.ArgumentParser(description="Train Pluto spectrogram CNN from labeled IQ captures")
p.add_argument("dataset", help="Dataset root containing label subfolders of .npz files")
p.add_argument("--out", default="models/house_rf_cnn.pt")
p.add_argument("--epochs", type=int, default=18)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--image-size", type=int, default=DEFAULT_SPEC_SIZE)
p.add_argument("--seed", type=int, default=570)
a = p.parse_args()

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
except ImportError as exc:
    raise SystemExit("PyTorch required. Install with: pip install torch") from exc

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
root = Path(a.dataset)
files = sorted(root.rglob("*.npz"))
if not files:
    raise SystemExit(f"No .npz examples found under {root}")

# Read labels once and keep only supported CNN taxonomy.
items: List[Tuple[Path,str]] = []
for path in files:
    try:
        _iq, _sr, _off, _bw, label = load_npz_example(str(path))
    except Exception as exc:
        print("Skipping", path, exc); continue
    if label in CNN_CLASS_NAMES:
        items.append((path, label))
if not items:
    raise SystemExit("No valid labeled CNN examples found")
classes = sorted({label for _, label in items})
if len(classes) < 2:
    raise SystemExit("Need at least two CNN classes")
class_to_idx = {c:i for i,c in enumerate(classes)}

class IQDataset(Dataset):
    def __len__(self): return len(items)
    def __getitem__(self, idx):
        path, label = items[idx]
        iq, sr, off, bw, _ = load_npz_example(str(path))
        img = iq_to_spectrogram(iq, sr, off, bw, a.image_size)
        # Mild training augmentation: time/frequency rolls and amplitude-independent image noise.
        if random.random() < 0.7:
            img = np.roll(img, random.randint(-3,3), axis=0)
        if random.random() < 0.7:
            img = np.roll(img, random.randint(-4,4), axis=1)
        if random.random() < 0.5:
            img = np.clip(img + np.random.normal(0, 0.015, img.shape), 0, 1).astype(np.float32)
        return torch.from_numpy(img[None,:,:]), class_to_idx[label]

ds = IQDataset()
val_n = max(1, int(0.20 * len(ds)))
train_n = len(ds) - val_n
if train_n < 2:
    raise SystemExit("Dataset too small")
gen = torch.Generator().manual_seed(a.seed)
train_ds, val_ds = random_split(ds, [train_n,val_n], generator=gen)
train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_cnn(len(classes)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
best = -1.0; best_state = None

for epoch in range(1, a.epochs+1):
    model.train(); train_loss=0.0; total=0
    for xb,yb in train_loader:
        xb=xb.to(device); yb=yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits=model(xb); loss=criterion(logits,yb); loss.backward(); opt.step()
        train_loss += float(loss.item()) * len(xb); total += len(xb)
    model.eval(); correct=0; vtotal=0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb=xb.to(device); yb=yb.to(device)
            pred=model(xb).argmax(1); correct += int((pred==yb).sum().item()); vtotal += len(xb)
    acc = correct/max(1,vtotal)
    print(f"epoch {epoch:02d}/{a.epochs} loss={train_loss/max(1,total):.4f} val_acc={acc:.3f}")
    if acc > best:
        best=acc
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "state_dict": best_state if best_state is not None else model.state_dict(),
    "class_names": classes,
    "image_size": a.image_size,
    "source": "user-labeled-spectrogram-cnn",
    "validation_accuracy": float(best),
    "warning": "Validation accuracy reflects this local dataset only; test on separate captures before trusting it broadly.",
}, a.out)
print(f"Saved {a.out} (best validation accuracy={best:.3f})")

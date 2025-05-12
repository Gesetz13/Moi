import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 🔧 Pfad zur Projektwurzel hinzufügen
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets.HarvardHSIDataset import HarvardHSIDataset
from models.simple_rgb2hsi import RGB2HSI

# 📊 PSNR berechnen
def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# 🧭 SAM (Spectral Angle Mapper)
def compute_sam(pred, target):
    # [C, H, W] → [C, N]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    dot = torch.sum(pred * target, dim=0)
    norm_pred = torch.norm(pred, dim=0)
    norm_target = torch.norm(target, dim=0)
    sam = torch.acos(torch.clamp(dot / (norm_pred * norm_target + 1e-8), -1.0, 1.0))
    return torch.mean(sam).item()

# 🔧 Pfade
project_root = Path(__file__).resolve().parent.parent
checkpoint_path = project_root / "outputs" / "checkpoints" / "rgb2hsi_baseline.pth"
data_dir = project_root / "data" / "harvard"
calib_path = data_dir / "calib.txt"

# 📂 Dataset + Modell
dataset = HarvardHSIDataset(mat_dir=data_dir, calib_path=calib_path)
model = RGB2HSI(out_channels=31)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# 🔁 Auswertung
results = []
for i in tqdm(range(len(dataset)), desc="Bewerte Modell"):
    rgb, hsi_true, name = dataset[i]
    with torch.no_grad():
        pred = model(rgb.unsqueeze(0)).squeeze(0)

    mse = torch.mean((pred - hsi_true) ** 2).item()
    psnr = compute_psnr(pred, hsi_true)
    sam = compute_sam(pred, hsi_true)

    results.append({
        "filename": name,
        "mse": mse,
        "psnr": psnr,
        "sam": sam
    })

# 📊 Ergebnisse als Tabelle
df = pd.DataFrame(results)
df_sorted = df.sort_values(by="mse")

# 🔍 Zeige Top- und schlechteste Ergebnisse
print("\n🎯 Top-5 rekonstruiert (niedrigster Fehler):")
print(df_sorted.head(5))

print("\n⚠️ Schlechteste 5 (höchster Fehler):")
print(df_sorted.tail(5))

# 💾 Optional speichern
df.to_csv(project_root / "outputs" / "eval_results.csv", index=False)
print(f"\n✅ Alle Ergebnisse gespeichert unter: outputs/eval_results.csv")

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Importpfad zur Projektwurzel hinzufügen
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Eigene Module importieren
from datasets.HarvardHSIDataset import HarvardHSIDataset
from models.simple_rgb2hsi import RGB2HSI

# 🔧 Robuste Pfade
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "harvard"
calib_path = data_dir / "calib.txt"

# 🔍 Ausgabe zur Kontrolle
print(f"[INFO] Arbeitsverzeichnis: {os.getcwd()}")
print(f"[INFO] Datensatzpfad: {data_dir}")
print(f"[INFO] Kalibrierung vorhanden: {calib_path.exists()}")

# 📦 Dataset laden
dataset = HarvardHSIDataset(
    mat_dir=data_dir,
    calib_path=calib_path if calib_path.exists() else None
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 🧠 Modell, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RGB2HSI(out_channels=31).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 🔁 Training
epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for rgb, hsi, _ in dataloader:
        rgb = rgb.to(device)
        hsi = hsi.to(device)

        pred = model(rgb)
        loss = criterion(pred, hsi)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[{epoch+1}/{epochs}] Loss: {epoch_loss / len(dataloader):.6f}")

# 💾 Modell speichern
save_dir = project_root / "outputs" / "checkpoints"
save_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), save_dir / "rgb2hsi_baseline.pth")
print(f"[✅] Modell gespeichert unter: {save_dir / 'rgb2hsi_baseline.pth'}")

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# 🔧 Projektwurzel zum Importpfad hinzufügen
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 📦 Eigene Module
from datasets.HarvardHSIDataset import HarvardHSIDataset
from models.simple_rgb2hsi import RGB2HSI

# 🔧 Pfade definieren
project_root = Path(__file__).resolve().parent.parent
checkpoint_path = project_root / "outputs" / "checkpoints" / "rgb2hsi_baseline.pth"
data_dir = project_root / "data" / "harvard"
calib_path = data_dir / "calib.txt"

# 📂 Dataset laden
dataset = HarvardHSIDataset(mat_dir=data_dir, calib_path=calib_path)
rgb, hsi_true, filename = dataset[0]  # Beispielbild 0

# 🧠 Modell laden
model = RGB2HSI(out_channels=31)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# 🔮 Vorhersage
with torch.no_grad():
    pred = model(rgb.unsqueeze(0)).squeeze(0)  # [1, 3, H, W] → [31, H, W]

# 📊 Vergleich eines Bands
band = 15
true_band = hsi_true[band].numpy()
pred_band = pred[band].numpy()

# 🖼️ Visualisierung
plt.figure(figsize=(15, 5))

# RGB-Bild
plt.subplot(1, 3, 1)
plt.imshow(rgb.permute(1, 2, 0).numpy())
plt.title(f"RGB-Bild (aus HSI)\n{filename}")

# Echtes Spektralband
plt.subplot(1, 3, 2)
plt.imshow(true_band, cmap="gray")
plt.title(f"Echter HSI-Band {band+1}")

# Rekonstruiertes Band
plt.subplot(1, 3, 3)
plt.imshow(pred_band, cmap="gray")
plt.title(f"Vorhergesagt Band {band+1}")

plt.tight_layout()
plt.show()

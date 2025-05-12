import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# 🔧 Projektpfad hinzufügen
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.simple_rgb2hsi import RGB2HSI

# ⚙️ Pfade
project_root = Path(__file__).resolve().parent.parent
img_dir = project_root / "test_images"
checkpoint_path = project_root / "outputs" / "checkpoints" / "rgb2hsi_baseline.pth"

# 📦 Modell laden
model = RGB2HSI(out_channels=31)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# 📐 Transformation: resize + tensor
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# 🔁 Über alle Bilder iterieren
for img_path in sorted(img_dir.glob("*.jpg")):
    print(f"\n🔍 Verarbeite Bild: {img_path.name}")
    
    image = Image.open(img_path).convert("RGB")
    rgb = transform(image).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        pred = model(rgb).squeeze(0)  # [31, H, W]

    # 📊 Statistiken berechnen
    band = 15  # z. B. mittleres Band
    hsi_band = pred[band].numpy()
    rgb_np = rgb.squeeze(0).permute(1, 2, 0).numpy()

    print(f"  → RGB mean: {rgb_np.mean():.4f}, min: {rgb_np.min():.4f}, max: {rgb_np.max():.4f}")
    print(f"  → HSI Band {band+1} mean: {hsi_band.mean():.4f}, min: {hsi_band.min():.4f}, max: {hsi_band.max():.4f}")
    print(f"  → HSI Band {band+1} shape: {hsi_band.shape}")

    # 🖼️ Visualisierung
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_np)
    plt.title(f"Original RGB – {img_path.name}")

    plt.subplot(1, 2, 2)
    plt.imshow(hsi_band, cmap="viridis")
    plt.title(f"Rekonstruiertes HSI – Band {band+1}")
    plt.tight_layout()
    plt.show()

from HarvardHSIDataset import HarvardHSIDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = HarvardHSIDataset(
    mat_dir="./data/harvard",
    calib_path="./data/harvard/calib.txt",
    rgb_bands=(27, 14, 8)  # ca. 700nm, 550nm, 450nm
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

rgb, hsi, name = next(iter(loader))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb[0].permute(1, 2, 0).numpy())
plt.title(f"RGB aus HSI – {name}")

plt.subplot(1, 2, 2)
plt.imshow(hsi[0][15].numpy(), cmap="gray")  # mittleres Spektralband
plt.title("Spektralband 16")
plt.show()

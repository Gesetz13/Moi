from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import glob

class HarvardHSIDataset(Dataset):
    def __init__(self, mat_dir, calib_path=None, rgb_bands=(27, 14, 8), apply_mask=True):
        """
        Args:
            mat_dir (str or Path): Pfad zu .mat-Dateien mit 'ref' und 'lbl'
            calib_path (str or Path): Optionaler Pfad zu calib.txt (für spätere Kalibrierung)
            rgb_bands (tuple): Indizes für (R, G, B) – 0-basiert, z. B. (27,14,8) für 700/550/450nm
            apply_mask (bool): Wenn True, werden Pixel mit lbl == 0 maskiert (auf 0 gesetzt)
        """
        self.mat_dir = Path(mat_dir)
        self.files = sorted(self.mat_dir.glob("*.mat"))
        self.rgb_bands = rgb_bands
        self.apply_mask = apply_mask

        # Kalibrierung laden (optional)
        self.calib = None
        if calib_path:
            self.calib = np.loadtxt(calib_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat_path = self.files[idx]
        mat = sio.loadmat(mat_path)

        hsi = mat["ref"]  # H x W x 31
        mask = mat["lbl"]  # H x W

        hsi = hsi.astype(np.float32)

        if self.calib is not None:
            hsi *= self.calib.reshape(1, 1, -1)

        if self.apply_mask:
            hsi[mask == 0] = 0.0  # Maske anwenden

        # RGB generieren (z. B. B27,R14,G8 → 700/550/450 nm)
        rgb = hsi[:, :, list(self.rgb_bands)]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

        # CHW-Tensoren zurückgeben
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        hsi_tensor = torch.from_numpy(hsi).permute(2, 0, 1)

        return rgb_tensor, hsi_tensor, str(mat_path.name)

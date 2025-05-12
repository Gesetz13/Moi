import torch
import torch.nn as nn

class RGB2HSI(nn.Module):
    def __init__(self, out_channels=31):
        super(RGB2HSI, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 1)  # 31 Kanäle (HSI-Bänder)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

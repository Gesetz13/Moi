# Moi

# RGB → Hyperspectral Image Reconstruction

Dieses Projekt rekonstruiert Hyperspektralbilder (31 Bänder, 420–720 nm) aus RGB-Bildern mithilfe eines Deep-Learning-Modells. Das Modell wurde auf dem Harvard Hyperspectral Image Dataset trainiert.

---

## 🔍 Features

- Eigener PyTorch-Dataset-Loader für `.mat` + `calib.txt`
- Einfaches CNN-Modell (Baseline)
- Training, Evaluation und Visualisierungsskripte
- Test auf beliebigen RGB-Bildern
- PSNR, MSE, SAM-Auswertung

---

## 🗂️ Projektstruktur

```bash
data/             # Harvard HSI Dataset (nicht im Repo enthalten)
datasets/         # PyTorch Dataset-Klassen
models/           # CNN-Modell (RGB → HSI)
train/            # Trainingsskript
eval/             # Evaluation, Visualisierung
test_images/      # Externe RGB-Testbilder
outputs/          # Modell-Checkpoints & Evaluationsergebnisse

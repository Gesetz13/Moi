from pathlib import Path
import scipy.io as sio
import numpy as np

def check_harvard_mat_files(mat_dir):
    mat_dir = Path(mat_dir)
    files = list(mat_dir.glob("*.mat"))
    assert len(files) > 0, f"❌ Keine .mat-Dateien gefunden in {mat_dir}"

    print(f"[INFO] Überprüfe {len(files)} Dateien in {mat_dir}...\n")
    all_ok = True

    for file in files:
        try:
            mat = sio.loadmat(file)
            ref = mat.get("ref", None)
            lbl = mat.get("lbl", None)

            if ref is None or lbl is None:
                print(f"[❌] {file.name}: 'ref' oder 'lbl' fehlt.")
                all_ok = False
                continue

            if ref.ndim != 3:
                print(f"[❌] {file.name}: ref hat keine 3 Dimensionen (shape: {ref.shape})")
                all_ok = False
                continue

            h, w, bands = ref.shape
            print(f"[INFO] {file.name}: Größe: {h}x{w}, Spektralbänder: {bands}")

            if bands != 31:
                print(f"[⚠️] {file.name}: Unerwartete Anzahl von Bändern: {bands} (erwartet: 31)")
                all_ok = False

            if lbl.shape != (h, w):
                print(f"[❌] {file.name}: lbl passt nicht zur ref-Größe (lbl: {lbl.shape}, ref: {h}x{w})")
                all_ok = False

            if np.all(lbl == 0):
                print(f"[⚠️] {file.name}: Maske enthält nur Nullen (keine gültigen Pixel).")

            # RGB-Test (optional)
            try:
                rgb = np.stack([ref[:, :, 27], ref[:, :, 14], ref[:, :, 8]], axis=-1)
                assert rgb.shape == (h, w, 3)
            except Exception as e:
                print(f"[⚠️] {file.name}: RGB-Erzeugung fehlgeschlagen: {e}")
                all_ok = False

        except Exception as e:
            print(f"[❌] Fehler beim Verarbeiten von {file.name}: {e}")
            all_ok = False

    if all_ok:
        print("\n✅ Alle Dateien sind korrekt formatiert und bereit für das Training.")
    else:
        print("\n⚠️ Einige Dateien sind fehlerhaft. Bitte prüfe die obigen Hinweise.")

if __name__ == "__main__":
    check_harvard_mat_files("./data/harvard")

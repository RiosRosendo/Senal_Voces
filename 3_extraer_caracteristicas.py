"""
Script 3 – Extracción de características (LPC → LSF)
──────────────────────────────────────────────────────
Para cada señal preprocesada en voz_preprocesadas/<palabra>/:
  1. Aplica ventana de Hamming (320 muestras, salto 128)
  2. Calcula coeficientes LPC de orden 12 por frame (Levinson-Durbin)
  3. Convierte LPC a LSF (Line Spectral Frequencies)
  4. Guarda en features/<palabra>/XX.npz

Al ejecutar verás:
  • Número de frames extraídos por archivo
  • Gráfica de los LSF a lo largo del tiempo para cada palabra
  • Resumen final con estadísticas de frames por palabra
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import (PALABRAS, LPC_ORDER, FRAME_LEN, HOP,
                   hamming_frames, compute_lpc, lpc_to_lsf, extract_features)

PRE_DIR     = "voz_preprocesadas"
FEAT_DIR    = "features"
PLOT_DIR    = os.path.join("resultados", "caracteristicas")

import scipy.io.wavfile as wav

def cargar_wav(path):
    fs, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483647.0
    if data.ndim > 1:
        data = data[:, 0]
    return fs, data


def graficar_lsf(lsf_matrix, palabra, archivo, plot_path):
    """Muestra los LSF frame a frame como espectrograma."""
    n_frames, order = lsf_matrix.shape
    t = np.arange(n_frames) * HOP / 16000

    fig, ax = plt.subplots(figsize=(10, 4))
    for k in range(order):
        ax.plot(t, lsf_matrix[:, k] / np.pi, alpha=0.6, linewidth=0.8)
    ax.set_title(f'LSF frame a frame – «{palabra}» – {archivo}')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('LSF / π')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    resumen = {}

    for palabra in PALABRAS:
        in_dir   = os.path.join(PRE_DIR,  palabra)
        out_dir  = os.path.join(FEAT_DIR, palabra)
        plt_dir  = os.path.join(PLOT_DIR, palabra)

        if not os.path.exists(in_dir):
            print(f"  [SKIP] {palabra}: sin señales preprocesadas")
            continue

        archivos = sorted(f for f in os.listdir(in_dir) if f.endswith(".wav"))
        if not archivos:
            print(f"  [SKIP] {palabra}: sin archivos")
            continue

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(plt_dir, exist_ok=True)

        print(f"\n  Extrayendo características: «{palabra}»")
        frames_por_archivo = []

        for fname in archivos:
            ruta_in  = os.path.join(in_dir, fname)
            ruta_out = os.path.join(out_dir, fname.replace(".wav", ".npz"))
            plot_path = os.path.join(plt_dir, fname.replace(".wav", ".png"))

            try:
                fs, signal = cargar_wav(ruta_in)
                feats = extract_features(signal, fs=fs,
                                         lpc_order=LPC_ORDER,
                                         frame_len=FRAME_LEN,
                                         hop=HOP)

                n_frames = len(feats['lsf'])
                frames_por_archivo.append(n_frames)

                np.savez_compressed(ruta_out,
                                    lsf=feats['lsf'],
                                    lpc=feats['lpc'],
                                    gain=feats['gain'],
                                    acf=feats['acf'])

                # Gráfica solo para primer archivo de cada palabra
                if fname == archivos[0]:
                    graficar_lsf(feats['lsf'], palabra, fname, plot_path)

                print(f"    ✓ {fname}  → {n_frames} frames")

            except Exception as exc:
                print(f"    ✗ {fname}: {exc}")

        if frames_por_archivo:
            resumen[palabra] = frames_por_archivo
            print(f"  Media frames: {np.mean(frames_por_archivo):.1f}  "
                  f"(min {min(frames_por_archivo)} / max {max(frames_por_archivo)})")

    # ── Resumen global ──
    print(f"\n{'='*55}")
    print(f"  {'Palabra':<12} {'Archivos':>8} {'Frames/arch':>12} {'Total frames':>14}")
    print(f"  {'-'*52}")
    for pal, frames in resumen.items():
        print(f"  {pal:<12} {len(frames):>8} {np.mean(frames):>12.1f} {sum(frames):>14}")
    print(f"{'='*55}")
    print(f"  Características guardadas en: {FEAT_DIR}/")
    print(f"  Gráficas en: {PLOT_DIR}/\n")


if __name__ == "__main__":
    main()

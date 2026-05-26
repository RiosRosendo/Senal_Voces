"""
Script 2 – Preprocesamiento de señales de voz
──────────────────────────────────────────────
Para cada grabación en voz/<palabra>/:
  1. Carga el audio
  2. Aplica filtro de preénfasis H(z) = 1 - 0.95·z^{-1}
  3. Detecta inicio y fin de la palabra (VAD usando energía + ZCR)
  4. Recorta la señal
  5. Guarda en voz_preprocesadas/<palabra>/

Al ejecutar verás:
  • Barra de progreso por palabra
  • Gráficas: señal original (con marcas de inicio/fin) y señal recortada
  • Resumen al final con el número de archivos procesados
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from utils import (FS, PALABRAS, FRAME_LEN, HOP,
                   apply_preemphasis, detect_endpoints)

VOZ_DIR    = "voz"
PRE_DIR    = "voz_preprocesadas"
PLOT_DIR   = os.path.join("resultados", "preprocesamiento")


def cargar_wav(path):
    """Carga WAV y normaliza a float32 en [-1, 1]."""
    fs, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483647.0
    if data.ndim > 1:
        data = data[:, 0]
    return fs, data


def guardar_wav(signal, path, fs=FS):
    int16 = np.clip(signal, -1.0, 1.0)
    int16 = (int16 * 32767).astype(np.int16)
    wav.write(path, fs, int16)


def procesar_archivo(ruta_in, ruta_out, plot_path=None):
    """
    Procesa un archivo: preénfasis → VAD → recorte.
    Opcionalmente guarda una gráfica de diagnóstico.
    """
    fs, signal = cargar_wav(ruta_in)

    # 1. Preénfasis
    signal_pre = apply_preemphasis(signal)

    # 2. Detección VAD
    start_s, end_s = detect_endpoints(signal_pre, FRAME_LEN, HOP)

    # 3. Recorte
    signal_trimmed = signal_pre[start_s:end_s]

    if len(signal_trimmed) < FRAME_LEN:
        # Señal muy corta: usar toda la señal con preénfasis
        signal_trimmed = signal_pre

    # 4. Guardar
    guardar_wav(signal_trimmed, ruta_out)

    # 5. Gráfica opcional
    if plot_path:
        t_orig    = np.arange(len(signal_pre)) / fs
        t_trim    = np.arange(len(signal_trimmed)) / fs

        fig, axes = plt.subplots(2, 1, figsize=(10, 5))

        axes[0].plot(t_orig, signal_pre, color='steelblue', linewidth=0.5)
        axes[0].axvline(start_s / fs, color='green',  linestyle='--',
                        label=f'Inicio ({start_s/fs:.3f} s)')
        axes[0].axvline(end_s / fs,   color='red',    linestyle='--',
                        label=f'Fin ({end_s/fs:.3f} s)')
        axes[0].set_title('Señal con preénfasis – detección VAD')
        axes[0].set_ylabel('Amplitud')
        axes[0].legend(fontsize=8)

        axes[1].plot(t_trim, signal_trimmed, color='darkorange', linewidth=0.5)
        axes[1].set_title('Señal recortada (solo la palabra)')
        axes[1].set_xlabel('Tiempo [s]')
        axes[1].set_ylabel('Amplitud')

        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()

    return len(signal_trimmed) / fs   # duración en segundos


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    total_ok  = 0
    total_err = 0

    for palabra in PALABRAS:
        in_dir  = os.path.join(VOZ_DIR,  palabra)
        out_dir = os.path.join(PRE_DIR,  palabra)
        plt_dir = os.path.join(PLOT_DIR, palabra)

        if not os.path.exists(in_dir):
            print(f"  [SKIP] {palabra}: directorio no encontrado ({in_dir})")
            continue

        archivos = sorted(f for f in os.listdir(in_dir) if f.endswith(".wav"))
        if not archivos:
            print(f"  [SKIP] {palabra}: sin archivos .wav")
            continue

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(plt_dir, exist_ok=True)

        print(f"\n  Procesando «{palabra}»  ({len(archivos)} archivos)...")
        duraciones = []

        for fname in archivos:
            ruta_in   = os.path.join(in_dir,  fname)
            ruta_out  = os.path.join(out_dir, fname)
            plot_path = os.path.join(plt_dir, fname.replace(".wav", ".png"))

            try:
                dur = procesar_archivo(ruta_in, ruta_out, plot_path)
                duraciones.append(dur)
                total_ok += 1
                print(f"    ✓ {fname}  →  {dur:.3f} s recortado")
            except Exception as exc:
                print(f"    ✗ {fname}: {exc}")
                total_err += 1

        if duraciones:
            print(f"  Media de duración recortada: {np.mean(duraciones):.3f} s "
                  f"(min {np.min(duraciones):.3f} / max {np.max(duraciones):.3f})")

    print(f"\n{'='*50}")
    print(f"  Archivos procesados OK : {total_ok}")
    print(f"  Archivos con error     : {total_err}")
    print(f"  Gráficas guardadas en  : {PLOT_DIR}/")
    print(f"  Señales en             : {PRE_DIR}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

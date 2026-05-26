"""
escuchar.py
===========
Graba una palabra desde el micrófono y la clasifica con los HMMs entrenados.
Muestra los scores de todos los modelos para que puedas debuggear.

Uso:
  python escuchar.py              # graba 2 s automáticamente
  python escuchar.py --dur 3      # graba 3 s
  python escuchar.py --ptt        # modo push-to-talk (Enter para grabar/parar)
  python escuchar.py --loop       # repite hasta Ctrl+C
  python escuchar.py --wav mi.wav # clasifica un WAV ya grabado (no graba)
"""

import os
import sys
import argparse
import numpy as np
import scipy.io.wavfile as wavfile

from utils import PALABRAS, FS, detect_endpoints
from hmm_utils import HMMBakis, extract_mfcc, quantize

HMM_DIR = os.path.join('models', 'hmm_models')
VQ_PATH = os.path.join('models', 'global_vq_K256.npz')
N_MFCC  = 13


# ─────────────────────────────────────────────────────────────────────────
# Carga de modelos (una sola vez)
# ─────────────────────────────────────────────────────────────────────────
def cargar_modelos():
    modelos = {}
    for palabra in PALABRAS:
        path = os.path.join(HMM_DIR, f'{palabra}_hmm.npz')
        if os.path.isfile(path):
            modelos[palabra] = HMMBakis.load(path)
        else:
            print(f"  ⚠ Modelo no encontrado: {path}")
    return modelos


def cargar_codebook():
    if not os.path.isfile(VQ_PATH):
        sys.exit(f"Error: codebook VQ no encontrado en {VQ_PATH}\n"
                 "Corre primero: python 6_extraer_mfcc.py")
    data = np.load(VQ_PATH)
    return data['centroids']


# ─────────────────────────────────────────────────────────────────────────
# Grabación
# ─────────────────────────────────────────────────────────────────────────
def grabar_fijo(dur_s: float) -> np.ndarray:
    """Graba `dur_s` segundos desde el micrófono."""
    import sounddevice as sd
    print(f"\n  🎙  Grabando {dur_s:.1f} s  — ¡habla ahora! ...", end='', flush=True)
    audio = sd.rec(int(dur_s * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    print("  [ok]")
    return audio.flatten()


def grabar_ptt() -> np.ndarray:
    """Push-to-talk: Enter para empezar, Enter para parar."""
    import sounddevice as sd
    import threading

    print("\n  🎙  Presiona ENTER para empezar a grabar ...")
    input()

    chunks = []
    stop_event = threading.Event()

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    stream = sd.InputStream(samplerate=FS, channels=1,
                            dtype='float32', callback=callback)
    stream.start()
    print("  ● Grabando ... presiona ENTER para parar.")
    input()
    stream.stop()
    stream.close()

    if not chunks:
        return np.zeros(1, dtype=np.float32)
    audio = np.concatenate(chunks).flatten()
    print(f"  [grabado {len(audio)/FS:.2f} s]")
    return audio


def cargar_wav(path: str) -> np.ndarray:
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    signal = data.astype(np.float32)
    if signal.max() > 1.5:          # int16 → float
        signal /= 32768.0
    if rate != FS:
        print(f"  ⚠ El WAV tiene {rate} Hz, el sistema espera {FS} Hz. "
              "Los resultados pueden ser incorrectos.")
    return signal


# ─────────────────────────────────────────────────────────────────────────
# Pipeline de reconocimiento
# ─────────────────────────────────────────────────────────────────────────
def reconocer(audio: np.ndarray, modelos: dict, centroids: np.ndarray,
              verbose: bool = True) -> str:
    """
    audio     : señal float32 normalizada [-1, 1]
    modelos   : {palabra: HMMBakis}
    centroids : (256, 13) codebook VQ
    Retorna la palabra predicha.
    """
    # 1. Detectar endpoints
    start, end = detect_endpoints(audio)
    segmento   = audio[start:end]

    dur_total = len(audio) / FS
    dur_seg   = len(segmento) / FS

    if verbose:
        print(f"\n  Audio total  : {dur_total:.2f} s  ({len(audio)} muestras)")
        print(f"  Segmento voz : {dur_seg:.2f} s  "
              f"[{start}–{end}]  ({len(segmento)} muestras)")

    if len(segmento) < int(0.1 * FS):
        print("  ⚠ Segmento demasiado corto — no se detectó voz clara.")
        print("     Intenta hablar más fuerte o más cerca del micrófono.")
        return None

    # 2. MFCC
    mfcc = extract_mfcc(segmento, fs=FS, n_mfcc=N_MFCC)
    if verbose:
        print(f"  Frames MFCC  : {len(mfcc)}")

    if len(mfcc) < 5:
        print("  ⚠ Muy pocos frames MFCC.")
        return None

    # 3. VQ
    indices = quantize(mfcc, centroids)
    if verbose:
        preview = indices[:15].tolist()
        print(f"  Secuencia VQ : {preview}{'...' if len(indices)>15 else ''} "
              f"  (len={len(indices)})")

    # 4. Forward en todos los modelos
    scores = {w: modelos[w].log_likelihood(indices) for w in modelos}
    pred   = max(scores, key=lambda w: scores[w])
    best   = scores[pred]

    # ── Tabla de scores ───────────────────────────────────────────
    if verbose:
        print()
        print(f"  {'Modelo':<12} {'Log-lik':>10}  {'Δ vs mejor':>12}  {'Bar'}")
        print("  " + "─" * 58)
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        max_bar = 30
        for w, s in sorted_scores:
            delta = s - best          # 0 para el ganador, negativo para el resto
            bar   = max(0, int(max_bar + delta * max_bar / max(abs(best), 1)))
            bar   = min(bar, max_bar)
            fill  = '█' * bar + '░' * (max_bar - bar)
            arrow = ' ◀ PREDICHO' if w == pred else ''
            print(f"  {w:<12} {s:>10.1f}  {delta:>+12.1f}  {fill}{arrow}")
        print()

    return pred


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Reconocimiento de voz en vivo')
    parser.add_argument('--dur',  type=float, default=2.0,
                        help='Duración de grabación en segundos (default: 2.0)')
    parser.add_argument('--ptt',  action='store_true',
                        help='Push-to-talk: Enter para grabar / parar')
    parser.add_argument('--loop', action='store_true',
                        help='Repite indefinidamente hasta Ctrl+C')
    parser.add_argument('--wav',  default=None,
                        help='Clasificar un archivo WAV existente (no graba)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Sistema de reconocimiento HMM+MFCC+VQ")
    print(f"  Palabras: {', '.join(PALABRAS)}")
    print("=" * 60)

    # Cargar modelos y codebook
    print("\nCargando modelos ...")
    modelos   = cargar_modelos()
    centroids = cargar_codebook()
    print(f"  {len(modelos)}/{len(PALABRAS)} modelos cargados")
    print(f"  Codebook VQ: {centroids.shape[0]} centroides × {centroids.shape[1]} dim")

    if not modelos:
        sys.exit("No hay modelos. Corre primero: python 7_entrenar_hmm.py")

    # Modo WAV
    if args.wav:
        print(f"\nClasificando archivo: {args.wav}")
        audio = cargar_wav(args.wav)
        pred  = reconocer(audio, modelos, centroids)
        if pred:
            print(f"\n  ══ RESULTADO: «{pred.upper()}» ══\n")
        return

    # Modo grabación
    n_iter = 0
    while True:
        n_iter += 1
        if args.loop:
            print(f"\n─── Intento #{n_iter} ───")

        if args.ptt:
            audio = grabar_ptt()
        else:
            audio = grabar_fijo(args.dur)

        pred = reconocer(audio, modelos, centroids)

        if pred:
            print(f"  ══ RESULTADO: «{pred.upper()}» ══")
        else:
            print("  ══ No se pudo clasificar ══")

        if not args.loop:
            break

        print("\n  Ctrl+C para salir.")
        try:
            pass
        except KeyboardInterrupt:
            print("\nSaliendo.")
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nSaliendo.")

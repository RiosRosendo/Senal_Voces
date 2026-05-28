"""
requantizar.py
==============
Re-cuantiza archivos WAV específicos usando el codebook VQ existente,
SIN reentrenar el codebook. Úsalo después de regrabar muestras de test
para no desestabilizar los modelos HMM.

Uso:
  python requantizar.py                          # re-cuantiza TODO (train+test)
  python requantizar.py --persona victor --word two --indices 11 12
  python requantizar.py --persona jordan --word two --indices 11 12
"""

import os
import argparse
import numpy as np
import scipy.io.wavfile as wavfile

from utils import PALABRAS, PERSONAS, detect_endpoints, FS, N_TRAIN, N_TEST
from hmm_utils import extract_mfcc, quantize

AUDIO_DIR = 'voz'
SEQ_DIR   = os.path.join('models', 'mfcc_sequences')
VQ_PATH   = os.path.join('models', 'global_vq_K256.npz')
N_MFCC    = 13


def cargar_codebook():
    if not os.path.isfile(VQ_PATH):
        raise FileNotFoundError(f"Codebook no encontrado: {VQ_PATH}")
    return np.load(VQ_PATH)['centroids']


def requantizar_archivo(persona, palabra, i, centroids):
    path = os.path.join(AUDIO_DIR, palabra, f'{persona}_{i:02d}.wav')
    if not os.path.isfile(path):
        print(f"  [SKIP] No existe: {path}")
        return False

    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    signal = data.astype(np.float32) / 32768.0

    start, end = detect_endpoints(signal)
    segment = signal[start:end]

    mfcc = extract_mfcc(segment, fs=FS, n_mfcc=N_MFCC)
    if len(mfcc) < 5:
        print(f"  [WARN] {persona}/{palabra}_{i:02d}: muy pocos frames ({len(mfcc)})")
        return False

    indices = quantize(mfcc, centroids)
    out = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}.npy')
    np.save(out, indices.astype(np.uint8))
    print(f"  OK  {persona}/{palabra}_{i:02d}  →  {len(indices)} frames")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona',  default=None, help='Filtrar por persona')
    parser.add_argument('--word',     default=None, help='Filtrar por palabra')
    parser.add_argument('--indices',  nargs='+', type=int, default=None,
                        help='Índices específicos (ej: 11 12 15). Default: todos')
    args = parser.parse_args()

    centroids = cargar_codebook()
    print(f"Codebook cargado: {centroids.shape[0]} centroides × {centroids.shape[1]} dim")
    print(f"(El codebook NO se reentrenará)\n")

    personas  = [args.persona] if args.persona else PERSONAS
    palabras  = [args.word]    if args.word    else PALABRAS
    indices   = args.indices   if args.indices else list(range(1, N_TRAIN + N_TEST + 1))

    n_ok = 0
    for persona in personas:
        for palabra in palabras:
            for i in indices:
                if requantizar_archivo(persona, palabra, i, centroids):
                    n_ok += 1

    print(f"\n{n_ok} secuencias actualizadas. Corre ahora:")
    print("  python 8_evaluar_hmm.py")


if __name__ == '__main__':
    main()

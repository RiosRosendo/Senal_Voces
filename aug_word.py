"""
aug_word.py
===========
Genera variantes de augmentation extra para una palabra específica
usando el codebook VQ existente (SIN reentrenarlo).
Luego reentrena solo el HMM de esa palabra.

Uso:
  python aug_word.py --word two
  python aug_word.py --word two --word left
"""

import os
import glob
import argparse
import numpy as np
import scipy.io.wavfile as wavfile

from utils import PALABRAS, PERSONAS, detect_endpoints, FS, N_TRAIN
from hmm_utils import extract_mfcc, quantize, HMMBakis

AUDIO_DIR = 'voz'
SEQ_DIR   = os.path.join('models', 'mfcc_sequences')
HMM_DIR   = os.path.join('models', 'hmm_models')
VQ_PATH   = os.path.join('models', 'global_vq_K256.npz')
N_MFCC    = 13
N_SYMBOLS = 256


# ── Funciones de augmentation ─────────────────────────────────────
def aug_speed(signal, factor):
    n_out = max(1, int(len(signal) * factor))
    idx = np.linspace(0, len(signal) - 1, n_out)
    return np.interp(idx, np.arange(len(signal)), signal).astype(np.float32)


def aug_noise(signal, snr_db):
    rng = np.random.default_rng(seed=snr_db)
    power = np.mean(signal ** 2) + 1e-10
    noise_power = power / (10 ** (snr_db / 10))
    return np.clip(signal + rng.normal(0, np.sqrt(noise_power), len(signal)), -1, 1).astype(np.float32)


def aug_volume(signal, gain):
    return np.clip(signal * gain, -1, 1).astype(np.float32)


# Variantes base (a1-a3) ya existen desde 6_extraer_mfcc.py
# Aquí generamos variantes extra (a4-a9)
EXTRA_VARIANTS = [
    ('a4', lambda s: aug_speed(s, 0.75)),               # muy rápido
    ('a5', lambda s: aug_speed(s, 1.25)),               # muy lento
    ('a6', lambda s: aug_noise(s, 10)),                 # ruido fuerte
    ('a7', lambda s: aug_noise(aug_speed(s, 0.90), 20)),# rápido + ruido
    ('a8', lambda s: aug_noise(aug_speed(s, 1.10), 20)),# lento + ruido
    ('a9', lambda s: aug_volume(s, 0.5)),               # volumen bajo
]


def cargar_codebook():
    if not os.path.isfile(VQ_PATH):
        raise FileNotFoundError(f"Codebook no encontrado: {VQ_PATH}")
    return np.load(VQ_PATH)['centroids']


def generar_aug_extra(palabra, centroids):
    """Genera variantes a4-a9 para todos los archivos de entrenamiento de una palabra."""
    n_gen = 0
    for persona in PERSONAS:
        for i in range(1, N_TRAIN + 1):
            wav_path = os.path.join(AUDIO_DIR, palabra, f'{persona}_{i:02d}.wav')
            if not os.path.isfile(wav_path):
                continue

            rate, data = wavfile.read(wav_path)
            if data.ndim > 1:
                data = data[:, 0]
            signal = data.astype(np.float32) / 32768.0
            start, end = detect_endpoints(signal)
            segment = signal[start:end]

            for tag, fn in EXTRA_VARIANTS:
                out = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}_{tag}.npy')
                if os.path.isfile(out):
                    continue  # ya existe, no regenerar
                aug_seg = fn(segment)
                mfcc = extract_mfcc(aug_seg, fs=FS, n_mfcc=N_MFCC)
                if len(mfcc) < 5:
                    continue
                indices = quantize(mfcc, centroids)
                np.save(out, indices.astype(np.uint8))
                n_gen += 1

    return n_gen


def entrenar_hmm(palabra, n_states=6):
    """Reentrena solo el HMM de una palabra cargando todas las secuencias aug."""
    sequences = []
    for persona in PERSONAS:
        for i in range(1, N_TRAIN + 1):
            # Original
            path = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}.npy')
            if os.path.isfile(path):
                seq = np.load(path).astype(np.int32)
                if len(seq) >= n_states:
                    sequences.append(seq)
            # Todas las augmentadas
            for aug_path in sorted(glob.glob(
                    os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}_a*.npy'))):
                seq = np.load(aug_path).astype(np.int32)
                if len(seq) >= n_states:
                    sequences.append(seq)

    hmm = HMMBakis(n_states=n_states, n_symbols=N_SYMBOLS)
    hmm.train(sequences)
    out = os.path.join(HMM_DIR, f'{palabra}_hmm.npz')
    hmm.save(out)
    return len(sequences), hmm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word',   nargs='+', required=True,
                        help='Palabra(s) a augmentar y reentrenar')
    parser.add_argument('--states', type=int, default=6)
    args = parser.parse_args()

    centroids = cargar_codebook()
    print(f"Codebook cargado: {centroids.shape} (SIN reentrenar)\n")

    for palabra in args.word:
        if palabra not in PALABRAS:
            print(f"  [ERROR] '{palabra}' no está en PALABRAS")
            continue

        print(f"── {palabra} ──")
        n = generar_aug_extra(palabra, centroids)
        print(f"  Variantes extra generadas : {n}")

        n_seqs, hmm = entrenar_hmm(palabra, args.states)
        diag = np.diag(hmm.A)
        print(f"  Secuencias de entrenamiento: {n_seqs}")
        print(f"  A diagonal: [{', '.join(f'{v:.2f}' for v in diag)}]")
        print(f"  HMM guardado: models/hmm_models/{palabra}_hmm.npz\n")

    print("Listo. Corre ahora:")
    print("  python 8_evaluar_hmm.py")


if __name__ == '__main__':
    main()

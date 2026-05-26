"""
7_entrenar_hmm.py
=================
Entrena un HMM Bakis (Left-to-Right) por cada palabra usando
ingeniería de conteos (segmentación lineal + acumulación de frecuencias).

Qué hace:
  1. Carga las secuencias de índices VQ de entrenamiento (01-10)
  2. Segmenta linealmente cada secuencia entre N estados
  3. Acumula conteos de emisión por estado → matriz B
  4. Estima matriz de transición A desde duraciones de segmento
  5. Aplica suavizado ε=1e-6 y renormaliza
  6. Guarda: models/hmm_models/{palabra}_hmm.npz

Resultado esperado al correr:
  · Tabla por palabra: secuencias usadas, diagonal de A, superdiagonal de A
  · La diagonal de A debe ser alta (>0.6): el robot permanece en cada estado
  · La superdiagonal de A refleja avance temporal

Uso:
  python3 7_entrenar_hmm.py [--states N]
  python3 7_entrenar_hmm.py --states 5
"""

import os
import argparse
import numpy as np

from utils import PALABRAS, PERSONAS, N_TRAIN
from hmm_utils import HMMBakis

SEQ_DIR   = os.path.join('models', 'mfcc_sequences')
HMM_DIR   = os.path.join('models', 'hmm_models')
N_SYMBOLS = 256

os.makedirs(HMM_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description='Entrenar HMM Bakis por ingeniería de conteos')
parser.add_argument('--states', type=int, default=6,
                    help='Número de estados ocultos por HMM (default: 6)')
args = parser.parse_args()
N_STATES = args.states

print("=" * 60)
print(f"PASO 7: Entrenamiento HMM Bakis  ({N_STATES} estados, {N_SYMBOLS} símbolos)")
print("=" * 60)

for palabra in PALABRAS:
    sequences = []
    for persona in PERSONAS:
        for i in range(1, N_TRAIN + 1):
            path = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}.npy')
            if os.path.isfile(path):
                seq = np.load(path).astype(np.int32)
                if len(seq) >= N_STATES:
                    sequences.append(seq)

    if not sequences:
        print(f"  [SKIP] {palabra}: no hay secuencias en {SEQ_DIR}/")
        continue

    hmm = HMMBakis(n_states=N_STATES, n_symbols=N_SYMBOLS)
    hmm.train(sequences)

    out_path = os.path.join(HMM_DIR, f'{palabra}_hmm.npz')
    hmm.save(out_path)

    # Mostrar estructura de A
    diag  = np.diag(hmm.A)
    offdi = np.diag(hmm.A, k=1)
    print(
        f"  {palabra:12s} | {len(sequences):2d} seqs "
        f"| A_diag=[{', '.join(f'{v:.2f}' for v in diag)}] "
        f"| A_off=[{', '.join(f'{v:.2f}' for v in offdi)}]"
    )

print(f"\n✓ Paso 7 completado. Modelos en {HMM_DIR}/")

# ── Visualizar estructura de B para la primera palabra ───────────
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    hmm0 = HMMBakis.load(os.path.join(HMM_DIR, f'{PALABRAS[0]}_hmm.npz'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Matriz A
    im = axes[0].imshow(hmm0.A, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title(f'Matriz A — {PALABRAS[0]} ({N_STATES} estados)')
    axes[0].set_xlabel('Estado siguiente')
    axes[0].set_ylabel('Estado actual')
    plt.colorbar(im, ax=axes[0])
    for r in range(N_STATES):
        for c in range(N_STATES):
            axes[0].text(c, r, f'{hmm0.A[r, c]:.2f}', ha='center', va='center',
                         fontsize=7, color='black' if hmm0.A[r, c] < 0.6 else 'white')

    # Sparsity de B (primeros 50 símbolos)
    axes[1].imshow(hmm0.B[:, :80], cmap='hot', aspect='auto')
    axes[1].set_title(f'Matriz B (primeros 80 símbolos) — {PALABRAS[0]}')
    axes[1].set_xlabel('Símbolo VQ')
    axes[1].set_ylabel('Estado')

    plt.tight_layout()
    plt.savefig('hmm_estructura_AB.png', dpi=150)
    plt.show()
    print("Figura guardada: hmm_estructura_AB.png")
except Exception as e:
    print(f"[Visualización omitida: {e}]")

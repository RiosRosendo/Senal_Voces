"""
8_evaluar_hmm.py
================
Evalúa el sistema HMM+MFCC+VQ con las 5 grabaciones de prueba por palabra.

Qué hace:
  1. Carga los 10 modelos HMM entrenados
  2. Para cada grabación de prueba (11-15):
       a. Carga la secuencia de índices VQ
       b. Calcula log P(O|λ) con el algoritmo Forward en log-espacio
       c. Asigna la palabra cuyo HMM da mayor log-verosimilitud
  3. Construye la matriz de confusión 10×10
  4. Muestra accuracy total y análisis por palabra

Resultado esperado:
  · Matriz de confusión con colores (diagonal = aciertos)
  · Accuracy global en porcentaje
  · Palabras más confundidas entre sí (similitud fonética)
  · Figura guardada: confusion_hmm.png

Uso:
  python3 8_evaluar_hmm.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils import PALABRAS, PERSONAS, N_TRAIN, N_TEST
from hmm_utils import HMMBakis

SEQ_DIR   = os.path.join('models', 'mfcc_sequences')
HMM_DIR   = os.path.join('models', 'hmm_models')

print("=" * 60)
print("PASO 8: Evaluación HMM+MFCC — matriz de confusión")
print("=" * 60)

# ── Cargar modelos ────────────────────────────────────────────────
hmm_models: dict[str, HMMBakis] = {}
for palabra in PALABRAS:
    path = os.path.join(HMM_DIR, f'{palabra}_hmm.npz')
    if os.path.isfile(path):
        hmm_models[palabra] = HMMBakis.load(path)
    else:
        print(f"  [WARN] Modelo no encontrado: {path}")

print(f"Modelos cargados: {len(hmm_models)}/{len(PALABRAS)}\n")

# ── Evaluación ────────────────────────────────────────────────────
n = len(PALABRAS)
word_idx = {w: i for i, w in enumerate(PALABRAS)}
confusion = np.zeros((n, n), dtype=int)
total, correct = 0, 0

for real_idx, palabra in enumerate(PALABRAS):
    for persona in PERSONAS:
        for i in range(N_TRAIN + 1, N_TRAIN + N_TEST + 1):
            path = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}.npy')
            if not os.path.isfile(path):
                continue

            seq = np.load(path).astype(np.int32)

            # Log-verosimilitud contra cada HMM
            scores = {w: hmm.log_likelihood(seq) for w, hmm in hmm_models.items()}
            predicted = max(scores, key=lambda w: scores[w])

            confusion[real_idx, word_idx[predicted]] += 1
            total  += 1
            correct += int(predicted == palabra)

accuracy = 100.0 * correct / max(total, 1)
print(f"Accuracy global: {correct}/{total} = {accuracy:.1f}%\n")

# ── Resultado por palabra ─────────────────────────────────────────
print(f"{'Palabra':12s}  {'Aciertos':>8}  {'Total':>5}  {'Acc %':>6}")
print("-" * 40)
for i, palabra in enumerate(PALABRAS):
    row   = confusion[i]
    hits  = confusion[i, i]
    tot   = row.sum()
    acc_w = 100.0 * hits / max(tot, 1)
    print(f"  {palabra:12s}  {hits:>4}/{tot:<3}          {acc_w:5.1f}%")

# ── Matriz de confusión ───────────────────────────────────────────
try:
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        confusion,
        annot=True, fmt='d', cmap='Blues',
        xticklabels=PALABRAS, yticklabels=PALABRAS,
        linewidths=0.5, linecolor='gray',
        ax=ax,
    )
except ImportError:
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(confusion, cmap='Blues')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(PALABRAS, rotation=45, ha='right')
    ax.set_yticklabels(PALABRAS)
    for r in range(n):
        for c in range(n):
            ax.text(c, r, str(confusion[r, c]),
                    ha='center', va='center', fontsize=9)

ax.set_xlabel('Predicho', fontsize=12)
ax.set_ylabel('Real',     fontsize=12)
ax.set_title(f'HMM + MFCC + VQ(256)  —  Accuracy: {accuracy:.1f}%', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_hmm.png', dpi=150)
plt.show()
print("\nFigura guardada: confusion_hmm.png")
print("\n✓ Paso 8 completado.")

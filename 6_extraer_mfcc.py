"""
6_extraer_mfcc.py
=================
Extrae MFCC, entrena el codebook VQ global (K=256) y visualiza resultados.

QUÉ VAS A VER AL CORRER:
  1. Cuántas grabaciones se procesaron y cuántos frames MFCC se extrajeron
  2. Figura 1 – Espectrograma MFCC de una grabación de ejemplo
       · Eje X = tiempo (frames), Eje Y = coeficiente MFCC (0-12)
       · Colores = valor del coeficiente (azul=bajo, rojo=alto)
       · Debes ver patrones distintos para cada palabra
  3. Figura 2 – Comparación de MFCCs promedio por palabra
       · Cada línea = perfil promedio de los 13 coeficientes para una palabra
       · Palabras distintas deben tener perfiles distintos
  4. Figura 3 – Codebook VQ global (256 centroides)
       · Mapa de calor de los 256 centroides × 13 coeficientes
       · Muestra que el codebook captura variedad de patrones
  5. Figura 4 – Distribución de uso del codebook
       · Histograma de cuántas veces se usó cada uno de los 256 símbolos
       · Distribución pareja = buen codebook

Uso:
  python3 6_extraer_mfcc.py
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

from utils import PALABRAS, PERSONAS, detect_endpoints, FS, N_TRAIN, N_TEST
from hmm_utils import extract_mfcc, quantize

# ── Configuración ─────────────────────────────────────────────────
AUDIO_DIR  = 'voz'
MODELS_DIR = 'models'
SEQ_DIR    = os.path.join(MODELS_DIR, 'mfcc_sequences')
N_MFCC     = 13
VQ_K       = 256

os.makedirs(SEQ_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("PASO 6: Extracción MFCC + VQ global (K=256)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 1. Extraer MFCC de todas las grabaciones
# ─────────────────────────────────────────────────────────────────
all_train_mfcc = []
seq_cache: dict = {}          # (persona, palabra, i) → array MFCC
mfcc_por_palabra: dict = {}   # palabra → lista de arrays MFCC (para visualizar)

for persona in PERSONAS:
    for palabra in PALABRAS:
        for i in range(1, N_TRAIN + N_TEST + 1):
            path = os.path.join(AUDIO_DIR, palabra, f'{persona}_{i:02d}.wav')
            if not os.path.isfile(path):
                continue

            rate, data = wavfile.read(path)
            if data.ndim > 1:
                data = data[:, 0]
            signal = data.astype(np.float32) / 32768.0

            start, end = detect_endpoints(signal)
            segment = signal[start:end]

            mfcc = extract_mfcc(segment, fs=FS, n_mfcc=N_MFCC)
            if len(mfcc) < 5:
                continue

            seq_cache[(persona, palabra, i)] = mfcc

            if i <= N_TRAIN:
                all_train_mfcc.append(mfcc)
                mfcc_por_palabra.setdefault(palabra, []).append(mfcc)

n_total  = len(PERSONAS) * len(PALABRAS) * (N_TRAIN + N_TEST)
n_frames = sum(len(m) for m in all_train_mfcc)
print(f"Grabaciones procesadas : {len(seq_cache)} / {n_total}")
print(f"Frames de entrenamiento: {n_frames}")

# ─────────────────────────────────────────────────────────────────
# 2. Entrenar codebook VQ global K=256
# ─────────────────────────────────────────────────────────────────
print(f"\nEntrenando codebook VQ global (K={VQ_K}) ...")
X = np.vstack(all_train_mfcc)
print(f"  Vectores: {X.shape[0]} × {X.shape[1]}")

kmeans = MiniBatchKMeans(
    n_clusters=VQ_K, n_init=5, max_iter=300,
    random_state=42, batch_size=4096,
)
kmeans.fit(X)
centroids = kmeans.cluster_centers_.astype(np.float32)

vq_path = os.path.join(MODELS_DIR, f'global_vq_K{VQ_K}.npz')
np.savez(vq_path, centroids=centroids)
print(f"  Codebook guardado: {vq_path}")

# ─────────────────────────────────────────────────────────────────
# 3. Cuantizar todas las secuencias
# ─────────────────────────────────────────────────────────────────
print("\nCuantizando secuencias ...")
symbol_counts = np.zeros(VQ_K, dtype=int)

for (persona, palabra, i), mfcc in seq_cache.items():
    indices = quantize(mfcc, centroids)
    out_path = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}.npy')
    np.save(out_path, indices.astype(np.uint8))
    symbol_counts += np.bincount(indices, minlength=VQ_K)

print(f"  Secuencias guardadas en: {SEQ_DIR}/")

# Mostrar ejemplo de secuencia
primera_key = next(iter(seq_cache))
ex_indices = quantize(seq_cache[primera_key], centroids)
persona_ej, palabra_ej, _ = primera_key
print(f"\nEjemplo ({persona_ej} / {palabra_ej}):")
print(f"  Longitud: {len(ex_indices)} frames")
print(f"  Secuencia: {ex_indices[:20].tolist()} ...")

# ─────────────────────────────────────────────────────────────────
# 4. Visualizaciones
# ─────────────────────────────────────────────────────────────────
os.makedirs('resultados', exist_ok=True)

# ── Figura 1: Espectrograma MFCC de una grabación ────────────────
fig1, ax = plt.subplots(figsize=(12, 4))
mfcc_ej = seq_cache[primera_key]
im = ax.imshow(mfcc_ej.T, aspect='auto', origin='lower', cmap='RdBu_r')
ax.set_xlabel('Frame (tiempo →)', fontsize=11)
ax.set_ylabel('Coeficiente MFCC', fontsize=11)
ax.set_title(f'MFCCs — {persona_ej} / "{palabra_ej}"', fontsize=13)
ax.set_yticks(range(N_MFCC))
ax.set_yticklabels([f'MFCC {i}' for i in range(N_MFCC)], fontsize=7)
plt.colorbar(im, ax=ax, label='Valor')
plt.tight_layout()
plt.savefig('resultados/mfcc_espectrograma.png', dpi=150)
plt.show()
print("\nFigura 1 guardada: resultados/mfcc_espectrograma.png")

# ── Figura 2: Perfil MFCC promedio por palabra ───────────────────
fig2, ax = plt.subplots(figsize=(12, 5))
colores = plt.cm.tab10(np.linspace(0, 1, len(PALABRAS)))
for (palabra, lista), color in zip(mfcc_por_palabra.items(), colores):
    promedio = np.vstack(lista).mean(axis=0)
    ax.plot(promedio, label=palabra, color=color, linewidth=2, marker='o', markersize=4)
ax.set_xlabel('Coeficiente MFCC (0-12)', fontsize=11)
ax.set_ylabel('Valor promedio', fontsize=11)
ax.set_title('Perfil MFCC promedio por palabra\n(líneas distintas = palabras distintas = sistema puede diferenciarlas)', fontsize=12)
ax.set_xticks(range(N_MFCC))
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('resultados/mfcc_perfil_por_palabra.png', dpi=150)
plt.show()
print("Figura 2 guardada: resultados/mfcc_perfil_por_palabra.png")

# ── Figura 3: Codebook VQ (centroides) ───────────────────────────
fig3, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(centroids, aspect='auto', cmap='viridis')
ax.set_xlabel('Coeficiente MFCC (dimensión)', fontsize=11)
ax.set_ylabel('Centroide del codebook (0-255)', fontsize=11)
ax.set_title(f'Codebook VQ global — {VQ_K} centroides × {N_MFCC} dimensiones', fontsize=13)
plt.colorbar(im, ax=ax, label='Valor del centroide')
plt.tight_layout()
plt.savefig('resultados/codebook_vq.png', dpi=150)
plt.show()
print("Figura 3 guardada: resultados/codebook_vq.png")

# ── Figura 4: Distribución de uso del codebook ───────────────────
fig4, ax = plt.subplots(figsize=(14, 4))
ax.bar(range(VQ_K), symbol_counts, color='steelblue', alpha=0.8, width=1.0)
ax.set_xlabel('Símbolo VQ (0-255)', fontsize=11)
ax.set_ylabel('Frecuencia de uso', fontsize=11)
ax.set_title('Distribución de uso del codebook\n(distribución pareja = buen codebook)', fontsize=12)
ax.axhline(symbol_counts.mean(), color='red', linestyle='--',
           linewidth=1.5, label=f'Promedio = {symbol_counts.mean():.0f}')
ax.legend()
plt.tight_layout()
plt.savefig('resultados/codebook_distribucion.png', dpi=150)
plt.show()
print("Figura 4 guardada: resultados/codebook_distribucion.png")

print("\n" + "=" * 60)
print("✓ Paso 6 completado.")
print(f"  Grabaciones procesadas : {len(seq_cache)}")
print(f"  Frames totales         : {n_frames}")
print(f"  Codebook               : {VQ_K} símbolos × {N_MFCC} dim")
print("  Siguiente: python3 7_entrenar_hmm.py")
print("=" * 60)

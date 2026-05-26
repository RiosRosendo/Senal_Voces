"""
Script 4 – Entrenamiento de codebooks con Cuantización Vectorial
─────────────────────────────────────────────────────────────────
Para cada palabra:
  1. Carga los LSF de las primeras 10 grabaciones (entrenamiento)
  2. Aplica K-means con K ∈ {16, 32, 64}
  3. Guarda el codebook en codebooks/<palabra>_K<k>.npz

Al ejecutar verás:
  • Progreso del K-means (inercia final por codebook)
  • Gráfica de centroides LSF para cada tamaño de codebook
  • Tabla resumen con inercia y tiempo de entrenamiento por palabra/K
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from utils import PALABRAS, LPC_ORDER, CODEBOOK_SIZES, N_TRAIN

FEAT_DIR     = "features"
CB_DIR       = "codebooks"
PLOT_DIR     = os.path.join("resultados", "codebooks")


def es_entrenamiento(fname):
    """True si el archivo corresponde a grabaciones 01-10 (entrenamiento)."""
    import re
    m = re.search(r'(\d+)\.npz$', fname)
    return bool(m) and int(m.group(1)) <= 10


def cargar_features_entrenamiento(palabra):
    """Carga los LSF de las grabaciones 01-10 de todos los integrantes."""
    feat_dir = os.path.join(FEAT_DIR, palabra)
    archivos = sorted(f for f in os.listdir(feat_dir)
                      if f.endswith(".npz") and es_entrenamiento(f))

    lsf_total = []
    for fname in archivos:
        data = np.load(os.path.join(feat_dir, fname))
        lsf_total.append(data['lsf'])

    if not lsf_total:
        raise ValueError(f"Sin datos de entrenamiento para «{palabra}»")

    return np.vstack(lsf_total)   # (N_total_frames, LPC_ORDER)


def graficar_centroides(centroides_dict, palabra, plot_path):
    """Grafica los centroides LSF para cada tamaño de codebook."""
    fig, axes = plt.subplots(1, len(centroides_dict), figsize=(14, 4))
    if len(centroides_dict) == 1:
        axes = [axes]

    for ax, (k, centroids) in zip(axes, centroides_dict.items()):
        for c in centroids:
            ax.plot(range(1, LPC_ORDER + 1), c / np.pi,
                    color='steelblue', alpha=0.4, linewidth=0.8)
        media = centroids.mean(axis=0)
        ax.plot(range(1, LPC_ORDER + 1), media / np.pi,
                color='red', linewidth=2, label='Media')
        ax.set_title(f'«{palabra}» – K={k}')
        ax.set_xlabel('Coeficiente LSF')
        ax.set_ylabel('LSF / π')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f'Centroides del codebook – «{palabra}»', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()


def main():
    os.makedirs(CB_DIR,   exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO DE CODEBOOKS  (K = {CODEBOOK_SIZES})")
    print(f"  Grabaciones de entrenamiento por palabra: {N_TRAIN}")
    print(f"{'='*60}\n")

    resumen = []

    for palabra in PALABRAS:
        feat_dir = os.path.join(FEAT_DIR, palabra)
        if not os.path.exists(feat_dir):
            print(f"  [SKIP] {palabra}: sin features")
            continue

        archivos = [f for f in os.listdir(feat_dir) if f.endswith(".npz")]
        if len(archivos) < 1:
            print(f"  [SKIP] {palabra}: sin archivos .npz")
            continue

        print(f"  Palabra: «{palabra}»")
        try:
            lsf_data = cargar_features_entrenamiento(palabra)
            print(f"    Frames de entrenamiento: {len(lsf_data)}")
        except Exception as exc:
            print(f"    ✗ Error cargando features: {exc}")
            continue

        centroides_dict = {}

        for K in CODEBOOK_SIZES:
            t0 = time.time()

            # K no puede ser mayor que el número de muestras disponibles
            K_real = min(K, len(lsf_data))
            if K_real < K:
                print(f"    ⚠ K={K} reducido a K={K_real} (pocos frames)")

            # K-means sobre los vectores LSF
            kmeans = KMeans(n_clusters=K_real,
                            n_init=10,
                            max_iter=300,
                            random_state=42)
            kmeans.fit(lsf_data)
            centroids = kmeans.cluster_centers_   # (K, LPC_ORDER)
            inercia   = kmeans.inertia_
            elapsed   = time.time() - t0

            # Guardar codebook
            cb_path = os.path.join(CB_DIR, f"{palabra}_K{K}.npz")
            np.savez_compressed(cb_path,
                                centroids=centroids,
                                K=np.array(K),
                                palabra=np.array(palabra))

            centroides_dict[K] = centroids
            resumen.append((palabra, K, len(lsf_data), inercia, elapsed))
            print(f"    K={K:>2}  inercia={inercia:10.2f}  t={elapsed:.2f}s  → {cb_path}")

        # Gráfica de centroides para esta palabra
        plot_path = os.path.join(PLOT_DIR, f"{palabra}_centroides.png")
        graficar_centroides(centroides_dict, palabra, plot_path)

    # ── Tabla resumen ──
    print(f"\n{'='*65}")
    print(f"  {'Palabra':<12} {'K':>4} {'Frames':>8} {'Inercia':>14} {'Tiempo':>8}")
    print(f"  {'-'*62}")
    for pal, k, nf, iner, t in resumen:
        print(f"  {pal:<12} {k:>4} {nf:>8} {iner:>14.2f} {t:>7.2f}s")
    print(f"{'='*65}")
    print(f"  Codebooks guardados en: {CB_DIR}/")
    print(f"  Gráficas en: {PLOT_DIR}/\n")


if __name__ == "__main__":
    main()

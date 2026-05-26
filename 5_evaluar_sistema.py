"""
Script 5 – Evaluación del sistema (Matriz de confusión)
────────────────────────────────────────────────────────
Para cada tamaño de codebook K ∈ {16, 32, 64}:
  1. Carga las 5 grabaciones de prueba por palabra
  2. Extrae sus características LSF/LPC
  3. Clasifica usando distancia de Itakura-Saito contra cada codebook
  4. Construye la matriz de confusión
  5. Calcula accuracy por palabra y global

Al ejecutar verás:
  • Progreso de clasificación en tiempo real
  • Matrices de confusión graficadas (una por K)
  • Tabla de accuracy por palabra para cada K
  • Comparativa final: ¿qué tamaño de codebook es mejor?
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.io.wavfile as wav

from utils import (PALABRAS, LPC_ORDER, CODEBOOK_SIZES, N_TRAIN,
                   extract_features, vq_distortion, lsf_to_lpc)

FEAT_DIR  = "features"
CB_DIR    = "codebooks"
PRE_DIR   = "voz_preprocesadas"
PLOT_DIR  = os.path.join("resultados", "evaluacion")


# ──────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────
def cargar_features_prueba(palabra):
    """Retorna lista de dicts de features para las grabaciones de prueba (11-15)."""
    import re
    feat_dir = os.path.join(FEAT_DIR, palabra)
    archivos = sorted(
        f for f in os.listdir(feat_dir)
        if f.endswith(".npz") and
        (lambda m: bool(m) and int(m.group(1)) > 10)(re.search(r'(\d+)\.npz$', f))
    )

    feats = []
    for fname in archivos:
        data = np.load(os.path.join(feat_dir, fname))
        feats.append({
            'lsf':  data['lsf'],
            'lpc':  data['lpc'],
            'gain': data['gain'],
            'acf':  data['acf'],
        })
    return feats


def cargar_codebook(palabra, K):
    """Carga el codebook de una palabra y tamaño K."""
    path = os.path.join(CB_DIR, f"{palabra}_K{K}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data['centroids']   # (K, LPC_ORDER)


# ──────────────────────────────────────────────
# Clasificación
# ──────────────────────────────────────────────
def clasificar(features, codebooks_por_palabra, K, use_is=True):
    """
    Clasifica un vector de features contra todos los codebooks.
    Retorna la palabra con menor distorsión VQ.
    """
    distorsiones = {}
    for palabra, codebook in codebooks_por_palabra.items():
        if codebook is None:
            distorsiones[palabra] = float('inf')
        else:
            distorsiones[palabra] = vq_distortion(features, codebook,
                                                   use_is=use_is)
    return min(distorsiones, key=distorsiones.get), distorsiones


# ──────────────────────────────────────────────
# Matriz de confusión
# ──────────────────────────────────────────────
def graficar_matriz(matriz, palabras, K, acc_global, plot_path):
    """Grafica la matriz de confusión normalizada por fila."""
    n = len(palabras)
    norm = matriz.astype(float)
    row_sums = norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_mat = norm / row_sums * 100.0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_mat, vmin=0, vmax=100,
                   cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='% por fila')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(palabras, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(palabras, fontsize=9)
    ax.set_xlabel('Predicho', fontsize=11)
    ax.set_ylabel('Real', fontsize=11)
    ax.set_title(f'Matriz de Confusión – K={K}   (Accuracy global: {acc_global:.1f}%)',
                 fontsize=12)

    for i in range(n):
        for j in range(n):
            val   = int(matriz[i, j])
            pct   = norm_mat[i, j]
            color = 'white' if pct > 50 else 'black'
            ax.text(j, i, f'{val}\n({pct:.0f}%)',
                    ha='center', va='center', fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"    Gráfica guardada: {plot_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Verificar que existan features de prueba
    palabras_disponibles = []
    import re
    for p in PALABRAS:
        feat_dir = os.path.join(FEAT_DIR, p)
        if os.path.exists(feat_dir):
            archivos_test = [
                f for f in os.listdir(feat_dir)
                if f.endswith(".npz") and
                (lambda m: bool(m) and int(m.group(1)) > 10)(re.search(r'(\d+)\.npz$', f))
            ]
            if archivos_test:
                palabras_disponibles.append(p)

    if not palabras_disponibles:
        print("\n  ERROR: No se encontraron features de prueba.")
        print(f"  Ejecuta primero los scripts 1, 2 y 3, y asegúrate de tener")
        print(f"  más de {N_TRAIN} grabaciones por palabra.\n")
        return

    n_palabras = len(palabras_disponibles)
    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN DEL SISTEMA DE RECONOCIMIENTO")
    print(f"  Palabras evaluadas  : {n_palabras}")
    print(f"  Tamaños de codebook : {CODEBOOK_SIZES}")
    print(f"{'='*60}\n")

    resultados_globales = {}   # K → accuracy global

    for K in CODEBOOK_SIZES:
        print(f"\n  ── Codebook K={K} ──")

        # Cargar todos los codebooks para este K
        codebooks = {}
        ok = True
        for p in palabras_disponibles:
            cb = cargar_codebook(p, K)
            if cb is None:
                print(f"    [SKIP] Codebook no encontrado: {p}_K{K}")
                ok = False
            codebooks[p] = cb

        if not ok:
            print(f"    Faltan codebooks para K={K}. "
                  f"Ejecuta primero 4_entrenar_codebook.py")
            continue

        # Matriz de confusión (filas = real, cols = predicho)
        idx = {p: i for i, p in enumerate(palabras_disponibles)}
        matriz = np.zeros((n_palabras, n_palabras), dtype=int)

        # Clasificar grabaciones de prueba
        for p_real in palabras_disponibles:
            feats_list = cargar_features_prueba(p_real)
            if not feats_list:
                print(f"    Sin grabaciones de prueba para «{p_real}»")
                continue

            for feats in feats_list:
                p_pred, distorsiones = clasificar(feats, codebooks, K)
                matriz[idx[p_real], idx[p_pred]] += 1

        # Métricas
        correctos = np.trace(matriz)
        total     = matriz.sum()
        acc_global = 100.0 * correctos / max(total, 1)

        print(f"\n  Matriz de confusión (K={K}):")
        header = f"{'':>12} " + "  ".join(f"{p[:6]:>6}" for p in palabras_disponibles)
        print(f"    {header}")
        for i, p in enumerate(palabras_disponibles):
            fila = "  ".join(f"{v:>6}" for v in matriz[i])
            print(f"    {p[:12]:>12} {fila}")

        print(f"\n  Accuracy por palabra:")
        for i, p in enumerate(palabras_disponibles):
            n_real  = matriz[i].sum()
            n_ok    = matriz[i, i]
            acc_p   = 100.0 * n_ok / max(n_real, 1)
            barra   = '█' * int(acc_p / 5)
            print(f"    {p:<12}  {n_ok}/{n_real:>2}  {acc_p:>6.1f}%  {barra}")

        print(f"\n  Accuracy GLOBAL K={K}: {correctos}/{total} = {acc_global:.1f}%")
        resultados_globales[K] = acc_global

        # Gráfica
        plot_path = os.path.join(PLOT_DIR, f"confusion_K{K}.png")
        graficar_matriz(matriz, palabras_disponibles, K, acc_global, plot_path)

    # ── Comparativa de tamaños de codebook ──
    if resultados_globales:
        print(f"\n{'='*50}")
        print(f"  COMPARATIVA DE TAMAÑOS DE CODEBOOK")
        print(f"{'='*50}")
        mejor_K   = max(resultados_globales, key=resultados_globales.get)
        for K, acc in sorted(resultados_globales.items()):
            marca = " ← MEJOR" if K == mejor_K else ""
            print(f"    K={K:>2}  Accuracy = {acc:.1f}%{marca}")

        print(f"\n  Conclusión: El codebook de tamaño K={mejor_K} ofrece")
        print(f"  el mejor desempeño con {resultados_globales[mejor_K]:.1f}% de accuracy.")
        print(f"\n  Análisis:")
        ks = sorted(resultados_globales)
        if len(ks) >= 2:
            tendencia = "aumenta" if resultados_globales[ks[-1]] >= resultados_globales[ks[0]] \
                        else "disminuye"
            print(f"    • La accuracy {tendencia} al aumentar K.")
            print(f"    • Codebook pequeño (K={ks[0]}): menos memoria, posible "
                  f"subrepresentación.")
            print(f"    • Codebook grande (K={ks[-1]}): mayor capacidad, "
                  f"posible sobreajuste con pocos datos.")

        # Graficar comparativa
        fig, ax = plt.subplots(figsize=(7, 4))
        ks_list  = sorted(resultados_globales)
        acc_list = [resultados_globales[k] for k in ks_list]
        ax.bar([str(k) for k in ks_list], acc_list,
               color=['steelblue', 'darkorange', 'seagreen'])
        ax.set_xlabel('Tamaño de codebook (K)')
        ax.set_ylabel('Accuracy global (%)')
        ax.set_title('Comparativa de tamaños de codebook')
        ax.set_ylim(0, 105)
        for i, (k, a) in enumerate(zip(ks_list, acc_list)):
            ax.text(i, a + 1, f'{a:.1f}%', ha='center', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        comp_path = os.path.join(PLOT_DIR, "comparativa_K.png")
        plt.savefig(comp_path, dpi=100)
        plt.close()
        print(f"\n  Gráfica comparativa: {comp_path}")

    print(f"\n  Todas las gráficas en: {PLOT_DIR}/\n")


if __name__ == "__main__":
    main()

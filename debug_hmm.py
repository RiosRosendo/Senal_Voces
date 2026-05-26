"""
debug_hmm.py
============
Herramienta de diagnóstico del sistema HMM+MFCC+VQ.

Muestra:
  1. Resumen de datos: cuántas secuencias hay por persona/palabra y qué
     va a entrenamiento vs prueba.
  2. Longitud de secuencias (frames por grabación) — secuencias muy
     cortas hacen que la segmentación lineal sea inestable.
  3. Inspección de los modelos entrenados: diagonal de A, entropía de B,
     y cuántos símbolos concentran el 90 % de la probabilidad por estado.
  4. Tabla de log-likelihoods: para CADA grabación de prueba muestra la
     puntuación que le da CADA modelo, no solo el ganador.
     Así puedes ver si los modelos están muy juntos o muy separados.
  5. Resumen de errores: qué palabras confunde, con quién, en qué persona.

Uso:
  python debug_hmm.py
  python debug_hmm.py --word start       # solo la palabra "start"
  python debug_hmm.py --persona hugo     # solo el hablante "hugo"
  python debug_hmm.py --no-likelihoods   # omite la tabla detallada
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils import PALABRAS, PERSONAS, N_TRAIN, N_TEST
from hmm_utils import HMMBakis
SEQ_DIR   = os.path.join('models', 'mfcc_sequences')
HMM_DIR   = os.path.join('models', 'hmm_models')

# ─────────────────────────────────────────────────────────────────────────
def sep(char='─', width=72):
    print(char * width)

def header(title):
    sep('═')
    print(f"  {title}")
    sep('═')

# ─────────────────────────────────────────────────────────────────────────
# 1. RESUMEN DE DATOS
# ─────────────────────────────────────────────────────────────────────────
def resumen_datos(filter_word=None, filter_persona=None):
    header("RESUMEN DE DATOS: train vs test")

    palabras_mostrar = [p for p in PALABRAS if filter_word is None or p == filter_word]
    personas_mostrar = [p for p in PERSONAS  if filter_persona is None or p == filter_persona]

    total_train = 0
    total_test  = 0
    faltantes   = []

    # Encabezado de tabla
    col = 14
    print(f"{'Palabra':<12} {'Persona':<10} {'Train':>7} {'Test':>6} {'Missing':>8}")
    sep()

    for palabra in palabras_mostrar:
        for persona in personas_mostrar:
            train_ok = 0
            test_ok  = 0
            miss     = []
            for i in range(1, N_TRAIN + N_TEST + 1):
                path = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}.npy')
                if os.path.isfile(path):
                    if i <= N_TRAIN:
                        train_ok += 1
                    else:
                        test_ok  += 1
                else:
                    miss.append(i)

            total_train += train_ok
            total_test  += test_ok
            miss_str = ','.join(map(str, miss)) if miss else '—'
            if miss:
                faltantes.append((persona, palabra, miss))
            print(f"  {palabra:<10} {persona:<10} {train_ok:>5}/{N_TRAIN:<2} {test_ok:>4}/{N_TEST:<2} {miss_str:>8}")
        print()  # separar por palabra

    sep()
    print(f"  TOTAL entrenamiento: {total_train} secuencias")
    print(f"  TOTAL prueba:        {total_test}  secuencias")
    print(f"  TOTAL archivos:      {total_train + total_test}")
    if faltantes:
        print(f"\n  ⚠ Archivos faltantes detectados:")
        for persona, palabra, miss in faltantes:
            print(f"    {persona}/{palabra}: índices {miss}")


# ─────────────────────────────────────────────────────────────────────────
# 2. LONGITUD DE SECUENCIAS
# ─────────────────────────────────────────────────────────────────────────
def resumen_longitudes(filter_word=None, filter_persona=None):
    header("LONGITUD DE SECUENCIAS (frames)")

    palabras_mostrar = [p for p in PALABRAS if filter_word is None or p == filter_word]
    personas_mostrar = [p for p in PERSONAS  if filter_persona is None or p == filter_persona]

    print(f"{'Palabra':<12} {'Conjunto':<8} {'N':>4} {'Min':>5} {'Max':>5} {'Media':>7} {'<20':>5}")
    sep()

    for palabra in palabras_mostrar:
        for label, rng in [('train', range(1, N_TRAIN+1)), ('test', range(N_TRAIN+1, N_TRAIN+N_TEST+1))]:
            lens = []
            for persona in personas_mostrar:
                for i in rng:
                    path = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}.npy')
                    if os.path.isfile(path):
                        seq = np.load(path)
                        lens.append(len(seq))
            if not lens:
                continue
            short = sum(1 for l in lens if l < 20)
            print(f"  {palabra:<10} {label:<8} {len(lens):>4} {min(lens):>5} {max(lens):>5} "
                  f"{np.mean(lens):>7.1f} {short:>5}")
    sep()
    print("  Secuencias con < 20 frames son problemáticas (segmentación pobre).")


# ─────────────────────────────────────────────────────────────────────────
# 3. INSPECCIÓN DE MODELOS HMM
# ─────────────────────────────────────────────────────────────────────────
def inspeccionar_modelos(filter_word=None):
    header("INSPECCIÓN DE MODELOS HMM")

    palabras_mostrar = [p for p in PALABRAS if filter_word is None or p == filter_word]

    for palabra in palabras_mostrar:
        path = os.path.join(HMM_DIR, f'{palabra}_hmm.npz')
        if not os.path.isfile(path):
            print(f"  {palabra}: ⚠ modelo no encontrado")
            continue

        hmm = HMMBakis.load(path)
        N   = hmm.N

        diag  = np.diag(hmm.A)
        offdi = np.diag(hmm.A, 1)

        # Sparsity de B: para cada estado, ¿cuántos símbolos acumulan el 90%?
        sparse_90 = []
        for s in range(N):
            row = hmm.B[s]
            idx = np.argsort(row)[::-1]
            cum = np.cumsum(row[idx])
            k90 = int(np.searchsorted(cum, 0.90)) + 1
            sparse_90.append(k90)

        # Entropía de B por estado
        entropias = []
        for s in range(N):
            row = np.maximum(hmm.B[s], 1e-12)
            h   = -np.sum(row * np.log2(row))
            entropias.append(h)

        print(f"\n  ── {palabra} ({N} estados, {hmm.M} símbolos) ──")
        print(f"     A diagonal:    [{', '.join(f'{v:.3f}' for v in diag)}]")
        print(f"     A superdiag:   [{', '.join(f'{v:.3f}' for v in offdi)}]")
        print(f"     B símbolos@90%:[{', '.join(f'{v:3d}' for v in sparse_90)}]  "
              f"(de 256 — menos es más discriminativo)")
        print(f"     B entropía:    [{', '.join(f'{v:.2f}' for v in entropias)}]  "
              f"(bits — más bajo = más enfocado)")

        # Advertencias
        if any(d < 0.5 for d in diag):
            print(f"     ⚠ Diagonal baja en A — algunos estados se saltan rápido")
        if any(k > 150 for k in sparse_90):
            print(f"     ⚠ B muy dispersa — el modelo no discrimina bien")


# ─────────────────────────────────────────────────────────────────────────
# 4. TABLA DETALLADA DE LOG-LIKELIHOODS
# ─────────────────────────────────────────────────────────────────────────
def tabla_likelihoods(filter_word=None, filter_persona=None):
    header("LOG-LIKELIHOODS POR GRABACIÓN DE PRUEBA")

    # Cargar todos los modelos
    hmm_models = {}
    for palabra in PALABRAS:
        path = os.path.join(HMM_DIR, f'{palabra}_hmm.npz')
        if os.path.isfile(path):
            hmm_models[palabra] = HMMBakis.load(path)

    if not hmm_models:
        print("  ⚠ No se encontraron modelos.")
        return

    palabras_mostrar  = [p for p in PALABRAS if filter_word is None or p == filter_word]
    personas_mostrar  = [p for p in PERSONAS  if filter_persona is None or p == filter_persona]

    errores = []

    for palabra in palabras_mostrar:
        print(f"\n  ── PALABRA REAL: «{palabra}» ──")

        # Encabezado: una columna por modelo
        col_w = 10
        header_cols = ''.join(f'{p[:8]:>{col_w}}' for p in PALABRAS)
        print(f"  {'Persona':<10} {'#':>2}  {header_cols}  {'PRED':>10}  OK?")
        sep('-')

        for persona in personas_mostrar:
            for i in range(N_TRAIN + 1, N_TRAIN + N_TEST + 1):
                path = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}.npy')
                if not os.path.isfile(path):
                    continue

                seq    = np.load(path).astype(np.int32)
                scores = {w: hmm_models[w].log_likelihood(seq)
                          for w in hmm_models}
                pred   = max(scores, key=lambda w: scores[w])
                ok     = '✓' if pred == palabra else '✗'

                # Normalizar scores para mostrar diferencias relativas
                best  = scores[pred]
                score_str = ''
                for w in PALABRAS:
                    if w not in scores:
                        score_str += f"{'—':>{col_w}}"
                        continue
                    s  = scores[w]
                    diff = s - best  # 0 para el ganador, negativo para los demás
                    if w == pred:
                        tag = f'[{s:.0f}]'
                    elif diff > -50:
                        tag = f'({s:.0f})'  # peligrosamente cerca
                    else:
                        tag = f'{s:.0f}'
                    score_str += f'{tag:>{col_w}}'

                print(f"  {persona:<10} {i:>2}  {score_str}  {pred:>10}  {ok}")

                if pred != palabra:
                    errores.append({
                        'real': palabra, 'pred': pred,
                        'persona': persona, 'idx': i,
                        'margen': scores[pred] - scores[palabra],
                    })

    # Resumen de errores
    sep()
    if not errores:
        print("\n  Sin errores en las grabaciones de prueba filtradas.")
    else:
        print(f"\n  ERRORES ({len(errores)} total):")
        print(f"  {'Real':<12} {'Predicho':<12} {'Persona':<10} {'#':>3} {'Margen LL':>10}")
        sep('-')
        for e in sorted(errores, key=lambda x: -x['margen']):
            print(f"  {e['real']:<12} {e['pred']:<12} {e['persona']:<10} "
                  f"{e['idx']:>3} {e['margen']:>10.1f}")


# ─────────────────────────────────────────────────────────────────────────
# 5. MATRIZ DE CONFUSIÓN TEXTUAL
# ─────────────────────────────────────────────────────────────────────────
def matriz_confusion(filter_persona=None):
    header("MATRIZ DE CONFUSIÓN (prueba)")

    hmm_models = {}
    for palabra in PALABRAS:
        path = os.path.join(HMM_DIR, f'{palabra}_hmm.npz')
        if os.path.isfile(path):
            hmm_models[palabra] = HMMBakis.load(path)

    personas_mostrar = [p for p in PERSONAS if filter_persona is None or p == filter_persona]
    n = len(PALABRAS)
    word_idx  = {w: i for i, w in enumerate(PALABRAS)}
    confusion = np.zeros((n, n), dtype=int)
    total, correct = 0, 0

    for real_idx, palabra in enumerate(PALABRAS):
        for persona in personas_mostrar:
            for i in range(N_TRAIN + 1, N_TRAIN + N_TEST + 1):
                path = os.path.join(SEQ_DIR, f'{persona}_{palabra}_{i:02d}.npy')
                if not os.path.isfile(path):
                    continue
                seq    = np.load(path).astype(np.int32)
                scores = {w: hmm_models[w].log_likelihood(seq) for w in hmm_models}
                pred   = max(scores, key=lambda w: scores[w])
                confusion[real_idx, word_idx[pred]] += 1
                total   += 1
                correct += int(pred == palabra)

    acc = 100.0 * correct / max(total, 1)
    print(f"\n  Accuracy: {correct}/{total} = {acc:.1f}%\n")

    # Tabla de texto
    col_w = 9
    print(f"  {'Real \\ Pred':<12}" + ''.join(f'{w[:7]:>{col_w}}' for w in PALABRAS))
    sep('-')
    for i, palabra in enumerate(PALABRAS):
        row_str = ''
        for j in range(n):
            val = confusion[i, j]
            if i == j:
                row_str += f'[{val}]'.rjust(col_w)
            elif val > 0:
                row_str += f'({val})'.rjust(col_w)
            else:
                row_str += f'·'.rjust(col_w)
        acc_w = 100.0 * confusion[i, i] / max(confusion[i].sum(), 1)
        print(f"  {palabra:<12}{row_str}   {acc_w:5.1f}%")

    # Figura
    try:
        fig, ax = plt.subplots(figsize=(12, 9))
        im = ax.imshow(confusion, cmap='Blues', vmin=0)
        ax.set_xticks(range(n)); ax.set_xticklabels(PALABRAS, rotation=45, ha='right')
        ax.set_yticks(range(n)); ax.set_yticklabels(PALABRAS)
        ax.set_xlabel('Predicho', fontsize=12)
        ax.set_ylabel('Real',     fontsize=12)
        ax.set_title(f'Matriz de Confusión — Accuracy: {acc:.1f}%\n'
                     f'({", ".join(personas_mostrar)})', fontsize=13)
        for r in range(n):
            for c in range(n):
                v = confusion[r, c]
                if v > 0:
                    color = 'white' if v > confusion.max() * 0.5 else 'black'
                    ax.text(c, r, str(v), ha='center', va='center',
                            fontsize=10, color=color, fontweight='bold')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        out = 'debug_confusion.png'
        plt.savefig(out, dpi=150)
        plt.show()
        print(f"\n  Figura: {out}")
    except Exception as e:
        print(f"  [Figura omitida: {e}]")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Diagnóstico del sistema HMM')
    parser.add_argument('--word',    default=None, help='Filtrar por palabra (e.g. start)')
    parser.add_argument('--persona', default=None, help='Filtrar por hablante (e.g. hugo)')
    parser.add_argument('--no-likelihoods', action='store_true',
                        help='Omitir la tabla detallada de log-likelihoods')
    args = parser.parse_args()

    # Validar filtros
    if args.word and args.word not in PALABRAS:
        print(f"⚠ Palabra '{args.word}' no reconocida. Opciones: {PALABRAS}")
        return
    if args.persona and args.persona not in PERSONAS:
        print(f"⚠ Persona '{args.persona}' no reconocida. Opciones: {PERSONAS}")
        return

    resumen_datos(filter_word=args.word, filter_persona=args.persona)
    print()
    resumen_longitudes(filter_word=args.word, filter_persona=args.persona)
    print()
    inspeccionar_modelos(filter_word=args.word)
    print()
    if not args.no_likelihoods:
        tabla_likelihoods(filter_word=args.word, filter_persona=args.persona)
        print()
    matriz_confusion(filter_persona=args.persona)


if __name__ == '__main__':
    main()

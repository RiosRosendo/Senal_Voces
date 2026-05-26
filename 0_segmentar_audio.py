"""
0_segmentar_audio.py
====================
Segmenta los audios del celular de cada integrante del equipo.

ESTRUCTURA DE CARPETAS:
  audios_celular/
    rosendo/
      start.m4a   ← rosendo diciendo "start" 15 veces
      stop.m4a
      left.m4a
      ... (10 palabras)
    jordan/
      start.m4a
      ...
    hugo/ juanjo/ victor/ → igual

RESULTADO:
  voz/
    start/
      rosendo_01.wav  rosendo_02.wav  ...  rosendo_15.wav
      jordan_01.wav   ...
      hugo_01.wav     ...
      juanjo_01.wav   ...
      victor_01.wav   ...
    stop/ left/ right/ ...  (mismo formato)

QUÉ VER AL CORRER:
  · Cuántos segmentos detectó por audio (debe ser 15)
  · Duración de cada segmento
  · Advertencia si detectó más o menos de 15
  · Gráfica de la señal con los cortes marcados

USO:
  python3 0_segmentar_audio.py                        ← procesa todo
  python3 0_segmentar_audio.py --persona rosendo      ← solo una persona
  python3 0_segmentar_audio.py --palabra start        ← solo una palabra
  python3 0_segmentar_audio.py --umbral 0.008         ← umbral más sensible
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal_proc
from pydub import AudioSegment

from utils import PALABRAS, FS

PERSONAS      = ['rosendo', 'jordan', 'hugo', 'juanjo', 'victor']
CELULAR_DIR   = 'audios_celular'
VOZ_DIR       = 'voz'
PLOT_DIR      = os.path.join('resultados', 'segmentacion')
N_SEG         = 15
FORMATOS      = ['.m4a', '.wav', '.mp3', '.ogg', '.flac', '.aac']


# ─────────────────────────────────────────────────────────────────
# Carga de audio (cualquier formato)
# ─────────────────────────────────────────────────────────────────
def cargar_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        fs, data = wav.read(path)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data /= 32768.0
    else:
        seg = AudioSegment.from_file(path).set_channels(1).set_frame_rate(FS)
        data = np.array(seg.get_array_of_samples(), dtype=np.float32)
        data /= 2 ** (seg.sample_width * 8 - 1)
        fs = FS

    if data.ndim > 1:
        data = data.mean(axis=1)
    if fs != FS:
        data = signal_proc.resample(data, int(len(data) * FS / fs))

    return data.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Detección de segmentos por energía
# ─────────────────────────────────────────────────────────────────
def detectar_segmentos(audio, umbral=0.015, max_silencio_s=0.4, min_dur_s=0.15):
    frame_len = int(0.020 * FS)   # 20 ms
    hop       = int(0.010 * FS)   # 10 ms
    n_frames  = (len(audio) - frame_len) // hop

    energy = np.array([
        np.sum(audio[i*hop:i*hop+frame_len] ** 2) / frame_len
        for i in range(n_frames)
    ])

    e_thresh = umbral * np.max(energy) if np.max(energy) > 0 else 1.0
    voz = energy > e_thresh

    # Encontrar bloques contiguos de voz
    regiones = []
    en_voz, inicio = False, 0
    for i, v in enumerate(voz):
        if v and not en_voz:
            inicio, en_voz = i, True
        elif not v and en_voz:
            regiones.append([inicio, i])
            en_voz = False
    if en_voz:
        regiones.append([inicio, n_frames])

    # Fusionar regiones separadas por silencio corto
    gap_max = int(max_silencio_s * FS / hop)
    fusionadas = []
    for r in regiones:
        if fusionadas and r[0] - fusionadas[-1][1] <= gap_max:
            fusionadas[-1][1] = r[1]
        else:
            fusionadas.append(r[:])

    # Filtrar por duración mínima
    min_fr = int(min_dur_s * FS / hop)
    fusionadas = [r for r in fusionadas if r[1] - r[0] >= min_fr]

    # Convertir a muestras con margen de 60 ms
    pad = int(0.06 * FS)
    segmentos = [(max(0, s*hop - pad), min(len(audio), e*hop + frame_len + pad))
                 for s, e in fusionadas]

    return segmentos, energy, e_thresh, hop


# ─────────────────────────────────────────────────────────────────
# Guardar WAVs segmentados
# ─────────────────────────────────────────────────────────────────
def guardar_segmentos(audio, segmentos, persona, palabra):
    carpeta = os.path.join(VOZ_DIR, palabra)
    os.makedirs(carpeta, exist_ok=True)
    for i, (s, e) in enumerate(segmentos, 1):
        chunk = np.clip(audio[s:e], -1.0, 1.0)
        ruta  = os.path.join(carpeta, f'{persona}_{i:02d}.wav')
        wav.write(ruta, FS, (chunk * 32767).astype(np.int16))


# ─────────────────────────────────────────────────────────────────
# Gráfica de diagnóstico
# ─────────────────────────────────────────────────────────────────
def graficar(audio, segmentos, energy, e_thresh, hop, persona, palabra):
    os.makedirs(PLOT_DIR, exist_ok=True)
    t   = np.arange(len(audio)) / FS
    t_e = np.arange(len(energy)) * hop / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    axes[0].plot(t, audio, color='steelblue', linewidth=0.4)
    for i, (s, e) in enumerate(segmentos):
        axes[0].axvspan(s/FS, e/FS, color='green', alpha=0.3)
        axes[0].text((s+e)/(2*FS), 0.85, str(i+1), ha='center',
                     fontsize=8, color='darkgreen',
                     transform=axes[0].get_xaxis_transform())
    axes[0].set_title(f'{persona} – {palabra}  ({len(segmentos)} segmentos detectados)')
    axes[0].set_ylabel('Amplitud')

    axes[1].plot(t_e, energy, color='darkorange', linewidth=0.5)
    axes[1].axhline(e_thresh, color='red', linestyle='--', linewidth=1,
                    label=f'umbral={e_thresh:.5f}')
    axes[1].set_title('Energía por frame')
    axes[1].set_xlabel('Tiempo [s]')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{persona}_{palabra}.png'), dpi=90)
    plt.close()


# ─────────────────────────────────────────────────────────────────
# Procesar una persona + palabra
# ─────────────────────────────────────────────────────────────────
def buscar_audio(persona, palabra):
    for ext in FORMATOS:
        p = os.path.join(CELULAR_DIR, persona, f'{palabra}{ext}')
        if os.path.isfile(p):
            return p
    return None


def procesar(persona, palabra, umbral, max_silencio, forzar):
    path = buscar_audio(persona, palabra)
    if path is None:
        return 'falta'

    audio = cargar_audio(path)
    segs, energy, e_thresh, hop = detectar_segmentos(audio, umbral, max_silencio)

    ok = len(segs) == N_SEG
    if not ok:
        simbolo = '⚠ ' if len(segs) < N_SEG else '⚠ '
        print(f'    {simbolo} {persona}/{palabra}: {len(segs)} segmentos (esperados {N_SEG})')
        if not forzar:
            return 'error'

    guardar_segmentos(audio, segs, persona, palabra)
    graficar(audio, segs, energy, e_thresh, hop, persona, palabra)
    return 'ok' if ok else 'parcial'


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--persona',   default=None, choices=PERSONAS)
    ap.add_argument('--palabra',   default=None, choices=PALABRAS)
    ap.add_argument('--umbral',    type=float, default=0.015)
    ap.add_argument('--silencio',  type=float, default=0.4)
    ap.add_argument('--forzar',    action='store_true',
                    help='Guardar aunque no sean exactamente 15 segmentos')
    args = ap.parse_args()

    personas = [args.persona] if args.persona else PERSONAS
    palabras = [args.palabra] if args.palabra else PALABRAS

    print('\n' + '=' * 60)
    print('  SEGMENTACIÓN DE AUDIOS — 5 integrantes')
    print('=' * 60)
    print(f'  Umbral energía : {args.umbral}')
    print(f'  Silencio max   : {args.silencio} s')
    print()

    resumen = {'ok': 0, 'parcial': 0, 'falta': 0, 'error': 0}

    for persona in personas:
        print(f'  ── {persona.upper()} ──')
        for palabra in palabras:
            estado = procesar(persona, palabra, args.umbral, args.silencio, args.forzar)
            resumen[estado] += 1
            if estado == 'ok':
                print(f'    ✓  {palabra}')
            elif estado == 'falta':
                print(f'    –  {palabra}  (archivo no encontrado en audios_celular/{persona}/)')
        print()

    print('=' * 60)
    print(f"  ✓ OK         : {resumen['ok']}")
    print(f"  ⚠ Parcial    : {resumen['parcial']}")
    print(f"  – Falta      : {resumen['falta']}")
    print(f"  ✗ Error      : {resumen['error']}")
    print()
    print('  Archivos en: voz/<palabra>/<persona>_XX.wav')
    print()
    print('  Si alguna palabra tiene ≠ 15 segmentos, reintenta con:')
    print('    python3 0_segmentar_audio.py --persona rosendo --umbral 0.008')
    print('    python3 0_segmentar_audio.py --persona rosendo --umbral 0.030')
    print()
    print('  Siguiente paso:')
    print('    python3 6_extraer_mfcc.py')
    print('=' * 60)


if __name__ == '__main__':
    main()

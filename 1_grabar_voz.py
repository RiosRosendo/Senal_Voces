"""
1_grabar_voz.py
===============
Graba 15 muestras por cada una de las 10 palabras a 16 kHz.
Cada integrante del equipo corre este script UNA vez.

Archivos generados:
  voz/<palabra>/<nombre>_01.wav  …  voz/<palabra>/<nombre>_15.wav

Pipeline:
  1. python 1_grabar_voz.py        ← cada integrante
  2. python 6_extraer_mfcc.py      ← después de que TODOS grabaron
  3. python 7_entrenar_hmm.py
  4. python 8_evaluar_hmm.py
  5. python escuchar.py            ← prueba en vivo
"""

import os
import sys
import time
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

from utils import FS, PALABRAS, PERSONAS, detect_endpoints

DURACION   = 2.5      # segundos por grabación
N_MUESTRAS = 15       # repeticiones por palabra (10 train + 5 test)
VOZ_DIR    = "voz"


# ── Audio ─────────────────────────────────────────────────────────────────
def grabar(duracion=DURACION, fs=FS, device=None):
    audio = sd.rec(int(duracion * fs), samplerate=fs,
                   channels=1, dtype='float32', device=device)
    sd.wait()
    return audio.flatten()


def reproducir(signal, fs=FS):
    sd.play(signal, samplerate=fs, device=None)  # siempre salida por defecto
    sd.wait()


def guardar_wav(signal, path, fs=FS):
    clip = np.clip(signal, -1.0, 1.0)
    wav.write(path, fs, (clip * 32767).astype(np.int16))


def evaluar_calidad(audio, fs=FS):
    """
    Evalúa si una grabación es buena o necesita regrabarse.
    Retorna (veredicto_str, es_buena: bool)
    """
    nivel = float(np.max(np.abs(audio)))
    barra = '█' * int(nivel * 20)

    # Detectar segmento de voz
    start, end = detect_endpoints(audio)
    dur_voz = (end - start) / fs
    dur_total = len(audio) / fs

    # Criterios
    nivel_ok  = nivel >= 0.10
    voz_ok    = dur_voz >= 0.25          # al menos 0.25 s de voz detectada
    no_clip   = nivel < 0.98             # señal no saturada

    problemas = []
    if not nivel_ok:
        problemas.append("señal muy baja, habla más fuerte")
    if not voz_ok:
        problemas.append(f"voz muy corta ({dur_voz:.2f}s), habla más claro")
    if not no_clip:
        problemas.append("señal saturada, aléjate del micrófono")

    es_buena = len(problemas) == 0

    linea_nivel = f"nivel={nivel:.2f} |{barra:<20}|  voz={dur_voz:.2f}s/{dur_total:.1f}s"

    if es_buena:
        veredicto = f"  {linea_nivel}\n  ✅ BUENA"
    else:
        veredicto = f"  {linea_nivel}\n  ⚠  REGRABAR — {', '.join(problemas)}"

    return veredicto, es_buena


# ── Selección de persona ──────────────────────────────────────────────────
def seleccionar_persona():
    print("\n" + "=" * 55)
    print("  ¿Quién está grabando?")
    print("=" * 55)
    for i, p in enumerate(PERSONAS, 1):
        print(f"    {i}. {p}")
    print(f"    n. Mi nombre no está en la lista")
    print("=" * 55)

    while True:
        op = input("  Selecciona (número o 'n'): ").strip().lower()
        if op == 'n':
            nombre = input("  Escribe tu nombre (sin espacios, minúsculas): ").strip().lower()
            if nombre:
                return nombre
        elif op.isdigit() and 1 <= int(op) <= len(PERSONAS):
            return PERSONAS[int(op) - 1]
        print("  Opción no válida.")


# ── Selección de dispositivo ──────────────────────────────────────────────
def seleccionar_dispositivo():
    print("\n  Probando dispositivos de entrada compatibles con sounddevice...")
    devs = sd.query_devices()
    compatibles = []
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0:
            try:
                sd.check_input_settings(device=i, samplerate=FS, channels=1)
                compatibles.append((i, d['name']))
                print(f"    {i:2d}. {d['name']}")
            except Exception:
                pass  # dispositivo no compatible, se omite

    if not compatibles:
        print("    (ninguno compatible encontrado — se usará el del sistema)")
        return None

    print("    d. Usar dispositivo por defecto del sistema")
    op = input("\n  Número de dispositivo (o 'd' para defecto): ").strip().lower()
    if op == 'd' or op == '':
        return None
    if op.isdigit() and any(i == int(op) for i, _ in compatibles):
        return int(op)
    print("  Opción no válida, usando dispositivo por defecto.")
    return None


# ── Grabación de una palabra ──────────────────────────────────────────────
def grabar_palabra(persona, palabra, device, inicio=1):
    carpeta = os.path.join(VOZ_DIR, palabra)
    os.makedirs(carpeta, exist_ok=True)

    pendientes = N_MUESTRAS - inicio + 1
    print(f"\n{'='*55}")
    print(f"  Persona  : {persona}")
    print(f"  Palabra  : «{palabra.upper()}»")
    print(f"  Faltan   : {pendientes} de {N_MUESTRAS}  (grabando {inicio}–{N_MUESTRAS})")
    print(f"  Duración : {DURACION} s por muestra")
    print(f"{'='*55}")

    i = inicio
    while i <= N_MUESTRAS:
        ruta = os.path.join(carpeta, f"{persona}_{i:02d}.wav")

        print(f"\n  [{i}/{N_MUESTRAS}] Di «{palabra}» en:", end='', flush=True)
        for cuenta in [3, 2, 1]:
            print(f"  {cuenta}...", end='', flush=True)
            time.sleep(1.0)
        print("  ¡GRABA!")

        audio = grabar(device=device)
        guardar_wav(audio, ruta)

        veredicto, es_buena = evaluar_calidad(audio)
        print(veredicto)

        if not es_buena:
            # Solo pregunta cuando la calidad es mala
            r = input("  r=regrabar / ENTER=continuar: ").strip().lower()
            if r == 'r':
                print("  Regrabando...")
                continue

        i += 1
        time.sleep(0.5)  # pausa breve antes de la siguiente

    print(f"\n  ✓ «{palabra}» completada.\n")


# ── Regrabar grabaciones específicas ─────────────────────────────────────
def regrabar_especificas(persona, device):
    # Elegir palabra
    print("\n  ¿De qué palabra quieres regrabar?")
    for idx, p in enumerate(PALABRAS, 1):
        print(f"    {idx:2d}. {p}")
    op = input("  Número de palabra: ").strip()
    if not op.isdigit() or not (1 <= int(op) <= len(PALABRAS)):
        print("  Opción no válida.")
        return
    palabra = PALABRAS[int(op) - 1]
    carpeta = os.path.join(VOZ_DIR, palabra)

    # Mostrar grabaciones existentes con nivel
    print(f"\n  Grabaciones de {persona} para «{palabra}»:")
    existentes = []
    for i in range(1, N_MUESTRAS + 1):
        ruta = os.path.join(carpeta, f"{persona}_{i:02d}.wav")
        if os.path.isfile(ruta):
            try:
                import scipy.io.wavfile as wf
                _, data = wf.read(ruta)
                sig = data.astype(np.float32) / 32767.0
                nivel = np.max(np.abs(sig))
                barra = '█' * int(nivel * 20)
                aviso = '  ⚠ bajo' if nivel < 0.10 else ''
                print(f"    {i:2d}. nivel={nivel:.2f} |{barra:<20}|{aviso}")
            except Exception:
                print(f"    {i:2d}. ⚠ ARCHIVO CORRUPTO — necesita regrabarse")
            existentes.append(i)
        else:
            print(f"    {i:2d}. (no existe)")

    # Elegir cuáles regrabar
    print("\n  Escribe los números a regrabar separados por coma (ej: 3,7,12)")
    print("  o ENTER para cancelar:")
    raw = input("  > ").strip()
    if not raw:
        return

    indices = []
    for tok in raw.split(','):
        tok = tok.strip()
        if tok.isdigit() and 1 <= int(tok) <= N_MUESTRAS:
            indices.append(int(tok))

    if not indices:
        print("  No se seleccionó ningún índice válido.")
        return

    # Regrabar uno a uno
    for i in sorted(indices):
        ruta = os.path.join(carpeta, f"{persona}_{i:02d}.wav")
        os.makedirs(carpeta, exist_ok=True)

        print(f"\n  Regrabando [{i}/{N_MUESTRAS}] «{palabra}» en:", end='', flush=True)
        for cuenta in [3, 2, 1]:
            print(f"  {cuenta}...", end='', flush=True)
            time.sleep(1.0)
        print("  ¡GRABA!")

        audio = grabar(device=device)
        guardar_wav(audio, ruta)

        veredicto, es_buena = evaluar_calidad(audio)
        print(veredicto)

        if es_buena:
            r = input("  ¿Reproducir? (ENTER=siguiente / p=reproducir / r=regrabar): ").strip().lower()
        else:
            r = input("  ¿Qué hacemos? (r=regrabar [recomendado] / ENTER=guardar igual / p=reproducir): ").strip().lower()

        if r == 'r':
            indices.insert(indices.index(i) + 1, i)
            continue
        if r == 'p':
            print("  ▶ Reproduciendo...", end='', flush=True)
            reproducir(audio)
            print("  [fin]")
            r2 = input("  ¿Regrabar? (r=sí / ENTER=no): ").strip().lower()
            if r2 == 'r':
                indices.insert(indices.index(i) + 1, i)
                continue

    print(f"\n  Regrabaciones completadas para «{palabra}».")


# ── Menú principal ────────────────────────────────────────────────────────
def menu(persona, device):
    while True:
        print("\n" + "=" * 55)
        print(f"  Hablante: {persona}   |   {len(PALABRAS)} palabras × {N_MUESTRAS} muestras")
        print("=" * 55)

        # Estado de cada palabra para esta persona
        total_ok = 0
        for idx, p in enumerate(PALABRAS, 1):
            carpeta = os.path.join(VOZ_DIR, p)
            grabadas = 0
            if os.path.exists(carpeta):
                grabadas = sum(
                    1 for f in os.listdir(carpeta)
                    if f.startswith(persona + '_') and f.endswith('.wav')
                )
            estado = f"✓ {grabadas}/{N_MUESTRAS}" if grabadas >= N_MUESTRAS \
                     else f"  {grabadas}/{N_MUESTRAS}"
            total_ok += min(grabadas, N_MUESTRAS)
            print(f"    {idx:2d}. {p:<12}  {estado}")

        completado = total_ok / (len(PALABRAS) * N_MUESTRAS) * 100
        print(f"\n    0. Grabar TODAS las palabras que falten")
        print(f"    r. Regrabar grabaciones específicas")
        print(f"    q. Salir")
        print(f"\n  Progreso: {total_ok}/{len(PALABRAS)*N_MUESTRAS}  ({completado:.0f}%)")

        op = input("\n  Opción: ").strip().lower()

        if op == 'q':
            print(f"\n  Sesión de {persona} guardada. ¡Hasta luego!")
            break

        elif op == '0':
            for p in PALABRAS:
                carpeta = os.path.join(VOZ_DIR, p)
                grabadas = sum(
                    1 for f in os.listdir(carpeta)
                    if f.startswith(persona + '_') and f.endswith('.wav')
                ) if os.path.exists(carpeta) else 0
                if grabadas < N_MUESTRAS:
                    grabar_palabra(persona, p, device, inicio=grabadas + 1)
            print("\n  ¡Todas las palabras completadas!")

        elif op == 'r':
            regrabar_especificas(persona, device)

        elif op.isdigit() and 1 <= int(op) <= len(PALABRAS):
            p = PALABRAS[int(op) - 1]
            carpeta = os.path.join(VOZ_DIR, p)
            grabadas = sum(
                1 for f in os.listdir(carpeta)
                if f.startswith(persona + '_') and f.endswith('.wav')
            ) if os.path.exists(carpeta) else 0

            if grabadas >= N_MUESTRAS:
                r = input(f"  «{p}» ya tiene {N_MUESTRAS} grabaciones. "
                          "¿Re-grabar desde 1? (s/n): ").strip().lower()
                if r == 's':
                    grabar_palabra(persona, p, device, inicio=1)
            else:
                grabar_palabra(persona, p, device, inicio=grabadas + 1)

        else:
            print("  Opción no válida.")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  GRABACIÓN DE VOZ — HMM + MFCC + VQ")
    print(f"  {len(PALABRAS)} palabras  ×  {N_MUESTRAS} muestras  ×  {DURACION} s")
    print("=" * 55)

    persona = seleccionar_persona()
    device  = seleccionar_dispositivo()

    os.makedirs(VOZ_DIR, exist_ok=True)
    menu(persona, device)


if __name__ == "__main__":
    main()

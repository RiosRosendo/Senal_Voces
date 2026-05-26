"""
Script 1 – Grabación de voz
───────────────────────────
Graba 15 muestras por cada una de las 10 palabras a 16 kHz.
Las guarda en:  voz/<palabra>/01.wav … 15.wav

Al ejecutar verás:
  • Menú para seleccionar qué palabra grabar (o todas)
  • Cuenta regresiva de 1 s antes de cada grabación
  • Confirmación de guardado por archivo
"""

import os
import time
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

from utils import FS, PALABRAS

DURACION   = 2.0        # segundos por grabación
N_MUESTRAS = 15         # repeticiones por palabra
VOZ_DIR    = "voz"

# Dispositivo 9 = sof-hda-dsp (hw:1,7) — micrófono real del equipo
DEVICE_IN  = 9


def grabar(duracion=DURACION, fs=FS):
    """Graba `duracion` segundos y devuelve array float32."""
    audio = sd.rec(int(duracion * fs), samplerate=fs,
                   channels=1, dtype='float32', device=DEVICE_IN)
    sd.wait()
    return audio.flatten()


def guardar_wav(signal, path, fs=FS):
    int16 = np.clip(signal, -1.0, 1.0)
    int16 = (int16 * 32767).astype(np.int16)
    wav.write(path, fs, int16)


def grabar_palabra(palabra, inicio=1):
    """
    Graba grabaciones `inicio` a 15 para una palabra.
    Permite retomar si ya hay archivos previos.
    """
    carpeta = os.path.join(VOZ_DIR, palabra)
    os.makedirs(carpeta, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Palabra: «{palabra.upper()}»")
    print(f"  Se grabarán {N_MUESTRAS - inicio + 1} muestras "
          f"({inicio} a {N_MUESTRAS})")
    print(f"  Duración por muestra: {DURACION} s")
    print(f"{'='*50}")
    input("  Presiona ENTER para comenzar...")

    for i in range(inicio, N_MUESTRAS + 1):
        ruta = os.path.join(carpeta, f"{i:02d}.wav")

        print(f"\n  [{i}/{N_MUESTRAS}] Di la palabra «{palabra}» en 3...")
        time.sleep(1.0)
        print("  2...")
        time.sleep(1.0)
        print("  1...")
        time.sleep(1.0)
        print("  ¡GRABANDO!")

        audio = grabar()
        guardar_wav(audio, ruta)

        print(f"  ✓ Guardado: {ruta}")
        time.sleep(0.5)

    print(f"\n  ✓ Palabra «{palabra}» completada.\n")


def menu_interactivo():
    print("\n" + "="*60)
    print("  SISTEMA DE GRABACIÓN DE VOZ – Cuantización Vectorial")
    print("="*60)
    print(f"  Frecuencia de muestreo : {FS} Hz")
    print(f"  Muestras por palabra   : {N_MUESTRAS}")
    print(f"  Duración por muestra   : {DURACION} s")
    print("="*60)
    print("\n  Palabras disponibles:")
    for idx, p in enumerate(PALABRAS, 1):
        carpeta = os.path.join(VOZ_DIR, p)
        grabadas = len([f for f in os.listdir(carpeta) if f.endswith(".wav")]) \
                   if os.path.exists(carpeta) else 0
        estado = f"✓ {grabadas}/{N_MUESTRAS}" if grabadas == N_MUESTRAS else \
                 f"  {grabadas}/{N_MUESTRAS}"
        print(f"    {idx:2d}. {p:<12} {estado}")

    print(f"    {'0':>2}. Grabar TODAS las palabras")
    print(f"    {'q':>2}. Salir")

    return input("\n  Selecciona una opción: ").strip().lower()


def main():
    os.makedirs(VOZ_DIR, exist_ok=True)

    while True:
        opcion = menu_interactivo()

        if opcion == 'q':
            print("\n  Saliendo...\n")
            break

        elif opcion == '0':
            for palabra in PALABRAS:
                carpeta = os.path.join(VOZ_DIR, palabra)
                grabadas = len([f for f in os.listdir(carpeta)
                                if f.endswith(".wav")]) \
                           if os.path.exists(carpeta) else 0
                if grabadas < N_MUESTRAS:
                    grabar_palabra(palabra, inicio=grabadas + 1)

        elif opcion.isdigit() and 1 <= int(opcion) <= len(PALABRAS):
            palabra  = PALABRAS[int(opcion) - 1]
            carpeta  = os.path.join(VOZ_DIR, palabra)
            grabadas = len([f for f in os.listdir(carpeta)
                            if f.endswith(".wav")]) \
                       if os.path.exists(carpeta) else 0

            if grabadas >= N_MUESTRAS:
                r = input(f"  «{palabra}» ya tiene {N_MUESTRAS} grabaciones. "
                          f"¿Regrabar desde 1? (s/n): ").strip().lower()
                if r == 's':
                    grabar_palabra(palabra, inicio=1)
            else:
                grabar_palabra(palabra, inicio=grabadas + 1)

        else:
            print("  Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    main()

"""
Reproductor de grabaciones para verificar calidad del audio.
Permite escuchar cualquier grabación de voz/  o voz_preprocesadas/
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import matplotlib.pyplot as plt

from utils import PALABRAS

VOZ_DIR    = "voz"
PRE_DIR    = "voz_preprocesadas"
DEVICE_OUT = "pulse"   # salida de audio
DEVICE_IN  = 9         # micrófono real (sof-hda-dsp hw:1,7)


def cargar_wav(path):
    fs, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483647.0
    if data.ndim > 1:
        data = data[:, 0]
    return fs, data


def reproducir(path):
    fs, data = cargar_wav(path)
    duracion = len(data) / fs
    nivel    = np.max(np.abs(data))
    rms      = np.sqrt(np.mean(data ** 2))

    print(f"\n  Archivo  : {path}")
    print(f"  Duración : {duracion:.3f} s")
    print(f"  Nivel    : {nivel:.4f}  RMS: {rms:.4f}")

    if nivel < 0.005:
        print("  ⚠ Señal muy baja — posible grabación vacía")
    else:
        print("  ▶ Reproduciendo...")
        sd.play(data, fs, device=DEVICE_OUT)
        sd.wait()
        print("  ■ Fin")


def graficar_señal(path):
    fs, data = cargar_wav(path)
    t = np.arange(len(data)) / fs

    plt.figure(figsize=(10, 3))
    plt.plot(t, data, color='steelblue', linewidth=0.5)
    plt.title(os.path.basename(path))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def listar_archivos(carpeta, palabra):
    path = os.path.join(carpeta, palabra)
    if not os.path.exists(path):
        return []
    return sorted(f for f in os.listdir(path) if f.endswith(".wav"))


def menu():
    print("\n" + "=" * 55)
    print("  REPRODUCTOR DE GRABACIONES DE VOZ")
    print("=" * 55)
    print("  Palabras disponibles:")

    palabras_ok = []
    for i, p in enumerate(PALABRAS, 1):
        orig = listar_archivos(VOZ_DIR, p)
        pre  = listar_archivos(PRE_DIR, p)
        if orig or pre:
            palabras_ok.append(p)
            print(f"    {i:2d}. {p:<12}  original:{len(orig):>2}  preprocesada:{len(pre):>2}")
        else:
            print(f"    {i:2d}. {p:<12}  (sin grabaciones)")

    print(f"\n    0. Salir")
    return palabras_ok


def submenu_palabra(palabra, palabras_ok):
    orig = listar_archivos(VOZ_DIR, palabra)
    pre  = listar_archivos(PRE_DIR, palabra)

    while True:
        print(f"\n  ── «{palabra.upper()}» ──")
        print(f"  1. Escuchar grabación ORIGINAL")
        print(f"  2. Escuchar grabación PREPROCESADA")
        print(f"  3. Escuchar TODAS las originales seguidas")
        print(f"  4. Ver forma de onda")
        print(f"  5. Volver al menú principal")

        op = input("  Opción: ").strip()

        if op == "1":
            if not orig:
                print("  Sin grabaciones originales.")
                continue
            print(f"\n  Archivos: {orig}")
            num = input(f"  Número (01-{len(orig):02d}): ").strip().zfill(2)
            path = os.path.join(VOZ_DIR, palabra, f"{num}.wav")
            if os.path.exists(path):
                reproducir(path)
            else:
                print(f"  No existe: {path}")

        elif op == "2":
            if not pre:
                print("  Sin grabaciones preprocesadas.")
                continue
            print(f"\n  Archivos: {pre}")
            num = input(f"  Número (01-{len(pre):02d}): ").strip().zfill(2)
            path = os.path.join(PRE_DIR, palabra, f"{num}.wav")
            if os.path.exists(path):
                reproducir(path)
            else:
                print(f"  No existe: {path}")

        elif op == "3":
            if not orig:
                print("  Sin grabaciones originales.")
                continue
            print(f"\n  Reproduciendo {len(orig)} grabaciones de «{palabra}»...")
            for fname in orig:
                path = os.path.join(VOZ_DIR, palabra, fname)
                print(f"\n  [{fname}]", end="", flush=True)
                fs, data = cargar_wav(path)
                nivel = np.max(np.abs(data))
                estado = "✓" if nivel > 0.005 else "⚠ baja"
                print(f"  nivel={nivel:.3f} {estado}")
                sd.play(data, fs, device=DEVICE_OUT)
                sd.wait()
            print("\n  ■ Fin de todas las grabaciones")

        elif op == "4":
            fuente = input("  Ver (1=original / 2=preprocesada): ").strip()
            archivos = orig if fuente == "1" else pre
            directorio = VOZ_DIR if fuente == "1" else PRE_DIR
            if not archivos:
                print("  Sin archivos.")
                continue
            num = input(f"  Número (01-{len(archivos):02d}): ").strip().zfill(2)
            path = os.path.join(directorio, palabra, f"{num}.wav")
            if os.path.exists(path):
                graficar_señal(path)
            else:
                print(f"  No existe: {path}")

        elif op == "5":
            break
        else:
            print("  Opción no válida.")


def main():
    while True:
        palabras_ok = menu()

        if not palabras_ok:
            print("\n  No hay grabaciones. Corre primero 1_grabar_voz.py\n")
            break

        op = input("\n  Selecciona palabra (número) o 0 para salir: ").strip()

        if op == "0":
            print("\n  Saliendo...\n")
            break
        elif op.isdigit() and 1 <= int(op) <= len(PALABRAS):
            palabra = PALABRAS[int(op) - 1]
            submenu_palabra(palabra, palabras_ok)
        else:
            print("  Opción no válida.")


if __name__ == "__main__":
    main()

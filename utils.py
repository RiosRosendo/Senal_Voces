"""
Funciones compartidas para el sistema de reconocimiento de voz con
cuantización vectorial.

Pipeline:
  preénfasis → ventaneo Hamming → detección VAD → LPC(12) → LSF
"""

import numpy as np
from scipy.linalg import toeplitz

# ──────────────────────────────────────────────
# Parámetros globales del sistema
# ──────────────────────────────────────────────
FS           = 16000   # Hz
FRAME_LEN    = 320     # muestras (~20 ms)
HOP          = 128     # muestras (~8 ms)
LPC_ORDER    = 12
ALPHA        = 0.95    # coeficiente de preénfasis
PALABRAS     = ["start", "stop", "left", "right", "forward",
                "backward", "lift", "leave", "break", "continue"]
N_TRAIN      = 10      # grabaciones para entrenamiento por palabra
N_TEST       = 5       # grabaciones para evaluación
CODEBOOK_SIZES = [16, 32, 64]


# ──────────────────────────────────────────────
# 1. Preénfasis
# ──────────────────────────────────────────────
def apply_preemphasis(signal, alpha=ALPHA):
    """H(z) = 1 - alpha·z^{-1}"""
    return np.concatenate([[signal[0]], signal[1:] - alpha * signal[:-1]])


# ──────────────────────────────────────────────
# 2. Ventaneo Hamming
# ──────────────────────────────────────────────
def hamming_frames(signal, frame_len=FRAME_LEN, hop=HOP):
    """Divide la señal en frames con ventana de Hamming."""
    n = len(signal)
    n_frames = max(1, 1 + (n - frame_len) // hop)
    window = np.hamming(frame_len)
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        start = i * hop
        end   = start + frame_len
        chunk = signal[start:min(end, n)]
        frames[i, :len(chunk)] = chunk * window[:len(chunk)]
    return frames


# ──────────────────────────────────────────────
# 3. Detección de actividad de voz (VAD)
# ──────────────────────────────────────────────
def compute_energy(frame):
    return np.sum(frame ** 2) / max(len(frame), 1)

def compute_zcr(frame):
    return np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * max(len(frame), 1))

def detect_endpoints(signal, frame_len=FRAME_LEN, hop=HOP):
    """
    Detecta inicio y fin de la palabra usando energía y ZCR.
    Retorna (start_sample, end_sample).
    """
    n = len(signal)
    n_frames = max(1, 1 + (n - frame_len) // hop)
    energy = np.zeros(n_frames)
    zcr    = np.zeros(n_frames)

    for i in range(n_frames):
        s = i * hop
        frame = signal[s:s + frame_len]
        energy[i] = compute_energy(frame)
        zcr[i]    = compute_zcr(frame)

    # Solo energía para detectar voz (más robusto que AND con ZCR)
    e_thresh = 0.02 * np.max(energy)
    voice    = energy > e_thresh

    voiced = np.where(voice)[0]
    if len(voiced) == 0:
        return 0, n  # sin voz detectada, devolver señal completa

    first_frame = voiced[0]
    last_frame  = voiced[-1]

    # margen de 4 frames para no cortar el inicio/fin de la palabra
    first_frame = max(0, first_frame - 4)
    last_frame  = min(n_frames - 1, last_frame + 4)

    start_sample = first_frame * hop
    end_sample   = min(last_frame * hop + frame_len, n)

    # garantizar duración mínima de 0.3 s (4800 muestras)
    MIN_SAMPLES = int(0.3 * FS)
    if end_sample - start_sample < MIN_SAMPLES:
        return 0, n

    return start_sample, end_sample


# ──────────────────────────────────────────────
# 4. LPC con Levinson-Durbin
# ──────────────────────────────────────────────
def compute_lpc(frame, order=LPC_ORDER):
    """
    Coeficientes LPC por autocorrelación + Levinson-Durbin.
    Retorna (lpc: array de longitud `order`, gain: error de predicción).
    """
    n = len(frame)
    # Autocorrelación
    r = np.array([np.dot(frame[:n - k], frame[k:]) for k in range(order + 1)])

    if r[0] < 1e-10:
        return np.zeros(order), 1.0

    # Levinson-Durbin
    a = np.zeros(order)
    e = r[0]

    for m in range(order):
        if e < 1e-10:
            break
        lam = -(r[m + 1] + np.dot(a[:m], r[m:0:-1])) / e
        a_new       = a.copy()
        if m > 0:
            a_new[:m] = a[:m] + lam * a[m - 1::-1]
        a_new[m] = lam
        a = a_new
        e = e * (1.0 - lam ** 2)
        if e < 0:
            e = 1e-10
            break

    return a, max(e, 1e-10)


# ──────────────────────────────────────────────
# 5. LPC → LSF
# ──────────────────────────────────────────────
def lpc_to_lsf(lpc):
    """
    Convierte coeficientes LPC a LSF (Line Spectral Frequencies) en radianes.
    Retorna array de longitud `order`, valores en (0, π).
    """
    order = len(lpc)
    a = np.concatenate([[1.0], lpc])          # [1, a1, ..., ap]
    a_pad     = np.concatenate([a, [0.0]])    # [1, a1, ..., ap, 0]
    a_rev_pad = np.concatenate([[0.0], a[::-1]])  # [0, ap, ..., a1, 1]

    P = a_pad + a_rev_pad   # polinomio simétrico
    Q = a_pad - a_rev_pad   # polinomio antisimétrico

    def _get_angles(poly, divisor):
        """Divide el polinomio por `divisor` y extrae ángulos de raíces en |z|≈1."""
        quotient, rem = np.polydiv(poly, divisor)
        roots = np.roots(quotient)
        angles = []
        for r in roots:
            if abs(abs(r) - 1.0) < 0.08 and np.imag(r) > 1e-6:
                angles.append(float(np.angle(r)))
        return sorted(angles)

    if order % 2 == 0:
        p_angles = _get_angles(P, [1.0,  1.0])   # raíz trivial z=-1
        q_angles = _get_angles(Q, [1.0, -1.0])   # raíz trivial z=+1
    else:
        p_angles = _get_angles(P, [1.0,  1.0])
        q_angles = _get_angles(Q, [1.0, -1.0])

    lsf = sorted(p_angles + q_angles)

    if len(lsf) < order:
        # Respaldo numérico: espaciado lineal
        lsf = list(np.linspace(0.05 * np.pi, 0.95 * np.pi, order))

    return np.array(lsf[:order])


# ──────────────────────────────────────────────
# 6. LSF → LPC  (necesario para distancia IS en codebook)
# ──────────────────────────────────────────────
def lsf_to_lpc(lsf):
    """
    Reconstruye coeficientes LPC a partir de LSF en radianes.
    """
    order = len(lsf)
    P_z = np.array([1.0])
    Q_z = np.array([1.0])

    for i, omega in enumerate(lsf):
        factor = np.array([1.0, -2.0 * np.cos(omega), 1.0])
        if i % 2 == 0:
            P_z = np.convolve(P_z, factor)
        else:
            Q_z = np.convolve(Q_z, factor)

    # Agregar raíces triviales
    if order % 2 == 0:
        P_z = np.convolve(P_z, [1.0,  1.0])  # (z + 1)
        Q_z = np.convolve(Q_z, [1.0, -1.0])  # (z - 1)
    else:
        P_z = np.convolve(P_z, [1.0,  1.0])
        Q_z = np.convolve(Q_z, [1.0, -1.0])

    A_coeffs = (P_z + Q_z) / 2.0
    A_coeffs /= A_coeffs[0]
    return A_coeffs[1:order + 1]


# ──────────────────────────────────────────────
# 7. Extracción de características por señal
# ──────────────────────────────────────────────
def extract_features(signal_trimmed, fs=FS, lpc_order=LPC_ORDER,
                     frame_len=FRAME_LEN, hop=HOP):
    """
    Extrae características frame a frame de una señal recortada.
    Retorna dict con claves:
      'lsf'  : (N_frames, lpc_order)
      'lpc'  : (N_frames, lpc_order)
      'gain' : (N_frames,)
      'acf'  : (N_frames, lpc_order+1)   autocorrelación por frame
    """
    x = apply_preemphasis(signal_trimmed)
    frames = hamming_frames(x, frame_len, hop)

    lsf_list  = []
    lpc_list  = []
    gain_list = []
    acf_list  = []

    for frame in frames:
        lpc, gain = compute_lpc(frame, lpc_order)
        lsf = lpc_to_lsf(lpc)
        n   = len(frame)
        acf = np.array([np.dot(frame[:n - k], frame[k:])
                        for k in range(lpc_order + 1)])

        lsf_list.append(lsf)
        lpc_list.append(lpc)
        gain_list.append(gain)
        acf_list.append(acf)

    return {
        'lsf':  np.array(lsf_list),
        'lpc':  np.array(lpc_list),
        'gain': np.array(gain_list),
        'acf':  np.array(acf_list),
    }


# ──────────────────────────────────────────────
# 8. Distancia de Itakura-Saito
# ──────────────────────────────────────────────
def itakura_saito_distance(lpc_a, gain_a, acf_a, lpc_b):
    """
    Distancia Itakura-Saito entre el frame A (modelo LPC) y el vector
    de codebook B (también en espacio LPC).

    d_IS(A→B) = (b^T R_a b) / gain_a  - log((b^T R_a b) / gain_a) - 1

    donde R_a es la matriz de Toeplitz de autocorrelación del frame A.
    Siempre ≥ 0, igual a 0 solo cuando A = B.
    """
    order  = len(lpc_a)
    b_full = np.concatenate([[1.0], lpc_b])   # [1, b1, ..., bp]

    R   = toeplitz(acf_a[:order + 1])
    num = float(b_full @ R @ b_full)
    den = float(gain_a) + 1e-12

    ratio = num / den
    ratio = max(ratio, 1e-10)
    dist  = ratio - np.log(ratio) - 1.0
    return max(0.0, dist)


# ──────────────────────────────────────────────
# 9. Distorsión VQ de una grabación contra un codebook
# ──────────────────────────────────────────────
def vq_distortion(features, codebook_lsf, use_is=True):
    """
    Calcula la distorsión VQ media de una grabación contra un codebook.

    features     : dict retornado por extract_features()
    codebook_lsf : (K, lpc_order) centroides en espacio LSF
    use_is       : True → distancia Itakura-Saito; False → Euclidiana en LSF

    Retorna escalar (distorsión promedio por frame).
    """
    if use_is:
        codebook_lpc = np.array([lsf_to_lpc(c) for c in codebook_lsf])

    n_frames = len(features['lsf'])
    total    = 0.0

    for i in range(n_frames):
        if use_is:
            dists = np.array([
                itakura_saito_distance(
                    features['lpc'][i], features['gain'][i],
                    features['acf'][i], codebook_lpc[k])
                for k in range(len(codebook_lsf))
            ])
        else:
            diff  = codebook_lsf - features['lsf'][i]
            dists = np.sum(diff ** 2, axis=1)

        total += np.min(dists)

    return total / max(n_frames, 1)

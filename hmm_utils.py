"""
Utilidades para la extensión HMM del sistema de reconocimiento de voz.

Pipeline:
  señal → preénfasis → frames Hamming → FFT → banco Mel → log → DCT → MFCC
        → VQ global (256) → secuencia discreta → HMM Bakis → Forward log
"""

import numpy as np
from utils import apply_preemphasis, hamming_frames, FS, FRAME_LEN, HOP


# ─────────────────────────────────────────────────────────────────
# 1. Banco de filtros Mel
# ─────────────────────────────────────────────────────────────────
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(n_mels=26, n_fft=512, fs=FS):
    """
    Construye banco de filtros triangulares en escala Mel.
    Retorna matriz (n_mels, n_fft//2 + 1).
    """
    low_mel  = _hz_to_mel(0)
    high_mel = _hz_to_mel(fs / 2)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = _mel_to_hz(mel_pts)
    bins     = np.floor((n_fft + 1) * hz_pts / fs).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        if mid > lo:
            fbank[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fbank[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fbank


# ─────────────────────────────────────────────────────────────────
# 2. Extracción MFCC
# ─────────────────────────────────────────────────────────────────
def extract_mfcc(signal, fs=FS, n_mfcc=13, n_mels=26, n_fft=512,
                 frame_len=FRAME_LEN, hop=HOP):
    """
    Extrae MFCC de una señal de voz.

    Returns:
        array (n_frames, n_mfcc)  float32
    """
    signal = apply_preemphasis(signal)
    frames = hamming_frames(signal, frame_len, hop)  # (n_frames, frame_len)

    if len(frames) == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32)

    # Rellenar frames al tamaño n_fft
    n_frames = len(frames)
    padded = np.zeros((n_frames, n_fft), dtype=np.float32)
    copy_len = min(frame_len, n_fft)
    padded[:, :copy_len] = frames[:, :copy_len]

    # Espectro de potencia (solo mitad positiva)
    spectrum = np.abs(np.fft.rfft(padded, n=n_fft)) ** 2  # (n_frames, n_fft//2+1)

    # Banco Mel y log-energías
    fbank = mel_filterbank(n_mels, n_fft, fs)          # (n_mels, bins)
    mel_e = spectrum @ fbank.T                          # (n_frames, n_mels)
    mel_e = np.maximum(mel_e, 1e-10)
    log_mel = np.log(mel_e)                             # (n_frames, n_mels)

    # DCT tipo-II → MFCCs
    j = np.arange(n_mels)
    k = np.arange(n_mfcc).reshape(-1, 1)
    dct = np.cos(np.pi * k * (2 * j + 1) / (2 * n_mels))  # (n_mfcc, n_mels)
    mfcc = log_mel @ dct.T                              # (n_frames, n_mfcc)

    return mfcc.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# 3. Cuantización vectorial — asignar índice centroide más cercano
# ─────────────────────────────────────────────────────────────────
def quantize(mfcc_seq, centroids):
    """
    Cuantiza una secuencia MFCC usando el codebook global.

    Args:
        mfcc_seq:  (T, D) secuencia de vectores MFCC
        centroids: (K, D) centroides del codebook global

    Returns:
        (T,) array de índices enteros en [0, K-1]
    """
    diff = mfcc_seq[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (T, K, D)
    dists = np.sum(diff ** 2, axis=2)                                  # (T, K)
    return np.argmin(dists, axis=1).astype(np.int32)


# ─────────────────────────────────────────────────────────────────
# 4. HMM Bakis (Left-to-Right) con ingeniería de conteos
# ─────────────────────────────────────────────────────────────────
class HMMBakis:
    """
    HMM Left-to-Right (Bakis) entrenado con ingeniería de conteos.

    Args:
        n_states:  número de estados ocultos (recomendado 5-8)
        n_symbols: tamaño del alfabeto VQ (256)
    """

    def __init__(self, n_states: int = 6, n_symbols: int = 256):
        self.N = n_states
        self.M = n_symbols
        self.A  = None  # (N, N) — matriz de transición
        self.B  = None  # (N, M) — matriz de emisión
        self.pi = None  # (N,)   — distribución inicial

    # ── Entrenamiento ────────────────────────────────────────────

    def _segment(self, T: int) -> np.ndarray:
        """Segmentación lineal: asigna a cada frame t el estado correspondiente."""
        boundaries = np.linspace(0, T, self.N + 1).astype(int)
        seg = np.zeros(T, dtype=int)
        for i in range(self.N):
            seg[boundaries[i]:boundaries[i + 1]] = i
        return seg

    def train(self, sequences: list) -> None:
        """
        Entrenamiento por ingeniería de conteos.

        Args:
            sequences: lista de arrays 1-D de índices VQ
        """
        B_counts = np.zeros((self.N, self.M))
        A_self   = np.zeros(self.N)       # tiempo en self-loop por estado
        A_next   = np.zeros(self.N - 1)   # número de avances al siguiente

        for seq in sequences:
            T = len(seq)
            if T < self.N:
                continue
            seg = self._segment(T)

            # Acumular emisiones
            for t, obs in enumerate(seq):
                B_counts[seg[t], int(obs)] += 1

            # Acumular transiciones por duración de segmento
            for i in range(self.N):
                cnt = int(np.sum(seg == i))
                if cnt > 0:
                    A_self[i] += max(cnt - 1, 0)
                    if i < self.N - 1:
                        A_next[i] += 1

        # Matriz B con smoothing ε=1e-6 y renormalización
        B_counts += 1e-6
        self.B = B_counts / B_counts.sum(axis=1, keepdims=True)

        # Matriz A (Bakis: sólo diagonal y superdiagonal)
        self.A = np.zeros((self.N, self.N))
        for i in range(self.N):
            total = A_self[i] + (A_next[i] if i < self.N - 1 else 0)
            if total < 1e-10:
                # Sin datos: usar probabilidad por defecto
                self.A[i, i] = 0.85
                if i < self.N - 1:
                    self.A[i, i + 1] = 0.15
            else:
                self.A[i, i] = A_self[i] / total
                if i < self.N - 1:
                    self.A[i, i + 1] = A_next[i] / total
        # El estado final es absorbente
        self.A[-1, :] = 0.0
        self.A[-1, -1] = 1.0

        # π: solo el estado 0 activo al inicio
        self.pi = np.zeros(self.N)
        self.pi[0] = 1.0

    # ── Algoritmo Forward en log-espacio ─────────────────────────

    def log_likelihood(self, sequence: np.ndarray) -> float:
        """
        Calcula log P(O | λ) usando el algoritmo Forward en log-espacio.

        Args:
            sequence: array 1-D de índices VQ

        Returns:
            log-verosimilitud (float, puede ser -inf si el modelo no está entrenado)
        """
        if self.A is None:
            return -np.inf

        T = len(sequence)
        if T == 0:
            return -np.inf

        log_A  = np.log(np.maximum(self.A, 1e-300))   # (N, N)
        log_B  = np.log(np.maximum(self.B, 1e-300))   # (N, M)
        log_pi = np.log(np.maximum(self.pi, 1e-300))  # (N,)

        # Inicialización: log α_0(i) = log π_i + log B(i, O_0)
        log_alpha = log_pi + log_B[:, sequence[0]]    # (N,)

        # Recursión Forward
        for t in range(1, T):
            # log_alpha_t(j) = logsumexp_i[log_alpha_{t-1}(i) + log A(i,j)] + log B(j, O_t)
            scores = log_alpha[:, np.newaxis] + log_A  # (N, N)
            log_alpha = _logsumexp(scores, axis=0) + log_B[:, sequence[t]]

        return float(_logsumexp(log_alpha))

    # ── Persistencia ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez(path, A=self.A, B=self.B, pi=self.pi,
                 n_states=np.array(self.N), n_symbols=np.array(self.M))

    @classmethod
    def load(cls, path: str) -> 'HMMBakis':
        data = np.load(path)
        hmm = cls(int(data['n_states']), int(data['n_symbols']))
        hmm.A  = data['A']
        hmm.B  = data['B']
        hmm.pi = data['pi']
        return hmm


# ─────────────────────────────────────────────────────────────────
# Auxiliar: logsumexp numéricamente estable
# ─────────────────────────────────────────────────────────────────
def _logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis) + 1e-300)
    if axis is not None:
        out += np.squeeze(a_max, axis=axis)
    else:
        out += a_max.item()
    return out

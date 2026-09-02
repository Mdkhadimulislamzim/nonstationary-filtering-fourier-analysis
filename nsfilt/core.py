"""
core.py -- Operators of

    
    "Fourier analysis of nonstationary filtering via cyclic convolution
     and combination matrices."

Everything here is a direct transcription of the paper: Section 2
(masks, conv/comb, Fourier operators), Section 3 (rank-one factorisations,
Lemmas 2-4, Corollary 1) and Section 4 (conjugate symmetry, frequency
decomposition, Lemmas 5-6).

DFT convention.  The paper uses

    X(j) = sum_k x(k) exp(-2 pi i j k / N),

which is exactly numpy.fft.fft, so no rescaling is ever needed.

All indices are modulo N.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fourier_matrix", "flip_operator", "exp_k", "unit_vector",
    "conv_matrix", "comb_matrix", "conv_inverse", "comb_inverse",
    "frequency_response", "mask_from_frequency_response",
    "comb_rank_one_apply", "conv_rank_one_apply", "rank_one_frequency_response",
    "has_conjugate_symmetry", "frequency_component_mask",
    "frequency_selective_apply", "conv_apply_lowrank",
]


# ---------------------------------------------------------------------------
# Section 2.4 / 2.5 : Fourier operators
# ---------------------------------------------------------------------------

def fourier_matrix(N: int) -> np.ndarray:
    """Unitary Fourier matrix V = (1/sqrt(N)) [exp(2 pi i j k / N)].

    Satisfies DFT(x) = sqrt(N) V^* x and IDFT(x) = (1/sqrt(N)) V x.
    """
    k = np.arange(N)
    return np.exp(2j * np.pi * np.outer(k, k) / N) / np.sqrt(N)


def flip_operator(N: int) -> np.ndarray:
    """The operator J of Section 2.5:  x0 -> x0,  xk -> x_{N-k}."""
    J = np.zeros((N, N))
    for i in range(N):
        J[i, (-i) % N] = 1.0
    return J


def exp_k(N: int, k: int) -> np.ndarray:
    """exp_k(j) = exp(2 pi i j k / N)  (Section 4 / appendix)."""
    j = np.arange(N)
    return np.exp(2j * np.pi * j * k / N)


def unit_vector(N: int, k: int) -> np.ndarray:
    """e_k : all zeros except a one in position k."""
    e = np.zeros(N)
    e[k] = 1.0
    return e


# ---------------------------------------------------------------------------
# Section 2.1 / 2.2 : cyclic convolution and combination matrices
# ---------------------------------------------------------------------------

def conv_matrix(C: np.ndarray) -> np.ndarray:
    """conv(C)(i, j) = C(i - j, j)      -- equation (2.1).

    Realises  y(t) = sum_tau C(t - tau, tau) x(tau);  the mask acting at
    time tau is column tau of C (mask indexed by the INPUT position).
    """
    C = np.asarray(C)
    N = C.shape[0]
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return C[(i - j) % N, j]


def comb_matrix(C: np.ndarray) -> np.ndarray:
    """comb(C)(i, j) = C(i - j, i)      -- equation (2.2).

    Realises  y(t) = sum_tau C(t - tau, t) x(tau);  the mask is indexed by
    the OUTPUT position.
    """
    C = np.asarray(C)
    N = C.shape[0]
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return C[(i - j) % N, i]


def conv_inverse(M: np.ndarray) -> np.ndarray:
    """Recover the mask matrix:  C(i, j) = conv(C)(i + j, j)."""
    M = np.asarray(M)
    N = M.shape[0]
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return M[(i + j) % N, j]


def comb_inverse(M: np.ndarray) -> np.ndarray:
    """Recover the mask matrix:  C(i, j) = comb(C)(j, j - i)."""
    M = np.asarray(M)
    N = M.shape[0]
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return M[j, (j - i) % N]


# ---------------------------------------------------------------------------
# Section 2.5 : frequency response matrix, equation (2.4)
# ---------------------------------------------------------------------------

def frequency_response(C: np.ndarray) -> np.ndarray:
    """F = V^* C^T V^* = (1/N) DFT2(C^T)."""
    C = np.asarray(C, dtype=complex)
    N = C.shape[0]
    return np.fft.fft2(C.T) / N


def mask_from_frequency_response(F: np.ndarray) -> np.ndarray:
    """Inverse of `frequency_response`:  C = (IDFT2(N F))^T   (Lemma 5)."""
    F = np.asarray(F, dtype=complex)
    N = F.shape[0]
    return np.fft.ifft2(N * F).T


# ---------------------------------------------------------------------------
# Section 3 : rank-one masks  C = c d^*
# ---------------------------------------------------------------------------

def comb_rank_one_apply(c: np.ndarray, d: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Lemma 2:  comb(c d^*) x = conj(d) .* IDFT(DFT(c) .* DFT(x))."""
    return np.conj(d) * np.fft.ifft(np.fft.fft(c) * np.fft.fft(x))


def conv_rank_one_apply(c: np.ndarray, d: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Lemma 3:  conv(c d^*) x = IDFT(DFT(conj(d) .* x) .* DFT(c)).

    `x` may carry leading batch axes; the transform acts on the last axis.
    """
    lam = np.conj(d) * x
    return np.fft.ifft(np.fft.fft(lam, axis=-1) * np.fft.fft(c), axis=-1)


def rank_one_frequency_response(c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Lemma 4:  F = (1/N) DFT(conj(d)) DFT(c)^T."""
    N = c.shape[0]
    return np.outer(np.fft.fft(np.conj(d)), np.fft.fft(c)) / N


# ---------------------------------------------------------------------------
# Section 4.1 : Fourier conjugate symmetry
# ---------------------------------------------------------------------------

def has_conjugate_symmetry(F: np.ndarray, tol: float = 1e-9) -> bool:
    """F(i, j) = conj(F(-i, -j)) for all i, j."""
    F = np.asarray(F)
    N = F.shape[0]
    idx = (-np.arange(N)) % N
    return bool(np.allclose(F, np.conj(F[np.ix_(idx, idx)]), atol=tol))


# ---------------------------------------------------------------------------
# Section 4.2 : frequency decomposition of the mask matrix (Lemma 5)
# ---------------------------------------------------------------------------

def frequency_component_mask(F: np.ndarray, k: int) -> np.ndarray:
    """C_k = (IDFT2(N F_k))^T, where F_k keeps only rows k and N-k of F.

    Lemma 5 then gives, for a real mask matrix,

        k = 0            C_0 = IDFT(f_0) exp_0^T
        N = 2n, k = n    C_n = IDFT(f_n) exp_n^T
        otherwise        C_k = 2 Re( IDFT(f_k) exp_k^T ).
    """
    F = np.asarray(F, dtype=complex)
    N = F.shape[0]
    Fk = np.zeros_like(F)
    Fk[k % N] = F[k % N]
    if (k % N) != ((-k) % N):
        Fk[(-k) % N] = F[(-k) % N]
    return mask_from_frequency_response(Fk)


def frequency_selective_apply(F: np.ndarray, k: int, x: np.ndarray) -> np.ndarray:
    """Lemma 6: conv(C_k) x, evaluated with FFTs only.

        k = 0            IDFT( DFT(x) .* f_0 )
        N = 2n, k = n    IDFT( DFT(x)_n .* f_n )
        otherwise        2 Re( IDFT( DFT(x)_k .* f_k ) )

    where DFT(x)_k(j) = DFT(x)(j - k), i.e. a circular shift by k.
    """
    F = np.asarray(F, dtype=complex)
    N = F.shape[0]
    k = k % N
    fk = F[k]
    X = np.fft.fft(x, axis=-1)
    Xk = np.roll(X, k, axis=-1)
    if k == 0:
        return np.fft.ifft(X * fk, axis=-1)
    if 2 * k == N:
        return np.fft.ifft(Xk * fk, axis=-1)
    return 2.0 * np.real(np.fft.ifft(Xk * fk, axis=-1))


# ---------------------------------------------------------------------------
# Low-rank application  (Section 3, "sums of rank-one terms")
# ---------------------------------------------------------------------------

def conv_apply_lowrank(C: np.ndarray, x: np.ndarray, rank: int | None = None):
    """Apply conv(C) to the rows of `x` through a rank-R SVD of C.

    C ~ sum_{r<R} sigma_r u_r v_r^*  is a sum of R rank-one masks, so by
    Lemma 3 the filter costs R forward/inverse FFT pairs instead of an
    N x N product.  This is the paper's own remark in Section 3 that
    arbitrary masks are approximated by sums of rank-one terms.

    Returns (y, singular_values).
    """
    C = np.asarray(C, dtype=complex)
    U, S, Vh = np.linalg.svd(C)
    R = len(S) if rank is None else min(rank, len(S))
    y = np.zeros(np.broadcast_shapes(x.shape, (C.shape[0],)), dtype=complex)
    for r in range(R):
        c = U[:, r] * S[r]
        d = np.conj(Vh[r, :])          # so that c d^* = sigma_r u_r v_r^*
        y = y + conv_rank_one_apply(c, d, x)
    return y, S

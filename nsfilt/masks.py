"""
masks.py -- the mask matrices used in Section 5.

Section 5.2 of the paper asks for a *spatially varying mask matrix* and for
the comparison with the stationary case. Approach 3, writes the
nonstationary filter as

    C = [c_0, c_1, ..., c_{N-1}]          (general, primary form)

or, optionally, as the rank-one  C = c d^*  of Section 3.  Both are provided.

-------------------------------------------------------------------------
Why the width is indexed by the OUTPUT position
-------------------------------------------------------------------------
A denoiser must leave a constant row unchanged, i.e. conv(C) 1 = 1, or it
merely rescales image intensity and the classification comparison is
confounded.  That condition is a statement about the *anti-diagonal* sums
of C:

    (conv(C) 1)(t) = sum_tau C(t - tau, tau).

Simply putting a unit-mass Gaussian of width sigma(tau) into column tau
makes the COLUMN sums of C equal to one, which preserves total intensity
sum_t y(t) = sum_t x(t) but does NOT give unit gain at every position.
Instead we use the mask

    C(u, tau) = g_{sigma(tau + u)}(u),

for which conv(C)(i, j) = g_{sigma(i)}(i - j): row i of the operator is a
unit-mass Gaussian of width sigma(i) centred at i.  Row sums are exactly
one, so conv(C) 1 = 1 holds identically for every profile sigma.

Setting sigma(i) = sigma constant gives C = g_sigma 1^T, which is exactly
the stationary mask matrix of Section 2.3 and makes conv(C) circulant.
The stationary baseline is therefore *nested* inside the nonstationary
model at zero modulation amplitude.
"""

from __future__ import annotations

import numpy as np

from .core import conv_matrix, frequency_response

__all__ = [
    "circular_distance", "gaussian_mask",
    "stationary_mask_matrix", "sigma_profile", "nonstationary_mask_matrix",
    "blend_mask_matrix", "blend_rank_one_terms",
    "rank_one_mask_matrix", "mask_report",
]


def circular_distance(N: int) -> np.ndarray:
    """Signed circular offset of each index from 0: 0, 1, ..., -2, -1."""
    i = np.arange(N)
    return ((i + N // 2) % N) - N // 2


def gaussian_mask(N: int, sigma: float) -> np.ndarray:
    """Unit-mass circular Gaussian mask centred at index 0.

    sigma <= 0 returns the identity mask (delta at 0), i.e. no smoothing.
    """
    if sigma <= 0:
        g = np.zeros(N)
        g[0] = 1.0
        return g
    u = circular_distance(N).astype(float)
    g = np.exp(-0.5 * (u / sigma) ** 2)
    return g / g.sum()


# ---------------------------------------------------------------------------
# Approach 2 : classical stationary Gaussian filter
# ---------------------------------------------------------------------------

def stationary_mask_matrix(N: int, sigma: float) -> np.ndarray:
    """C = g_sigma 1^T   (Section 2.3).  conv(C) is circulant."""
    g = gaussian_mask(N, sigma)
    return np.outer(g, np.ones(N))


# ---------------------------------------------------------------------------
# Approach 3 : proposed nonstationary filter
# ---------------------------------------------------------------------------

def sigma_profile(N: int, sigma_mid: float, sigma_amp: float, k: int = 1,
                  phase: float = 0.0, sigma_min: float = 0.0) -> np.ndarray:
    """Spatially varying width  sigma(i) = sigma_mid + sigma_amp cos(2 pi k i / N + phase).

    k is the spatial modulation frequency of the filter along the row.  It is
    the parameter that controls how far the frequency response couples modes:
    output mode i draws on input modes i and i +/- k (see `mask_report`).
    """
    i = np.arange(N)
    s = sigma_mid + sigma_amp * np.cos(2 * np.pi * k * i / N + phase)
    return np.clip(s, sigma_min, None)


def nonstationary_mask_matrix(N: int, sigma: np.ndarray) -> np.ndarray:
    """C(u, tau) = g_{sigma(tau + u)}(u) for a per-position width profile.

    Guarantees conv(C) 1 = 1 exactly (unit gain at every position) while the
    smoothing width genuinely varies along the row.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (N,):
        raise ValueError(f"sigma must have shape ({N},), got {sigma.shape}")
    # Row i of conv(C) is the Gaussian of width sigma(i) centred at i.
    G = np.stack([gaussian_mask(N, s) for s in sigma])       # G[i] = g_{sigma(i)}
    u, tau = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return G[(tau + u) % N, u]


def blend_mask_matrix(N: int, sigma_1: float, sigma_2: float,
                      a: float = 0.5, b: float = 0.5, k: int = 1,
                      phase: float = 0.0) -> np.ndarray:
    """Two-Gaussian blend, output-indexed:  C(u, tau) = g2(u) + w(tau+u)(g1(u) - g2(u)),

    with  w(i) = a + b cos(2 pi k i / N).  Then

        conv(C)(i, j) = g2(i-j) + w(i)(g1(i-j) - g2(i-j)),

    i.e. row i of the operator is the convex blend  w(i) g1 + (1 - w(i)) g2
    of two unit-mass Gaussians.  Two consequences, both wanted in Section 5.2:

      * row sums are exactly one, so conv(C) 1 = 1 for every profile;
      * the tau-dependence enters only through cos(2 pi k(tau+u)/N), whose
        Fourier content in tau is exactly {0, k, N-k}.  So F has exactly
        three nonzero rows, and by comb(F)(i, j) = F(i - j, i) the operator
        couples output mode i to input modes i and i +/- k and to nothing
        else.  b = 0 collapses comb(F) to a diagonal and recovers the
        circulant case of Section 2.3.

    Require |b| <= min(a, 1-a) to keep w(i) in [0, 1].
    """
    if abs(b) > min(a, 1.0 - a) + 1e-12:
        raise ValueError("need |b| <= min(a, 1-a) so that w(i) stays in [0, 1]")
    g1 = gaussian_mask(N, sigma_1)
    g2 = gaussian_mask(N, sigma_2)
    i = np.arange(N)
    w = a + b * np.cos(2 * np.pi * k * i / N + phase)
    u, tau = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    W = w[(tau + u) % N]
    return g2[u] + W * (g1 - g2)[u]


def blend_rank_one_terms(N: int, sigma_1: float, sigma_2: float,
                         a: float = 0.5, b: float = 0.5, k: int = 1,
                         phase: float = 0.0):
    """Exact decomposition of `blend_mask_matrix` into three rank-one masks.

    Writing w(i) = a + b cos(2 pi k i / N + phase) as

        w(i) = a + (b/2) e^{i phi} exp_k(i) + (b/2) e^{-i phi} conj(exp_k)(i),

    and Delta = g1 - g2, the mask C(u, tau) = g2(u) + w(tau+u) Delta(u) becomes

        C =  c0 d0^*  +  c+ d+^*  +  c- d-^*

        c0 = a g1 + (1-a) g2               d0 = 1
        c+ = (b/2) e^{i phi} Delta .* exp_k       d+ = conj(exp_k)
        c- = conj(c+)                       d- = exp_k

    So the two descriptions of Approach 3 -- the general column form
    C = [c_0, ..., c_{N-1}] and the rank-one form C = c d^* of Section 3 --
    are the same object here: the general mask *is* a sum of exactly three
    Section-3 rank-one masks whose modulation vectors d are pure Fourier
    modes.  Stage 3's mandated formula

        conv(C) x = IDFT[ DFT(conj(d) .* x) .* DFT(c) ]                (Lemma 3)

    is therefore applied verbatim, three times.  Term 0 is the stationary
    part; terms +/- k carry the spatial modulation and are complex conjugates
    of each other, so their sum is real.

    Returns [(c0, d0), (c_plus, d_plus), (c_minus, d_minus)].
    """
    g1 = gaussian_mask(N, sigma_1)
    g2 = gaussian_mask(N, sigma_2)
    D = g1 - g2
    ek = np.exp(2j * np.pi * k * np.arange(N) / N)
    ph = np.exp(1j * phase)
    c_plus = (b / 2.0) * ph * D * ek
    return [
        (a * g1 + (1.0 - a) * g2, np.ones(N)),
        (c_plus, np.conj(ek)),
        (np.conj(c_plus), ek),
    ]


def rank_one_mask_matrix(c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """C = c d^*  (Section 3). 
    
    Note this is a gain modulation conj(d) followed by a fixed convolution
    with c: every column of C is the same shape scaled by a scalar.  It is
    nonstationary in the sense of the paper but cannot vary the filter width,
    and conv(C) 1 = c (*) conj(d) is not identically one unless d is constant.
    Reported as an ablation, not as the primary Approach 3.
    """
    return np.outer(np.asarray(c), np.conj(np.asarray(d)))


# ---------------------------------------------------------------------------
# Diagnostics tying Section 5.2 to Sections 3 and 4
# ---------------------------------------------------------------------------

def mask_report(C: np.ndarray, coupling_tol: float = 1e-8) -> dict:
    """Structural summary of a mask matrix, for Section 5.2 and Stage 6.

    Returns the DC-gain deviation, the singular values of C (how many
    rank-one terms of Section 3 are needed), and which rows of the frequency
    response F are active -- i.e. which Fourier modes the operator couples.
    """
    C = np.asarray(C)
    N = C.shape[0]
    M = conv_matrix(C)
    F = frequency_response(C)
    row_energy = np.abs(F).max(axis=1)
    active = np.flatnonzero(row_energy > coupling_tol * max(row_energy.max(), 1e-300))
    S = np.linalg.svd(C, compute_uv=False)
    return {
        "N": N,
        "dc_gain_max_error": float(np.abs(M.sum(axis=1) - 1.0).max()),
        "column_sum_max_error": float(np.abs(C.sum(axis=0) - 1.0).max()),
        "singular_values": S,
        "numerical_rank": int(np.sum(S > 1e-10 * S[0])),
        "energy_rank_99": int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), 0.99) + 1),
        "active_frequency_rows": active,
        "is_circulant": bool(np.allclose(C, np.outer(C[:, 0], np.ones(N)))),
    }

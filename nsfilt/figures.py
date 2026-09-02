"""
figures.py -- Section 5.2.

Section 5.2 requires the relation between spatial variation of the filter and
the corresponding frequency-domain coupling; Stage 6 asks for at least one
example showing the mask matrix C, the frequency-response matrix F,
stationary versus nonstationary coupling, selected frequency-component
contributions, and the reconstructed sum of the components.

The point this makes: with sigma(i) = mid + amp cos(2 pi k i / N), the mask
excites essentially only rows 0, k and N-k of F, so by comb(F)(i, j) =
F(i - j, i) the operator couples output mode i to input modes i and i +/- k
and to nothing else.  Setting amp = 0 collapses comb(F) to a diagonal and
recovers the classical circulant case of Section 2.3.  The stationary
baseline is the zero-modulation limit of the proposed filter.
"""

from __future__ import annotations

import numpy as np

from .core import (conv_matrix, comb_matrix, frequency_response,
                   frequency_component_mask, frequency_selective_apply)
from .masks import stationary_mask_matrix, sigma_profile, nonstationary_mask_matrix

__all__ = ["figure_mask_and_response", "figure_frequency_components",
           "figure_denoising_examples", "cost_table"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figure_mask_and_response(N=64, sigma_mid=2.0, sigma_amp=1.0, k=2,
                             path="fig_mask_response.pdf"):
    """C, conv(C), |F| and |comb(F)| for the stationary and nonstationary masks."""
    plt = _plt()
    Cs = stationary_mask_matrix(N, sigma_mid)
    s = sigma_profile(N, sigma_mid, sigma_amp, k)
    Cn = nonstationary_mask_matrix(N, s)

    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    for row, (C, name) in enumerate([(Cs, "stationary"), (Cn, "nonstationary")]):
        F = frequency_response(C)
        panels = [
            (np.real(C), f"mask $C$ ({name})"),
            (np.real(conv_matrix(C)), r"$\mathrm{conv}(C)$"),
            (np.log10(np.abs(F) + 1e-12), r"$\log_{10}|F|$"),
            (np.log10(np.abs(comb_matrix(F)) + 1e-12), r"$\log_{10}|\mathrm{comb}(F)|$"),
        ]
        for col, (M, title) in enumerate(panels):
            im = ax[row, col].imshow(M, cmap="viridis", aspect="equal")
            ax[row, col].set_title(title, fontsize=10)
            ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
            fig.colorbar(im, ax=ax[row, col], fraction=0.046)
    fig.suptitle(rf"$N={N}$, $\sigma(i)={sigma_mid}+{sigma_amp}\cos(2\pi\cdot{k}i/N)$")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def figure_frequency_components(N=64, sigma_mid=2.0, sigma_amp=1.0, k=2,
                                path="fig_components.pdf", seed=0):
    """Lemma 5 / Lemma 6: individual C_k contributions and their sum."""
    plt = _plt()
    s = sigma_profile(N, sigma_mid, sigma_amp, k)
    C = nonstationary_mask_matrix(N, s)
    F = frequency_response(C)

    energy = np.abs(F).max(axis=1)
    order = [j for j in range(N // 2 + 1)
             if max(energy[j], energy[(-j) % N]) > 1e-10 * energy.max()]

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    exact = np.real(conv_matrix(C) @ x)

    parts, running = [], np.zeros(N)
    for kk in order:
        y = np.real(frequency_selective_apply(F, kk, x.astype(complex)))
        parts.append((kk, y))
        running = running + y

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for kk, y in parts:
        ax[0].plot(y, lw=1.0, label=rf"$\mathrm{{conv}}(C_{{{kk}}})x$")
    ax[0].set_title("frequency-selective contributions (Lemma 6)")
    ax[0].legend(fontsize=8); ax[0].set_xlabel("position")

    ax[1].plot(exact, lw=2.5, alpha=0.5, label=r"$\mathrm{conv}(C)x$")
    ax[1].plot(running, "--", lw=1.2, label=r"$\sum_k \mathrm{conv}(C_k)x$")
    ax[1].set_title(f"reconstruction, max error {np.abs(exact - running).max():.2e}")
    ax[1].legend(fontsize=8); ax[1].set_xlabel("position")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path, float(np.abs(exact - running).max())


def figure_denoising_examples(clean, noisy, stat_op, nonstat_op, n=4,
                              path="fig_denoising.pdf"):
    """Visual panel: clean / noisy / stationary / nonstationary."""
    from .filtering import filter_images
    plt = _plt()
    xs = filter_images(noisy[:n], stat_op)
    xn = filter_images(noisy[:n], nonstat_op)
    cols = [("clean", clean[:n]), ("noisy", noisy[:n]),
            ("stationary Gaussian", xs), ("nonstationary", xn)]
    fig, ax = plt.subplots(n, 4, figsize=(9, 2.3 * n))
    ax = np.atleast_2d(ax)
    for j, (name, imgs) in enumerate(cols):
        for i in range(n):
            ax[i, j].imshow(np.clip(imgs[i], 0, 1), cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def cost_table(N=224, n_rows=4096, ranks=(1, 2, 4, 8, None), sigma_mid=2.0,
               sigma_amp=1.0, k=1, seed=0):
    """Accuracy/cost of dense conv(C) versus rank-R FFT and frequency-selective.

    Section 6 argues the low-rank / FFT representations point toward
    computationally efficient convolutional layers; this produces the numbers
    for that claim.
    """
    from .filtering import FilterOperator
    s = sigma_profile(N, sigma_mid, sigma_amp, k)
    C = nonstationary_mask_matrix(N, s)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, N)).astype(np.float32)

    dense = FilterOperator(C, method="dense")
    ref = dense.apply_rows(X)
    rows = [{"method": "dense conv(C)", "rank": "-",
             "time_s": dense.timeit(X), "rel_err": 0.0}]
    for R in ranks:
        op = FilterOperator(C, method="lowrank", rank=R)
        y = op.apply_rows(X)
        rows.append({"method": "rank-R (Lemma 3)", "rank": R or "full",
                     "time_s": op.timeit(X),
                     "rel_err": float(np.linalg.norm(y - ref) / np.linalg.norm(ref))})
    fop = FilterOperator(C, method="freq")
    yf = fop.apply_rows(X)
    rows.append({"method": "frequency-selective (Lemma 6)",
                 "rank": f"{len(fop.active_k)} rows",
                 "time_s": fop.timeit(X),
                 "rel_err": float(np.linalg.norm(yf - ref) / np.linalg.norm(ref))})
    return rows

"""
filtering.py -- row-wise application of the operators to images.

Section 5.1: "the filtering framework is applied along image rows, treating
each row as a one-dimensional signal".

    x_i^stat    = conv(C_stat) x_i
    x_i^nonstat = conv(C)      x_i

and the filtered rows are stacked back into a complete image.  No padding,
no vertical pass, no two-dimensional operator: Section 5 forbids additional
preprocessing, so the cyclic wrap of the last pixel into the first is left
intact and simply reported.

Three evaluation paths, all mathematically identical:

  "dense"    y = X conv(C)^T                        (one BLAS matmul)
  "lowrank"  R rank-one terms via Lemma 3           (R FFT pairs)
  "freq"     nonzero rows of F via Lemma 6          (frequency-selective)

`dense` is the fastest in practice for N <= 256 and is the default; the other
two exist to demonstrate Sections 3 and 4 and to produce the cost comparison
that Section 6 motivates.
"""

from __future__ import annotations

import time

import numpy as np

from .core import (conv_matrix, conv_apply_lowrank, frequency_response,
                   frequency_selective_apply)

__all__ = ["FilterOperator", "filter_images"]


class FilterOperator:
    """A row filter defined by a mask matrix C."""

    def __init__(self, C: np.ndarray, method: str = "dense", rank: int | None = None,
                 freq_tol: float = 1e-10):
        self.C = np.asarray(C)
        self.N = self.C.shape[0]
        self.method = method
        self.rank = rank

        if method == "dense":
            M = conv_matrix(self.C)
            self.M = np.real_if_close(M).astype(np.float32)
        elif method == "lowrank":
            self.M = None
        elif method == "freq":
            F = frequency_response(self.C)
            energy = np.abs(F).max(axis=1)
            keep = energy > freq_tol * max(energy.max(), 1e-300)
            # Lemma 6 pairs row k with row N-k, so iterate k = 0..floor(N/2).
            self.F = F
            self.active_k = [k for k in range(self.N // 2 + 1)
                             if keep[k] or keep[(-k) % self.N]]
        else:
            raise ValueError(f"unknown method {method!r}")

    # -- application ------------------------------------------------------
    def apply_rows(self, X: np.ndarray) -> np.ndarray:
        """Filter every row of X (..., N) along the last axis."""
        X = np.asarray(X, dtype=np.float32)
        if self.method == "dense":
            return X @ self.M.T
        if self.method == "lowrank":
            y, _ = conv_apply_lowrank(self.C, X.astype(np.complex128), rank=self.rank)
            return np.real(y).astype(np.float32)
        y = np.zeros(X.shape, dtype=np.complex128)
        for k in self.active_k:
            y += frequency_selective_apply(self.F, k, X.astype(np.complex128))
        return np.real(y).astype(np.float32)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        return filter_images(images, self)

    # -- diagnostics ------------------------------------------------------
    def dc_gain_error(self) -> float:
        M = conv_matrix(self.C)
        return float(np.abs(np.real(M).sum(axis=1) - 1.0).max())

    def timeit(self, X: np.ndarray, repeats: int = 3) -> float:
        """Median wall time (seconds) for one application, for the cost table."""
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.apply_rows(X)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))


def filter_images(images: np.ndarray, op: "FilterOperator") -> np.ndarray:
    """Apply a row filter to a stack of images.

    Accepts (H, W), (n, H, W) or (n, H, W, c); the operator acts on the last
    axis, so colour channels are filtered independently by the unchanged
    one-dimensional operator (PathMNIST).
    """
    x = np.asarray(images, dtype=np.float32)
    if x.ndim == 4:                      # (n, H, W, c) -> filter along W
        x = np.moveaxis(x, -1, 1)        # (n, c, H, W)
        y = op.apply_rows(x)
        return np.moveaxis(y, 1, -1)
    if x.shape[-1] != op.N:
        raise ValueError(f"row length {x.shape[-1]} != operator size {op.N}")
    return op.apply_rows(x)

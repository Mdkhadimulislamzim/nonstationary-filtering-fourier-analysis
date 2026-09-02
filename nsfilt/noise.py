"""
noise.py -- the synthetic degradations of Section 5.5.

Section 5.5 names three families: "Gaussian noise as well as selected
non-Gaussian perturbations, such as heavy-tailed and asymmetric noise",
each at several levels.  Implemented here on float images in [0, 1].

  gaussian      additive N(0, sigma^2)
  contaminated  eps-contaminated normal, (1-eps) N(0, s^2) + eps N(0, (rs)^2)
  student_t     additive Student-t, scaled to the requested sigma
  speckle       multiplicative gamma speckle (the physically correct
                degradation for OCT, and asymmetric)
  exponential   additive centred exponential (asymmetric, light-tailed)

`level` is the nominal standard deviation of the perturbation in intensity
units, so the three families are comparable at equal level.
"""

from __future__ import annotations

import numpy as np

__all__ = ["NOISE_MODELS", "add_noise"]


def _gaussian(x, level, rng, **kw):
    return x + rng.normal(0.0, level, size=x.shape)


def _contaminated(x, level, rng, eps=0.10, ratio=5.0, **kw):
    """(1-eps) N(0, s^2) + eps N(0, (ratio s)^2), variance matched to level^2."""
    s = level / np.sqrt(1.0 - eps + eps * ratio ** 2)
    scale = np.where(rng.random(x.shape) < eps, ratio * s, s)
    return x + rng.normal(0.0, 1.0, size=x.shape) * scale


def _student_t(x, level, rng, df=3.0, **kw):
    if df <= 2:
        raise ValueError("df must exceed 2 for finite variance")
    z = rng.standard_t(df, size=x.shape)
    return x + z * level / np.sqrt(df / (df - 2.0))


def _speckle(x, level, rng, **kw):
    """Multiplicative gamma speckle with unit mean and variance level^2."""
    if level <= 0:
        return x.copy()
    shape = 1.0 / (level ** 2)
    return x * rng.gamma(shape, 1.0 / shape, size=x.shape)


def _exponential(x, level, rng, **kw):
    """Centred exponential: strongly right-skewed, standard deviation = level."""
    return x + (rng.exponential(level, size=x.shape) - level)


NOISE_MODELS = {
    "gaussian": _gaussian,
    "contaminated": _contaminated,
    "student_t": _student_t,
    "speckle": _speckle,
    "exponential": _exponential,
}

# For Section 5.5's "several noise levels".
DEFAULT_LEVELS = (0.05, 0.10, 0.20)


def add_noise(images: np.ndarray, model: str, level: float,
              seed: int = 0, clip: bool = True, **kwargs) -> np.ndarray:
    """Corrupt float images in [0, 1].  Deterministic given `seed`.

    The same seed must be used across the three denoising approaches so that
    each of them sees an identical noisy image (PDF-3, Stage 2).
    """
    if model not in NOISE_MODELS:
        raise KeyError(f"unknown noise model {model!r}; have {sorted(NOISE_MODELS)}")
    if model == "none" or level == 0:
        return np.asarray(images, dtype=np.float32)
    rng = np.random.default_rng(seed)
    x = np.asarray(images, dtype=np.float64)
    y = NOISE_MODELS[model](x, float(level), rng, **kwargs)
    if clip:
        y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32)


NOISE_MODELS["none"] = lambda x, level, rng, **kw: x.copy()

"""
pipeline.py -- on-the-fly noise + filtering.

Why this module exists.  Materialising OCTMNIST-224 as float32 costs

    (97,477 + 10,832 + 1,000) x 224 x 224 x 4 bytes  =  21.9 GB

per copy, and the experiment needs a noisy copy and a filtered copy on top of
the original.  PathMNIST-224 is 64.5 GB.  So images are held as uint8 (memory
mapped where possible) and the noise and the row filter are applied per item
inside the Dataset, in the worker processes.

Cost per image: one (H, N) x (N, N) matmul, about 11 MFLOP at N = 224 --
negligible next to a ResNet-18 forward/backward pass, and it overlaps with GPU
compute through `num_workers`.

Determinism.  The noise seed for item i of a split is `noise_seed_base + i`,
and `noise_seed_base` depends only on (noise_model, noise_level, split).  All
three approaches therefore see byte-identical noisy images, which is what
Stage 2 of the protocol requires, without ever storing them.
"""

from __future__ import annotations

import numpy as np

from .noise import add_noise

__all__ = ["noise_seed_base", "FilteredDataset", "materialise"]


def noise_seed_base(noise_model: str, noise_level: float, split: str) -> int:
    """Stable across processes and runs (unlike Python's salted hash())."""
    import zlib
    key = f"{noise_model}|{noise_level:.6f}|{split}".encode()
    return int(zlib.crc32(key)) % 1_000_000


class FilteredDataset:
    """uint8 images -> [0,1] float -> optional noise -> optional row filter.

    Parameters
    ----------
    images   : (n, H, W) or (n, H, W, C) uint8 array, may be a memmap
    labels   : (n,) int64
    op       : FilterOperator or None (None = Approach 1, no denoising)
    """

    def __init__(self, images, labels, noise_model="none", noise_level=0.0,
                 split="train", op=None, as_rgb=True, normalise=True):
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.noise_model = noise_model
        self.noise_level = float(noise_level)
        self.op = op
        self.as_rgb = as_rgb
        self.normalise = normalise
        self.base = noise_seed_base(noise_model, noise_level, split)

    def __len__(self):
        return len(self.labels)

    def _prepare(self, i):
        x = np.asarray(self.images[i], dtype=np.float32) / 255.0
        if self.noise_model != "none" and self.noise_level > 0:
            x = add_noise(x, self.noise_model, self.noise_level, seed=self.base + i)
        if self.op is not None:
            x = self.op.apply_rows(x) if x.ndim == 2 else \
                np.moveaxis(self.op.apply_rows(np.moveaxis(x, -1, 0)), 0, -1)
        return x

    def __getitem__(self, i):
        import torch
        x = self._prepare(i)
        if x.ndim == 2:
            x = x[None, :, :]
        else:
            x = np.moveaxis(x, -1, 0)
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        if self.as_rgb and t.shape[0] == 1:
            t = t.expand(3, -1, -1)
        if self.normalise:
            t = (t - 0.5) / 0.5
        return t, int(self.labels[i])

    # -- helpers ---------------------------------------------------------
    def clean_float(self, i):
        """The undegraded image, for PSNR/SSIM references."""
        return np.asarray(self.images[i], dtype=np.float32) / 255.0

    def processed(self, idx):
        """Stack of processed images for a small index list (figures, PSNR)."""
        return np.stack([self._prepare(i) for i in idx])


def materialise(ds: FilteredDataset, idx=None, max_items=2000):
    """Realise a bounded subset as a float32 array (test-set metrics, figures).

    Guards against accidentally expanding a 97k-image training split.
    """
    idx = np.arange(len(ds)) if idx is None else np.asarray(idx)
    if len(idx) > max_items:
        raise ValueError(f"refusing to materialise {len(idx)} images; "
                         f"raise max_items deliberately if you have the RAM")
    return ds.processed(idx)

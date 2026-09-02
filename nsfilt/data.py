"""
data.py -- MedMNIST v2 / MedMNIST+ loading (PDF-3, Stage 1).

Primary dataset: OCTMNIST.  Retinal optical coherence tomography, four
diagnostic classes, official split 97,477 / 10,832 / 1,000 (Yang et al.,
MedMNIST v2).  Grayscale, so the one-dimensional row operator applies with
no channel decision.  Secondary: PathMNIST (RGB, nine classes, external
test centre).

Official splits are used unchanged; the only preprocessing is the uint8 ->
[0, 1] float conversion required to filter, per Section 5's instruction that
no additional preprocessing be introduced.

`size` selects the MedMNIST+ resolution (28, 64, 128, 224).  Sizes above 28
require the MedMNIST+ release; `medmnist` >= 3.0 downloads them with the
same API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Split", "DatasetBundle", "load_medmnist", "subsample"]


@dataclass
class Split:
    images: np.ndarray          # (n, H, W) or (n, H, W, 3), uint8; may be a memmap
    labels: np.ndarray          # (n,) int64
    name: str

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class DatasetBundle:
    train: Split
    val: Split
    test: Split
    n_classes: int
    n_channels: int
    flag: str
    size: int

    @property
    def row_length(self) -> int:
        return self.train.images.shape[2]


def load_medmnist(flag: str = "octmnist", size: int = 224,
                  root: str | None = None, download: bool = True,
                  mmap: bool | None = None) -> DatasetBundle:
    """Load a MedMNIST subset with its official train/val/test split.

    Images stay uint8.  Converting OCTMNIST-224 to float32 would cost 21.9 GB
    (PathMNIST-224: 64.5 GB); the conversion happens per image in
    `pipeline.FilteredDataset` instead.

    `root=None` is NOT forwarded: medmnist raises if it receives an explicit
    None, so the default `~/.medmnist` is created and used.
    """
    import os
    import medmnist
    from medmnist import INFO

    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])

    if root is None:
        root = os.path.expanduser("~/.medmnist")
    os.makedirs(root, exist_ok=True)

    kw = dict(download=download, root=root)
    if size != 28:
        kw["size"] = size
    if mmap is None:
        mmap = size > 28
    if mmap:
        kw["mmap_mode"] = "r"

    splits = {}
    for split in ("train", "val", "test"):
        ds = DataClass(split=split, **kw)
        labels = np.asarray(ds.labels).reshape(-1).astype(np.int64)
        splits[split] = Split(images=ds.imgs, labels=labels, name=split)

    nc = info["n_channels"]
    n_channels = len(nc) if isinstance(nc, (list, tuple)) else int(nc)
    return DatasetBundle(
        train=splits["train"], val=splits["val"], test=splits["test"],
        n_classes=len(info["label"]), n_channels=n_channels,
        flag=flag, size=size,
    )


def subsample(split: Split, n: int, seed: int = 0, stratified: bool = True) -> Split:
    """Class-stratified subsample, for the validation parameter search.

    PDF-3 item 6 requires the filter parameters to be chosen on validation
    data and then fixed.  Running that search on the full 97k training set is
    wasteful; a stratified subset keeps the class proportions intact.
    """
    if n >= len(split):
        return split
    rng = np.random.default_rng(seed)
    if not stratified:
        idx = np.sort(rng.choice(len(split), size=n, replace=False))
    else:
        idx = []
        classes, counts = np.unique(split.labels, return_counts=True)
        quota = np.maximum(1, np.round(n * counts / counts.sum()).astype(int))
        for c, q in zip(classes, quota):
            pool = np.flatnonzero(split.labels == c)
            idx.append(rng.choice(pool, size=min(q, len(pool)), replace=False))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        idx = np.sort(idx[:n])          # sorted: memmap-friendly fancy indexing
    return Split(np.asarray(split.images[idx]), split.labels[idx],
                 f"{split.name}[{len(idx)}]")

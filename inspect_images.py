"""
inspect_images.py -- look at real OCT images before running anything expensive.

Purpose.  Before 81 training runs, confirm on actual data that:

  1. the dataset loads and the images look like OCT scans;
  2. the noise model produces a visible, plausible degradation;
  3. the stationary and nonstationary filters produce *visibly different*
     outputs.

Point 3 is the one that matters.  If Approach 2 and Approach 3 give nearly
identical images, no classifier difference is possible and the parameters must
change before anything else happens.  The script prints a separation figure for
exactly this: the RMS difference between the two filtered images relative to
the RMS difference between noisy and clean.  Below about 5% the two approaches
are effectively the same filter.

Run:
    python inspect_images.py
    python inspect_images.py --size 224 --noise speckle --level 0.15
    python inspect_images.py --b 0.5 --k 4 --sigma1 4.0 --sigma2 0.5

Writes  inspect_images.png  and prints diagnostics.  Nothing is trained.
"""

from __future__ import annotations

import argparse

import numpy as np

from nsfilt.data import load_medmnist
from nsfilt.noise import add_noise
from nsfilt.masks import (stationary_mask_matrix, blend_mask_matrix,
                          gaussian_mask, mask_report)
from nsfilt.filtering import FilterOperator, filter_images
from nsfilt.metrics import psnr, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="octmnist")
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--root", default=None)
    ap.add_argument("--noise", default="gaussian")
    ap.add_argument("--level", type=float, default=0.10)
    ap.add_argument("--n", type=int, default=6, help="images to display")
    ap.add_argument("--sigma-stat", type=float, default=1.5)
    ap.add_argument("--sigma1", type=float, default=3.0, help="wide kernel")
    ap.add_argument("--sigma2", type=float, default=0.8, help="narrow kernel")
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--b", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--phase", type=float, default=0.0,
                    help="radians; pi puts the NARROWEST kernel at the cyclic "
                         "wrap (row edge) instead of the widest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = (f"inspect_{args.dataset}_{args.size}_{args.noise}{args.level}"
                    f"_s{args.sigma1}-{args.sigma2}_b{args.b}_k{args.k}"
                    f"_ph{args.phase:.2f}.png")

    # ---- 1. load ------------------------------------------------------
    print(f"loading {args.dataset} at {args.size}px ...")
    b = load_medmnist(args.dataset, args.size, root=args.root)
    N = b.row_length
    print(f"  train {len(b.train)}  val {len(b.val)}  test {len(b.test)}")
    print(f"  classes {b.n_classes}  channels {b.n_channels}  dtype {b.test.images.dtype}")
    print(f"  labels present in test: {np.bincount(b.test.labels)}")

    # One image per class where possible, else the first n.
    idx = []
    for c in range(b.n_classes):
        hit = np.flatnonzero(b.test.labels == c)
        if len(hit):
            idx.append(hit[0])
    idx = (idx + list(range(len(b.test))))[:args.n]
    idx = np.array(idx)

    clean = np.stack([np.asarray(b.test.images[i], dtype=np.float32) / 255.0
                      for i in idx])
    print(f"  intensity range {clean.min():.3f} .. {clean.max():.3f}, "
          f"mean {clean.mean():.3f}")

    # ---- 2. noise -----------------------------------------------------
    noisy = np.stack([add_noise(clean[j], args.noise, args.level, seed=1000 + j)
                      for j in range(len(idx))])

    # ---- 3. filters ---------------------------------------------------
    Cs = stationary_mask_matrix(N, args.sigma_stat)
    Cn = blend_mask_matrix(N, args.sigma1, args.sigma2,
                           a=args.a, b=args.b, k=args.k, phase=args.phase)
    op_s, op_n = FilterOperator(Cs), FilterOperator(Cn)

    for name, C in (("stationary", Cs), ("nonstationary", Cn)):
        r = mask_report(C)
        print(f"  mask {name:14s} DC-gain err {r['dc_gain_max_error']:.1e}  "
              f"rank {r['numerical_rank']}  "
              f"active F rows {list(r['active_frequency_rows'])[:6]}")

    xs = filter_images(noisy, op_s)
    xn = filter_images(noisy, op_n)

    # ---- 4. diagnostics ----------------------------------------------
    print()
    print("=" * 70)
    print(f"{'':22s} {'PSNR':>8s} {'SSIM':>8s}")
    for name, arr in (("noisy (no denoise)", noisy),
                      ("stationary Gaussian", xs),
                      ("nonstationary", xn)):
        print(f"{name:22s} {psnr(clean, arr):>8.2f} {ssim(clean, arr):>8.3f}")

    dn = np.sqrt(np.mean((noisy - clean) ** 2))
    d_sn = np.sqrt(np.mean((xs - xn) ** 2))
    sep = d_sn / dn if dn > 0 else float("nan")
    print()
    print(f"RMS(noisy - clean)              = {dn:.5f}")
    print(f"RMS(stationary - nonstationary) = {d_sn:.5f}")
    print(f"separation ratio                = {sep:.1%}")
    if sep < 0.05:
        print("  --> TOO SMALL.  The two approaches are almost the same filter;")
        print("      no classification difference is possible.  Increase --b,")
        print("      or widen the gap between --sigma1 and --sigma2.")
    else:
        print("  --> the two approaches differ measurably; proceed.")

    if clean.ndim == 4:      # RGB, e.g. PathMNIST
        print()
        print("RGB dataset: the unchanged 1-D row operator is applied to each")
        print("channel independently.  Per-channel separation:")
        for ch in range(clean.shape[-1]):
            d = np.sqrt(np.mean((xs[..., ch] - xn[..., ch]) ** 2))
            print(f"  channel {ch}: RMS(stat - nonstat) = {d:.5f}")

    # Per-position smoothing width actually applied, for Section 5.2.
    w = args.a + args.b * np.cos(2 * np.pi * args.k * np.arange(N) / N + args.phase)
    print()
    print(f"blend weight w(i) ranges {w.min():.3f} .. {w.max():.3f} "
          f"(1 = wide sigma={args.sigma1}, 0 = narrow sigma={args.sigma2})")
    print(f"so the effective width sweeps between sigma {args.sigma2} and "
          f"{args.sigma1} across each row, {args.k} full cycle(s)")
    print(f"at the cyclic wrap (row edge, i=0) the kernel width is "
          f"sigma={args.sigma2 + w[0]*(args.sigma1-args.sigma2):.2f}; "
          f"widest smoothing at the wrap smears the left/right join")

    # ---- 5. figure ----------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("clean", clean), (f"noisy ({args.noise} {args.level})", noisy),
            (f"stationary sigma={args.sigma_stat}", xs),
            (f"nonstationary k={args.k} b={args.b}", xn),
            ("difference (stat - nonstat)", xs - xn)]
    n = len(idx)
    fig, ax = plt.subplots(len(rows), n, figsize=(2.0 * n, 2.1 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (name, arr) in enumerate(rows):
        for c in range(n):
            img = arr[c]
            if "difference" in name:
                # RGB differences are signed and cannot go through imshow as
                # colour; collapse to a per-pixel magnitude across channels.
                d = np.abs(img).max(axis=-1) if img.ndim == 3 else img
                v = np.abs(d).max() if img.ndim == 3 else np.abs(arr).max()
                if img.ndim == 3:
                    ax[r, c].imshow(d, cmap="magma", vmin=0, vmax=max(v, 1e-8))
                else:
                    ax[r, c].imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
            else:
                ax[r, c].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f"class {b.test.labels[idx[c]]}", fontsize=9)
            if c == 0:
                ax[r, c].set_ylabel(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

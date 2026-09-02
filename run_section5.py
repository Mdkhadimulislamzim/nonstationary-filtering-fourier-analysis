"""
run_section5.py -- the Section 5 experiment.

Two experiments, exactly as the paper separates them:

  Section 5.4  "Numerical illustration: raw data"
       "the proposed nonstationary filtering framework is applied directly
        to the original image data"
       -> no synthetic noise; three approaches; classification only.
       No clean reference exists, so no PSNR/SSIM here.

  Section 5.5  "Evaluation under synthetic noise"
       clean -> synthetic noise -> denoising -> classification, over the
       noise families of Section 5.5 (Gaussian, heavy-tailed, asymmetric)
       and several levels.  PSNR/SSIM reported as supplementary.

Three approaches :
  1  no denoising
  2  stationary Gaussian convolution filter
  3  proposed nonstationary filter

Filter parameters come from `nsfilt.search`, which sees validation data only
and writes a frozen selection to disk before the test set is touched.

Usage
-----
  python run_section5.py verify
  python run_section5.py search  --noise gaussian --level 0.10
  python run_section5.py raw                                  # Section 5.4
  python run_section5.py noisy   --noise gaussian --level 0.10  # Section 5.5
  python run_section5.py figures
  python run_section5.py cost
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from nsfilt.data import load_medmnist
from nsfilt.noise import DEFAULT_LEVELS
from nsfilt.masks import (stationary_mask_matrix, blend_mask_matrix,
                          sigma_profile, nonstationary_mask_matrix, mask_report)
from nsfilt.filtering import FilterOperator
from nsfilt.pipeline import FilteredDataset
from nsfilt.classifier import TrainConfig, train_and_predict
from nsfilt.metrics import (classification_report, mcnemar, holm, psnr, ssim,
                            summarise_runs)
from nsfilt.search import (SearchSpace, Selection, select_parameters,
                           default_selection)


APPROACHES = ("none", "stationary", "nonstationary")


# ---------------------------------------------------------------------------

def build_operators(N: int, sel: Selection, mask_family: str = "blend",
                    method: str = "dense"):
    """Approach 2 and Approach 3 operators from a frozen selection."""
    Cs = stationary_mask_matrix(N, sel.stationary["sigma"])
    p = sel.nonstationary
    if mask_family == "blend":
        Cn = blend_mask_matrix(N, sigma_1=p["sigma_1"], sigma_2=p["sigma_2"],
                               a=p.get("a", 0.5), b=p["b"], k=int(p["k"]),
                               phase=float(p.get("phase", 0.0)))
    else:
        s = sigma_profile(N, p["sigma_mid"], p["sigma_amp"], int(p["modulation_k"]),
                          float(p.get("phase", 0.0)))
        Cn = nonstationary_mask_matrix(N, s)
    return {"stationary": FilterOperator(Cs, method=method),
            "nonstationary": FilterOperator(Cn, method=method)}


def make_datasets(bundle, approach, ops, noise_model, noise_level):
    """One FilteredDataset per split for a given approach; nothing is copied."""
    op = None if approach == "none" else ops[approach]
    return {s: FilteredDataset(getattr(bundle, s).images, getattr(bundle, s).labels,
                               noise_model=noise_model, noise_level=noise_level,
                               split=s, op=op)
            for s in ("train", "val", "test")}


def run_one_condition(bundle, noise_model, noise_level, sel, cfg, seeds,
                      mask_family="blend", outdir="results", tag="", verbose=True):
    """Train and evaluate all three approaches under one noise condition."""
    os.makedirs(outdir, exist_ok=True)
    N = bundle.row_length
    ops = build_operators(N, sel, mask_family=mask_family)

    if verbose:
        for name, op in ops.items():
            rep = mask_report(op.C)
            print(f"[mask] {name:14s} DC-gain err {rep['dc_gain_max_error']:.2e}  "
                  f"rank {rep['numerical_rank']}  "
                  f"active F rows {list(rep['active_frequency_rows'][:6])}")

    # Stage 2: noise is applied per item with a seed derived from
    # (model, level, split, index), so all three approaches see byte-identical
    # noisy images without any of them ever being stored.
    results = {}
    for approach in APPROACHES:
        ds = make_datasets(bundle, approach, ops, noise_model, noise_level)

        # Per-image filtering throughput, for the Stage 5 timing column.
        t0 = time.perf_counter()
        n_probe = min(64, len(ds["test"]))
        ds["test"].processed(np.arange(n_probe))
        filt_time = (time.perf_counter() - t0) / n_probe * len(bundle.train)

        per_seed = []
        for seed in seeds:
            c = TrainConfig(**{**cfg.as_dict(), "seed": seed})
            if verbose:
                print(f"[run] {tag} approach={approach} seed={seed}", flush=True)
            res = train_and_predict(ds["train"], ds["val"], ds["test"],
                                    bundle.n_classes, c, verbose=verbose)
            rep = classification_report(res["test_true"], res["test_pred"],
                                        res["test_logits"], bundle.n_classes)
            rep.update(train_time_s=res["train_time_s"],
                       infer_time_s=res["infer_time_s"],
                       filter_time_s=filt_time, val_acc=res["val_acc"], seed=seed)
            rep["test_pred"] = res["test_pred"]
            per_seed.append(rep)
            if verbose:
                print(f"      acc {rep['accuracy']:.4f}  macro-F1 {rep['macro_f1']:.4f}"
                      f"  auc {rep.get('auc', float('nan')):.4f}", flush=True)

        entry = {"per_seed": per_seed, "summary": summarise_runs(per_seed),
                 "filter_time_s": filt_time}
        # Supplementary image quality, only where a clean reference exists
        # (Section 5.5).  Computed on the 1,000-image test split.
        if noise_model != "none":
            idx = np.arange(min(1000, len(ds["test"])))
            ref = np.stack([ds["test"].clean_float(i) for i in idx])
            out = ds["test"].processed(idx)
            entry["psnr"] = psnr(ref, out)
            entry["ssim"] = ssim(ref, out)
        results[approach] = entry

    # Paired significance tests on identical test items, seed 0.
    y = bundle.test.labels
    pairs = {}
    for a, b in (("none", "stationary"), ("none", "nonstationary"),
                 ("stationary", "nonstationary")):
        pairs[f"{a}_vs_{b}"] = mcnemar(y, results[a]["per_seed"][0]["test_pred"],
                                       results[b]["per_seed"][0]["test_pred"])
    corrected = holm({k: v["p_value"] for k, v in pairs.items()})
    for k in pairs:
        pairs[k]["p_holm"] = corrected[k]
    results["_mcnemar"] = pairs
    results["_condition"] = {"noise_model": noise_model, "noise_level": noise_level,
                             "mask_family": mask_family, "selection": sel.__dict__,
                             "dataset": bundle.flag, "size": bundle.size,
                             "config": cfg.as_dict(), "seeds": list(seeds)}

    path = os.path.join(outdir, f"results_{tag or noise_model}_{noise_level}_{bundle.size}px.pkl")
    with open(path, "wb") as fh:
        pickle.dump(results, fh)
    print_condition(results)
    print(f"[saved] {path}")
    return results


def print_condition(results):
    cond = results["_condition"]
    print()
    print("=" * 78)
    print(f"{cond['dataset']} {cond['size']}x{cond['size']} | "
          f"noise {cond['noise_model']} @ {cond['noise_level']}")
    print("=" * 78)
    print(f"{'approach':16s} {'accuracy':>16s} {'macro-F1':>16s} {'AUC':>10s} "
          f"{'PSNR':>7s} {'SSIM':>6s}")
    for a in APPROACHES:
        s = results[a]["summary"]
        acc = f"{s.get('accuracy_mean', float('nan')):.4f}±{s.get('accuracy_std', 0):.4f}"
        f1 = f"{s.get('macro_f1_mean', float('nan')):.4f}±{s.get('macro_f1_std', 0):.4f}"
        auc = f"{s.get('auc_mean', float('nan')):.4f}"
        pv = results[a].get("psnr", float("nan"))
        sv = results[a].get("ssim", float("nan"))
        print(f"{a:16s} {acc:>16s} {f1:>16s} {auc:>10s} {pv:>7.2f} {sv:>6.3f}")
    print()
    print("McNemar (paired, seed 0, Holm-corrected):")
    for k, v in results["_mcnemar"].items():
        print(f"  {k:32s} b={v['b']:4d} c={v['c']:4d} "
              f"p={v['p_value']:.4g}  p_holm={v['p_holm']:.4g}  [{v['test']}]")
    print()
    print("Timing (s):")
    for a in APPROACHES:
        r = results[a]
        ps = r["per_seed"][0]
        print(f"  {a:16s} filter~{r['filter_time_s']:8.1f}  "
              f"train {ps['train_time_s']:8.1f}  infer {ps['infer_time_s']:6.2f}")

    # Stage 5 requires the confusion matrix and class-wise sensitivity/recall.
    K = results[APPROACHES[0]]["per_seed"][0]["confusion_matrix"].shape[0]
    print()
    print("Class-wise recall (sensitivity), seed 0:")
    print(f"  {'approach':16s} " + " ".join(f"{'cls'+str(c):>8s}" for c in range(K)))
    for a in APPROACHES:
        rc = results[a]["per_seed"][0]["per_class_recall"]
        print(f"  {a:16s} " + " ".join(f"{v:>8.4f}" for v in rc))

    print()
    print("Confusion matrices (rows = true, cols = predicted), seed 0:")
    for a in APPROACHES:
        cm = results[a]["per_seed"][0]["confusion_matrix"]
        print(f"  {a}:")
        for r_i in range(K):
            print("    " + " ".join(f"{int(v):6d}" for v in cm[r_i]))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["verify", "search", "raw", "noisy",
                                        "sweep", "figures", "cost"])
    ap.add_argument("--dataset", default="octmnist")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--noise", default="gaussian")
    ap.add_argument("--level", type=float, default=0.10)
    ap.add_argument("--mask-family", default="blend", choices=["blend", "sigma"])
    ap.add_argument("--method", default="dense", choices=["dense", "lowrank", "freq"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--num-workers", type=int, default=4,
                    help="0 is safest on macOS if DataLoader workers misbehave")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "resnet50",
                             "resnet18_cifar", "resnet18_imagenet",
                             "resnet50_cifar", "resnet50_imagenet"])
    ap.add_argument("--selection", default="selection.json")
    ap.add_argument("--default-params", action="store_true",
                    help="use provisional filter parameters instead of a "
                         "validation-selected selection.json. For smoke tests "
                         "only; results obtained this way are not reportable.")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    if args.command == "verify":
        os.system("python verify_theory.py")
        return

    if args.command == "cost":
        from nsfilt.figures import cost_table
        rows = cost_table(N=args.size)
        print(f"{'method':32s} {'rank':>10s} {'time (s)':>10s} {'rel. error':>12s}")
        for r in rows:
            print(f"{r['method']:32s} {str(r['rank']):>10s} "
                  f"{r['time_s']:>10.4f} {r['rel_err']:>12.2e}")
        return

    if args.command == "figures":
        from nsfilt.figures import figure_mask_and_response, figure_frequency_components
        os.makedirs(args.outdir, exist_ok=True)
        p1 = figure_mask_and_response(path=os.path.join(args.outdir, "fig_mask_response.pdf"))
        p2, err = figure_frequency_components(
            path=os.path.join(args.outdir, "fig_components.pdf"))
        print(f"wrote {p1}\nwrote {p2}  (reconstruction error {err:.2e})")
        return

    bundle = load_medmnist(args.dataset, args.size, root=args.root)
    print(f"[data] {args.dataset} {args.size}px | "
          f"train {len(bundle.train)} val {len(bundle.val)} test {len(bundle.test)} | "
          f"{bundle.n_classes} classes, {bundle.n_channels} channel(s)")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      device=args.device, arch=args.arch,
                      num_workers=args.num_workers,
                      input_size=bundle.row_length, in_channels=3)

    if args.command == "search":
        sel = select_parameters(bundle, args.noise, args.level,
                                space=SearchSpace(), mask_family=args.mask_family,
                                device=args.device)
        sel.save(args.selection)
        print(f"[saved] frozen selection -> {args.selection}")
        return

    if os.path.exists(args.selection):
        sel = Selection.load(args.selection)
        print(f"[selection] loaded from {args.selection}: "
              f"stationary {sel.stationary}, nonstationary {sel.nonstationary}")
        if getattr(sel, "provisional", False):
            print("[selection] WARNING: this selection is PROVISIONAL "
                  "(not validation-selected). Do not report these results.")
    elif args.default_params:
        sel = default_selection(args.noise, args.level)
        print("=" * 70)
        print("WARNING: running with PROVISIONAL filter parameters.")
        print("  These were not selected on validation data. The protocol "
              "requires\n  parameters to be chosen by `search` and frozen "
              "before testing.")
        print("  Use this only to check that the pipeline runs.")
        print("=" * 70)
        print(f"[selection] provisional: stationary {sel.stationary}, "
              f"nonstationary {sel.nonstationary}")
    else:
        raise SystemExit(
            f"\nNo selection file at {args.selection!r}.\n\n"
            f"Filter parameters must be chosen on validation data before "
            f"testing:\n"
            f"    python run_section5.py search --size {args.size} "
            f"--noise {args.noise} --level {args.level}\n\n"
            f"For a quick pipeline check only, re-run with --default-params.\n")

    if args.command == "raw":
        # Section 5.4: original image data, no synthetic noise.
        run_one_condition(bundle, "none", 0.0, sel, cfg, args.seeds,
                          mask_family=args.mask_family, outdir=args.outdir,
                          tag="raw")
    elif args.command == "noisy":
        run_one_condition(bundle, args.noise, args.level, sel, cfg, args.seeds,
                          mask_family=args.mask_family, outdir=args.outdir)
    elif args.command == "sweep":
        for model in ("gaussian", "contaminated", "speckle"):
            for level in DEFAULT_LEVELS:
                run_one_condition(bundle, model, level, sel, cfg, args.seeds,
                                  mask_family=args.mask_family, outdir=args.outdir)


if __name__ == "__main__":
    main()

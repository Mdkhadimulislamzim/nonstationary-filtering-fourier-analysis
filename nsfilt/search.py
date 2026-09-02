"""
search.py -- "explore several parameter settings".

The instruction is to explore nonstationary-filter parameters, stationary
Gaussian parameters and noise conditions, and to identify a representative
condition in which the proposed filter performs best -- but explicitly:

    "This should not mean selecting the best result from the final test data.
     Parameters should be selected using validation data and then fixed
     before final testing."

That is enforced structurally here.  `select_parameters` never receives the
test split; it returns a frozen `Selection` which `run_section5.py` then
applies unchanged to the test set.  Every quantity it reports is a
validation quantity.

To keep the search affordable it uses a cheap proxy classifier (short
schedule, stratified training subset) rather than the full 100-epoch run.
The proxy is only used to *rank* configurations; the selected configuration
is then retrained under the full protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
import math
from itertools import product

import numpy as np

from .masks import (stationary_mask_matrix, sigma_profile,
                    nonstationary_mask_matrix, blend_mask_matrix, mask_report)
from .filtering import FilterOperator
from .pipeline import FilteredDataset
from .data import subsample
from .classifier import TrainConfig, train_and_predict
from .metrics import classification_report

__all__ = ["SearchSpace", "Selection", "select_parameters", "default_selection"]


@dataclass
class SearchSpace:
    # Approach 2: stationary Gaussian width
    sigma_stat: tuple = (0.5, 1.0, 1.5, 2.0, 3.0)
    # Approach 3 (blend): w(i) = a + b cos(2 pi k i / N)
    sigma_1: tuple = (2.0, 3.0, 4.0)      # wide kernel
    sigma_2: tuple = (0.5, 0.8, 1.2)      # narrow kernel
    a: tuple = (0.5,)
    b: tuple = (0.25, 0.5)
    k: tuple = (1, 2, 4)
    phase: tuple = (0.0, math.pi)   # pi = narrowest kernel at the cyclic wrap
    # Alternative --mask-family sigma: sigma(i) = mid + amp cos(2 pi k i / N)
    sigma_mid: tuple = (1.0, 1.5, 2.0)
    sigma_amp: tuple = (0.25, 0.5, 0.75)
    modulation_k: tuple = (1, 2, 4)
    sigma_phase: tuple = (0.0, math.pi)

    def stationary_grid(self):
        return [{"sigma": s} for s in self.sigma_stat]

    def nonstationary_grid(self, family: str = "blend"):
        out = []
        if family == "blend":
            for s1, s2, a, b, k, ph in product(self.sigma_1, self.sigma_2,
                                               self.a, self.b, self.k, self.phase):
                if s2 >= s1:                       # narrow must be narrower
                    continue
                if abs(b) > min(a, 1.0 - a) + 1e-12:   # keep w(i) in [0, 1]
                    continue
                out.append({"sigma_1": s1, "sigma_2": s2, "a": a, "b": b,
                            "k": k, "phase": ph})
        else:
            for mid, amp, k, ph in product(self.sigma_mid, self.sigma_amp,
                                           self.modulation_k, self.sigma_phase):
                if amp >= mid:                     # keep sigma(i) > 0
                    continue
                out.append({"sigma_mid": mid, "sigma_amp": amp,
                            "modulation_k": k, "phase": ph})
        return out


@dataclass
class Selection:
    """Frozen configuration. Written to disk before the test set is touched."""
    noise_model: str
    noise_level: float
    stationary: dict
    nonstationary: dict
    validation_scores: dict = field(default_factory=dict)
    proxy_config: dict = field(default_factory=dict)
    provisional: bool = False      # True = NOT validation-selected, not reportable

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=float)

    @staticmethod
    def load(path: str) -> "Selection":
        with open(path) as fh:
            return Selection(**json.load(fh))


def default_selection(noise_model: str = "none", noise_level: float = 0.0) -> "Selection":
    """Provisional parameters for smoke tests only.

    These are NOT validation-selected.  `select_parameters` must be run before
    any result is reported, or the protocol's requirement that parameters be
    "selected using validation data and then fixed before final testing" is
    violated.  The values below are a plausible mid-grid point, not a winner.
    """
    return Selection(
        noise_model=noise_model,
        noise_level=noise_level,
        stationary={"sigma": 1.5},
        nonstationary={"sigma_1": 2.5, "sigma_2": 1.2, "a": 0.5, "b": 0.25,
                       "k": 2, "phase": 0.0},
        provisional=True,
    )


def build_operator(N: int, kind: str, params: dict, method: str = "dense",
                   rank: int | None = None) -> FilterOperator:
    if kind == "stationary":
        C = stationary_mask_matrix(N, params["sigma"])
    elif kind == "nonstationary":
        if "sigma_1" in params:                     # blend family
            C = blend_mask_matrix(N, params["sigma_1"], params["sigma_2"],
                                  a=params.get("a", 0.5), b=params["b"],
                                  k=int(params["k"]),
                                  phase=float(params.get("phase", 0.0)))
        else:
            s = sigma_profile(N, params["sigma_mid"], params["sigma_amp"],
                              int(params["modulation_k"]), params.get("phase", 0.0))
            C = nonstationary_mask_matrix(N, s)
    else:
        raise ValueError(kind)
    return FilterOperator(C, method=method, rank=rank)


def select_parameters(bundle, noise_model: str, noise_level: float,
                      space: SearchSpace | None = None,
                      proxy_train_n: int = 12000, proxy_val_n: int = 4000,
                      proxy_epochs: int = 12, seed: int = 0, mask_family: str = "blend",
                      device: str = "auto", verbose: bool = True) -> Selection:
    """Choose sigma for Approach 2 and (mid, amp, k) for Approach 3 on validation."""
    space = space or SearchSpace()
    N = bundle.row_length

    tr = subsample(bundle.train, proxy_train_n, seed=seed)
    va = subsample(bundle.val, proxy_val_n, seed=seed)

    cfg = TrainConfig(epochs=proxy_epochs, seed=seed, device=device,
                      input_size=N, in_channels=3)

    def score(op):
        mk = lambda sp, name: FilteredDataset(sp.images, sp.labels,
                                              noise_model=noise_model,
                                              noise_level=noise_level,
                                              split=name, op=op)
        dtr, dva = mk(tr, "train"), mk(va, "val")
        res = train_and_predict(dtr, dva, dva, bundle.n_classes, cfg, verbose=False)
        rep = classification_report(res["test_true"], res["test_pred"],
                                    res["test_logits"], bundle.n_classes)
        return rep["macro_f1"], rep["accuracy"]

    scores = {}
    f1, acc = score(None)
    scores["none"] = {"macro_f1": f1, "accuracy": acc}
    if verbose:
        print(f"[search] no denoising            val macro-F1 {f1:.4f}", flush=True)

    best_stat, best_stat_f1 = None, -np.inf
    for p in space.stationary_grid():
        op = build_operator(N, "stationary", p)
        f1, acc = score(op)
        scores[f"stationary:{p}"] = {"macro_f1": f1, "accuracy": acc}
        if verbose:
            print(f"[search] stationary sigma={p['sigma']:<5} val macro-F1 {f1:.4f}", flush=True)
        if f1 > best_stat_f1:
            best_stat, best_stat_f1 = p, f1

    best_ns, best_ns_f1 = None, -np.inf
    for p in space.nonstationary_grid(mask_family):
        op = build_operator(N, "nonstationary", p)
        f1, acc = score(op)
        scores[f"nonstationary:{p}"] = {"macro_f1": f1, "accuracy": acc}
        if verbose:
            print(f"[search] nonstat {p}  val macro-F1 {f1:.4f}", flush=True)
        if f1 > best_ns_f1:
            best_ns, best_ns_f1 = p, f1

    sel = Selection(noise_model=noise_model, noise_level=noise_level,
                    stationary=best_stat, nonstationary=best_ns,
                    validation_scores=scores, proxy_config=cfg.as_dict())
    if verbose:
        C = build_operator(N, "nonstationary", best_ns).C
        rep = mask_report(C)
        print(f"[search] selected stationary   {best_stat}")
        print(f"[search] selected nonstationary{best_ns}")
        print(f"[search] mask DC-gain error {rep['dc_gain_max_error']:.2e}, "
              f"rank(99% energy) {rep['energy_rank_99']}, "
              f"active F rows {list(rep['active_frequency_rows'][:8])}")
    return sel

# Section 5 — Nonstationary image filtering for downstream classification

Paper: FOURIER ANALYSIS OF NONSTATIONARY FILTERING VIA
CYCLIC CONVOLUTION AND COMBINATION MATRICES

## Run order

```bash
pip install -r requirements.txt

python run_section5.py verify                                  # 105 checks, no data needed
python inspect_images.py                                       # pre-flight on real images
python run_section5.py search --noise gaussian --level 0.10    # validation only -> selection.json
python run_section5.py raw                                     # Section 5.4
python run_section5.py noisy  --noise gaussian --level 0.10    # Section 5.5
python run_section5.py sweep                                   # 3 families x 3 levels
python run_section5.py figures                                 # Stage 6
python run_section5.py cost                                    # dense vs FFT cost table
```

Develop at `--size 28` (fast), run finals at `--size 224`.

## Mapping to the protocol

| Requirement | Where |
|---|---|
| 1. Images from a classification dataset | `nsfilt/data.py` — OCTMNIST, official splits |
| 2. Row-wise filtering, no extra operators | `nsfilt/filtering.py` — last-axis only |
| 3. Three approaches | `run_section5.py: APPROACHES` |
| 4. Evaluation by classification | `nsfilt/metrics.py` |
| Classifier | `nsfilt/classifier.py` — aligned with the official MedMNIST benchmark |
| 5. Synthetic noise | `nsfilt/noise.py` — five models, `DEFAULT_LEVELS` |
| 6. Parameter exploration on validation, then frozen | `nsfilt/search.py` → `selection.json` |
| Stage 3, rank-one FFT (Lemma 3) | `core.conv_rank_one_apply`, `core.conv_apply_lowrank` |
| Stage 5 metrics | accuracy, macro-F1, confusion matrix, per-class recall, timing; PSNR/SSIM supplementary |
| Stage 6 frequency illustration | `nsfilt/figures.py` |

Section 5.4 (raw) and 5.5 (synthetic noise) are separate runs, as the paper
separates them. PSNR/SSIM are reported only in 5.5, where a clean reference
exists.

## The mask (Section 5.2)

Approach 3 uses the two-Gaussian blend
`C(u,τ) = g₂(u) + w(τ+u)·(g₁(u) − g₂(u))`, `w(i) = a + b·cos(2πki/N)`.

Row *i* of `conv(C)` is the convex blend `w(i)g₁ + (1−w(i))g₂` of two unit-mass
Gaussians. Hence:

* **Unit gain at every position** (verified to 2.2e-16). Necessary — a filter
  that rescales intensity position-dependently confounds the classification
  comparison.
* **`F` has exactly three nonzero rows: 0, k, N−k** (verified). By
  `comb(F)(i,j) = F(i−j,i)`, output mode *i* couples to input modes *i* and
  *i ± k* and nothing else. This is the spatial-variation ↔ frequency-coupling
  statement Section 5.2 asks for.
* **`b = 0` gives exactly the stationary mask** and a diagonal `comb(F)`,
  recovering Section 2.3. The baseline is *nested* inside the model.
* **Rank 3**, so three Lemma-3 terms reproduce the operator exactly, and two
  Lemma-6 terms do — this is Section 3's "sums of rank-one terms" realised, not
  approximated.

`--mask-family sigma` gives the alternative `σ(i) = mid + amp·cos(2πki/N)`
width modulation. It also has exact unit gain but activates all harmonic rows
`mk` of `F` (about 10% of the energy sits outside `{0,k,N−k}`), so `blend` is
the default.

The rank-one form `C = cd*` is available in `masks.rank_one_mask_matrix` as an
ablation. Note it is a position-dependent *gain* followed by a *fixed*
convolution — every column is the same shape scaled — and `conv(C)1 = c ⊛ conj(d)`
is not identically one unless `d` is constant, in which case the filter is
stationary. It cannot vary the filter width, which is why it is not Approach 3.

## Notes

* The cyclic wrap of the last pixel into the first is left intact. Section 5
  forbids additional preprocessing, so no padding is applied; for OCT both row
  edges are dark background.
* `search.py` never receives the test split. `selection.json` is written before
  testing.
* OCTMNIST's test set is 1,000 images, so a 1% accuracy gap is within sampling
  noise. Three seeds and a paired McNemar test with Holm correction are run by
  default; report mean ± std.
* The classifier follows the official MedMNIST v2 benchmark implementation:
  resolution-dependent ResNet-18 stem (CIFAR-style below 64px, torchvision at
  and above), best-validation-AUC epoch selection, milestones at 50% and 75% of
  the epoch budget.
* Expect Section 5.4 (raw, unnoised data) to show filtering *reduces* accuracy —
  blurring clean images costs information. Section 5.4's wording ("investigate
  whether") accommodates that honestly, and it motivates 5.5.

"""
verify_theory.py -- checks the implementation against the paper itself.

Reproduces the N = 4 mask/convolution layout of Section 2.1, the combination
layout of Section 2.2, the stationary/circulant collapse of Section 2.3,
Proposition 1, Lemma 1, Lemmas 2-4 and Corollary 1, Proposition 2, Lemma 5,
Lemma 6, and the printed matrices of Examples 1, 2 and 3.

Pure NumPy; no data, no training.  Run first, before anything else.
"""

import numpy as np

from nsfilt.core import (fourier_matrix, flip_operator, exp_k, conv_matrix,
                         comb_matrix, conv_inverse, comb_inverse,
                         frequency_response, mask_from_frequency_response,
                         comb_rank_one_apply, conv_rank_one_apply,
                         rank_one_frequency_response, has_conjugate_symmetry,
                         frequency_component_mask, frequency_selective_apply,
                         conv_apply_lowrank)
from nsfilt.masks import (gaussian_mask, stationary_mask_matrix, sigma_profile,
                          nonstationary_mask_matrix, blend_mask_matrix,
                          blend_rank_one_terms, mask_report)

PASS = FAIL = 0


def check(name, ok, extra=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}{(' | ' + extra) if extra else ''}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' | ' + extra) if extra else ''}")


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


print("=" * 78)
print("Section 2.1 -- conv(C) layout for N = 4")
print("=" * 78)
C4 = np.array([[f"c{i}{j}" for j in range(4)] for i in range(4)])
sym = conv_matrix(np.arange(16).reshape(4, 4))
expected_conv_labels = np.array([
    ["c00", "c31", "c22", "c13"],
    ["c10", "c01", "c32", "c23"],
    ["c20", "c11", "c02", "c33"],
    ["c30", "c21", "c12", "c03"]])
check("conv(C) symbolic layout matches the printed N=4 matrix",
      (conv_matrix(C4) == expected_conv_labels).all())

expected_comb_labels = np.array([
    ["c00", "c30", "c20", "c10"],
    ["c11", "c01", "c31", "c21"],
    ["c22", "c12", "c02", "c32"],
    ["c33", "c23", "c13", "c03"]])
check("comb(C) symbolic layout matches the printed N=4 matrix",
      (comb_matrix(C4) == expected_comb_labels).all())

rng = np.random.default_rng(0)
C = rng.standard_normal((7, 7)) + 1j * rng.standard_normal((7, 7))
check("conv inverse: C(i,j) = conv(C)(i+j, j)", close(conv_inverse(conv_matrix(C)), C))
check("comb inverse: C(i,j) = comb(C)(j, j-i)", close(comb_inverse(comb_matrix(C)), C))

x = rng.standard_normal(7)
direct = np.array([sum(C[(t - tau) % 7, tau] * x[tau] for tau in range(7)) for t in range(7)])
check("conv operator equals the defining sum", close(conv_matrix(C) @ x, direct))
direct_comb = np.array([sum(C[(t - tau) % 7, t] * x[tau] for tau in range(7)) for t in range(7)])
check("comb operator equals the defining sum (2.2)", close(comb_matrix(C) @ x, direct_comb))

print()
print("=" * 78)
print("Proposition 1 -- relation between conv and comb")
print("=" * 78)
N = 9
C = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
A, B = conv_matrix(C), comb_matrix(C)
i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
check("comb(C)(i,j) = conv(C)(2i-j, i)", close(B, A[(2 * i - j) % N, i % N]))
check("conv(C)(i,j) = comb(C)(j, 2j-i)", close(A, B[j % N, (2 * j - i) % N]))

print()
print("=" * 78)
print("Section 2.3 -- stationary masks give circulant matrices")
print("=" * 78)
c = rng.standard_normal(6)
Cs = np.outer(c, np.ones(6))
check("conv(C) = comb(C) for a stationary mask", close(conv_matrix(Cs), comb_matrix(Cs)))
circ = np.array([[c[(i - j) % 6] for j in range(6)] for i in range(6)])
check("stationary conv(C) is the circulant matrix of c", close(conv_matrix(Cs), circ))

print()
print("=" * 78)
print("Section 2.4 / 2.5 -- Fourier identities")
print("=" * 78)
N = 8
V = fourier_matrix(N)
x = rng.standard_normal(N)
check("DFT(x) = sqrt(N) V* x", close(np.fft.fft(x), np.sqrt(N) * V.conj().T @ x))
check("IDFT(x) = (1/sqrt(N)) V x", close(np.fft.ifft(x), V @ x / np.sqrt(N)))
check("V is unitary", close(V.conj().T @ V, np.eye(N)))
check("V^T = V (symmetry used in Lemma 5)", close(V.T, V))
J = flip_operator(N)
check("(V*)^2 x = J x", close(np.linalg.matrix_power(V.conj().T, 2) @ x, J @ x))
check("V^2 x = J x", close(np.linalg.matrix_power(V, 2) @ x, J @ x))
J4 = flip_operator(4)
check("J for N=4 matches the printed permutation matrix",
      close(J4, np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])))

C = rng.standard_normal((N, N))
check("F = V* C^T V* = (1/N) DFT2(C^T)",
      close(frequency_response(C), V.conj().T @ C.T @ V.conj().T))
check("C = (IDFT2(N F))^T inverts F", close(mask_from_frequency_response(frequency_response(C)), C))

print()
print("=" * 78)
print("Lemma 1 -- V* conv(C) V = comb(F) and V* comb(C) V = conv(F)")
print("=" * 78)
F = frequency_response(C)
check("V* conv(C) V = comb(F)", close(V.conj().T @ conv_matrix(C) @ V, comb_matrix(F)))
check("V* comb(C) V = conv(F)", close(V.conj().T @ comb_matrix(C) @ V, conv_matrix(F)))

print()
print("=" * 78)
print("Section 3 -- rank-one masks (Lemmas 2, 3, 4 and Corollary 1)")
print("=" * 78)
N = 12
V = fourier_matrix(N)
c = rng.standard_normal(N) + 1j * rng.standard_normal(N)
d = rng.standard_normal(N) + 1j * rng.standard_normal(N)
x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
C = np.outer(c, np.conj(d))
check("Lemma 2: comb(cd*)x = conj(d).*IDFT(DFT(c).*DFT(x))",
      close(comb_matrix(C) @ x, comb_rank_one_apply(c, d, x)))
check("Lemma 3: conv(cd*)x = IDFT(DFT(conj(d).*x).*DFT(c))",
      close(conv_matrix(C) @ x, conv_rank_one_apply(c, d, x)))
F = frequency_response(C)
check("Lemma 4: F = (1/N) DFT(conj(d)) DFT(c)^T", close(F, rank_one_frequency_response(c, d)))
check("rank-one mask gives a rank-one F", np.linalg.matrix_rank(F, tol=1e-8) == 1)

Dc, Dd = np.fft.fft(c), np.fft.fft(np.conj(d))
i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
check("Corollary 1 (3.4): V* conv(C) V (i,j) = (1/N) DFT(conj(d))(i-j) DFT(c)(i)",
      close(V.conj().T @ conv_matrix(C) @ V, Dd[(i - j) % N] * Dc[i % N] / N))
check("Corollary 1 (3.5): V* comb(C) V (i,j) = (1/N) DFT(conj(d))(i-j) DFT(c)(j)",
      close(V.conj().T @ comb_matrix(C) @ V, Dd[(i - j) % N] * Dc[j % N] / N))

print()
print("=" * 78)
print("Section 4.1 -- Fourier conjugate symmetry (Proposition 2)")
print("=" * 78)
for N in (8, 9):
    C = rng.standard_normal((N, N))
    F = frequency_response(C)
    a = has_conjugate_symmetry(F)
    b = has_conjugate_symmetry(comb_matrix(F))
    cc = has_conjugate_symmetry(conv_matrix(F))
    check(f"N={N}: F, comb(F), conv(F) all conjugate-symmetric for real C",
          a and b and cc)
    fk_ok = all(close(F[(-k) % N], np.conj(flip_operator(N) @ F[k]))
                for k in range(N // 2 + 1))
    check(f"N={N}: f_(N-k) = conj(J f_k)", fk_ok)

print()
print("=" * 78)
print("Section 4.2 -- Lemma 5, frequency decomposition")
print("=" * 78)
for N in (8, 9):
    C = rng.standard_normal((N, N))
    F = frequency_response(C)
    parts = [frequency_component_mask(F, k) for k in range(N // 2 + 1)]
    check(f"N={N}: C = sum_k C_k", close(np.real(sum(parts)), C))
    check(f"N={N}: every C_k is real", all(close(np.imag(p), 0) for p in parts))
    f0 = F[0]
    check(f"N={N}: C_0 = IDFT(f_0) exp_0^T",
          close(parts[0], np.outer(np.fft.ifft(f0), exp_k(N, 0))))
    ok = True
    for k in range(1, N // 2 + 1):
        if 2 * k == N:
            tgt = np.outer(np.fft.ifft(F[k]), exp_k(N, k))
        else:
            tgt = 2 * np.real(np.outer(np.fft.ifft(F[k]), exp_k(N, k)))
        ok &= close(parts[k], tgt)
    check(f"N={N}: C_k = 2 Re(IDFT(f_k) exp_k^T)  (and the k=N/2 case)", ok)

print()
print("=" * 78)
print("Section 4.3 -- Lemma 6, frequency-selective representation")
print("=" * 78)
for N in (8, 9):
    C = rng.standard_normal((N, N))
    F = frequency_response(C)
    x = rng.standard_normal(N)
    ok = True
    for k in range(N // 2 + 1):
        Ck = np.real(frequency_component_mask(F, k))
        lhs = conv_matrix(Ck) @ x
        rhs = frequency_selective_apply(F, k, x.astype(complex))
        ok &= close(lhs, np.real(rhs))
    check(f"N={N}: conv(C_k)x agrees with the Lemma 6 FFT formula", ok)
    total = sum(np.real(frequency_selective_apply(F, k, x.astype(complex)))
                for k in range(N // 2 + 1))
    check(f"N={N}: sum_k conv(C_k)x = conv(C)x", close(total, conv_matrix(C) @ x))

print()
print("=" * 78)
print("Example 1 -- printed F, C, conv(C) and the components")
print("=" * 78)
F_ex1 = np.array([
    [1, 2 - 1j, 1, 2 + 1j],
    [-1 + 2j, 1 - 4j, -1, 3 + 1j],
    [3, 1 - 1j, 2, 1 + 1j],
    [-1 - 2j, 3 - 1j, -1, 1 + 4j]])
C_ex1 = np.array([
    [4.25, 0.25, 2.25, -0.75],
    [3.75, -0.25, -1.25, -0.25],
    [-2.75, -3.75, 3.25, 1.25],
    [-3.25, -2.25, 1.75, 1.75]])
conv_ex1 = np.array([
    [4.25, -2.25, 3.25, -0.25],
    [3.75, 0.25, 1.75, 1.25],
    [-2.75, -0.25, 2.25, 1.75],
    [-3.25, -3.75, -1.25, -0.75]])
C_from_F = np.real(mask_from_frequency_response(F_ex1))
check("C recovered from F matches the printed C", close(C_from_F, C_ex1, 1e-8))
check("conv(C) matches the printed conv(C)", close(np.real(conv_matrix(C_ex1)), conv_ex1, 1e-8))
check("F is conjugate-symmetric (C is real)", has_conjugate_symmetry(F_ex1, 1e-8))

C0 = np.real(frequency_component_mask(F_ex1, 0))
C1 = np.real(frequency_component_mask(F_ex1, 1))
C2 = np.real(frequency_component_mask(F_ex1, 2))
C0_p = np.array([[1.5] * 4, [0.5] * 4, [-0.5] * 4, [-0.5] * 4])
C1_p = np.array([[1.0, 0.5, -1.0, -0.5], [2.5, 0.0, -2.5, 0.0],
                 [-3.0, -2.5, 3.0, 2.5], [-2.5, -2.0, 2.5, 2.0]])
C2_p = np.array([[1.75, -1.75, 1.75, -1.75], [0.75, -0.75, 0.75, -0.75],
                 [0.75, -0.75, 0.75, -0.75], [-0.25, 0.25, -0.25, 0.25]])
check("C_0 matches the printed matrix", close(C0, C0_p, 1e-8))
check("C_1 matches the printed matrix", close(C1, C1_p, 1e-8))
check("C_2 matches the printed matrix", close(C2, C2_p, 1e-8))
check("C_0 + C_1 + C_2 = C", close(C0 + C1 + C2, C_ex1, 1e-8))

conv_C0_p = np.array([[1.5, -0.5, -0.5, 0.5], [0.5, 1.5, -0.5, -0.5],
                      [-0.5, 0.5, 1.5, -0.5], [-0.5, -0.5, 0.5, 1.5]])
conv_C1_p = np.array([[1.0, -2.0, 3.0, 0.0], [2.5, 0.5, 2.5, 2.5],
                      [-3.0, 0.0, -1.0, 2.0], [-2.5, -2.5, -2.5, -0.5]])
conv_C2_p = np.array([[1.75, 0.25, 0.75, -0.75], [0.75, -1.75, -0.25, -0.75],
                      [0.75, -0.75, 1.75, 0.25], [-0.25, -0.75, 0.75, -1.75]])
check("conv(C_0) matches the printed matrix", close(np.real(conv_matrix(C0)), conv_C0_p, 1e-8))
check("conv(C_1) matches the printed matrix", close(np.real(conv_matrix(C1)), conv_C1_p, 1e-8))
check("conv(C_2) matches the printed matrix", close(np.real(conv_matrix(C2)), conv_C2_p, 1e-8))

F0 = np.zeros((4, 4), complex); F0[0] = F_ex1[0]
comb_F0_p = np.array([[1, 2 - 1j, 1, 2 + 1j], [0, 2 - 1j, 0, 0],
                      [0, 0, 1, 0], [0, 0, 0, 2 - 1j]])
check("comb(F_0) is diag(f_0) = diag(1, 2-i, 1, 2+i)",
      close(comb_matrix(F0), np.diag(F_ex1[0]), 1e-8))
check("PAPER ERROR: printed comb(F_0) disagrees with comb(F)(i,j) = F(i-j, i)",
      not close(comb_matrix(F0), comb_F0_p, 1e-8),
      "printed matrix has a full first row and (3,3) = 2-i; correct is diagonal with (3,3) = 2+i")
F1 = np.zeros((4, 4), complex); F1[1] = F_ex1[1]; F1[3] = F_ex1[3]
comb_F1_p = np.array([[0, -1 - 2j, 0, -1 + 2j], [1 - 4j, 0, 3 - 1j, 0],
                      [0, -1, 0, -1], [1 + 4j, 0, 3 + 1j, 0]])
check("comb(F_1) matches the printed matrix", close(comb_matrix(F1), comb_F1_p, 1e-8))
F2 = np.zeros((4, 4), complex); F2[2] = F_ex1[2]
comb_F2_p = np.array([[0, 0, 3, 0], [0, 0, 0, 1 - 1j],
                      [2, 0, 0, 0], [0, 1 + 1j, 0, 0]])
check("comb(F_2) matches the printed matrix (paper's row is mis-set)",
      close(comb_matrix(F2), comb_F2_p, 1e-8))

print()
print("=" * 78)
print("Example 2 -- rank-one mask with a diagonal convolution matrix, N = 6")
print("=" * 78)
N = 6
e1 = exp_k(N, 1)
C1 = 2 * np.real(np.outer(np.fft.ifft(np.ones(N)), e1))
C1_p = np.zeros((6, 6)); C1_p[0] = [2, 1, -1, -2, -1, 1]
check("C_1 matches the printed mask matrix", close(C1, C1_p, 1e-8))
check("conv(C_1) = diag(2, 1, -1, -2, -1, 1)",
      close(np.real(conv_matrix(C1)), np.diag([2, 1, -1, -2, -1, 1]), 1e-8))
F1_ex2 = np.zeros((6, 6)); F1_ex2[1] = 1; F1_ex2[5] = 1
check("F_1 matches the printed matrix", close(frequency_response(C1), F1_ex2, 1e-8))
comb_F1_ex2 = np.array([[0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0]])
check("comb(F_1) matches the printed matrix",
      close(comb_matrix(frequency_response(C1)), comb_F1_ex2, 1e-8))

print()
print("=" * 78)
print("Example 3 -- N = 8, three-term frequency decomposition")
print("=" * 78)
N = 8
f0 = np.array([1, 2 + 1j, 3 - 1j, 1j, 2, -1j, 3 + 1j, 2 - 1j])
f1 = np.array([2 + 3j, 1, 2, -1j, 1j, 3, -1 - 1j, -1])
f2 = np.array([-1j, 1j, 1, 4, -1j, 2 + 1j, 2 + 1j, -1])
J = flip_operator(N)
F = np.zeros((N, N), complex)
F[0], F[1], F[2] = f0, f1, f2                      # f_3 = f_4 = 0 implicitly
F[7], F[6] = np.conj(J @ f1), np.conj(J @ f2)
comb_F_p = np.array([
    [1, 2 - 3j, 1j, 0, 0, 0, -1j, 2 + 3j],
    [1, 2 + 1j, -1, -1, 0, 0, 0, 1j],
    [1, 2, 3 - 1j, -1 + 1j, 2 - 1j, 0, 0, 0],
    [0, 4, -1j, 1j, 3, 2 - 1j, 0, 0],
    [0, 0, -1j, 1j, 2, -1j, 1j, 0],
    [0, 0, 0, 2 + 1j, 3, -1j, 1j, 4],
    [1, 0, 0, 0, 2 + 1j, -1 - 1j, 3 + 1j, 2],
    [1, -1j, 0, 0, 0, -1, -1, 2 - 1j]])
check("comb(F) matches the printed matrix (confirms f_3 = f_4 = 0)",
      close(comb_matrix(F), comb_F_p, 1e-8))
check("F is conjugate-symmetric", has_conjugate_symmetry(F, 1e-8))

x = np.array([1, -2, 3, 1, 1, 0, -2, 1], dtype=float)
t0 = np.real(frequency_selective_apply(F, 0, x.astype(complex)))
t1 = np.real(frequency_selective_apply(F, 1, x.astype(complex)))
t2 = np.real(frequency_selective_apply(F, 2, x.astype(complex)))
p0 = np.array([1.4608, -0.5895, 1.8750, 2.3750, 4.2892, -5.6605, -3.1250, 2.3750])
p1 = np.array([-3.9142, 0.1893, -1.5858, -2.0074, -11.4142, 2.3107, -3.0858, -7.6642])
p2 = np.array([-1.3107, 5.9372, -7.4393, 2.9372, 5.8107, -8.4372, 2.9393, -8.4372])
tot = np.array([-3.7641, 5.5371, -7.1501, 3.3048, -1.3143, -11.7871, -3.2714, -13.7264])
check("k = 0 contribution matches the printed vector", close(t0, p0, 1e-4),
      f"max dev {np.abs(t0 - p0).max():.2e}")
check("k = 1 contribution matches the printed vector", close(t1, p1, 1e-4),
      f"max dev {np.abs(t1 - p1).max():.2e}")
check("k = 2 contribution matches the printed vector", close(t2, p2, 1e-4),
      f"max dev {np.abs(t2 - p2).max():.2e}")
check("sum matches the printed conv(C)x", close(t0 + t1 + t2, tot, 1e-4),
      f"max dev {np.abs(t0 + t1 + t2 - tot).max():.2e}")
C_ex3 = np.real(mask_from_frequency_response(F))
check("conv(C)x from the mask agrees with the frequency decomposition",
      close(np.real(conv_matrix(C_ex3)) @ x, t0 + t1 + t2))

print()
print("=" * 78)
print("Section 5 masks -- properties required for a fair denoising comparison")
print("=" * 78)
N = 64
Cs = stationary_mask_matrix(N, 2.0)
rep_s = mask_report(Cs)
check("stationary mask: conv(C) has unit gain at every position",
      rep_s["dc_gain_max_error"] < 1e-12, f"max error {rep_s['dc_gain_max_error']:.2e}")
check("stationary mask has rank one", rep_s["numerical_rank"] == 1)
check("stationary conv(C) is circulant",
      close(conv_matrix(Cs), np.array([[gaussian_mask(N, 2.0)[(i - j) % N]
                                        for j in range(N)] for i in range(N)])))

s = sigma_profile(N, 2.0, 1.0, k=2)
Cn = nonstationary_mask_matrix(N, s)
rep_n = mask_report(Cn)
check("nonstationary mask: conv(C) has unit gain at every position",
      rep_n["dc_gain_max_error"] < 1e-12, f"max error {rep_n['dc_gain_max_error']:.2e}")
check("nonstationary mask is genuinely nonstationary (columns differ)",
      not rep_n["is_circulant"])
check("nonstationary conv(C) is not circulant",
      not close(conv_matrix(Cn), conv_matrix(Cn)[np.ix_((np.arange(N) + 1) % N,
                                                        (np.arange(N) + 1) % N)]))
zero_amp = nonstationary_mask_matrix(N, sigma_profile(N, 2.0, 0.0, k=2))
check("zero modulation recovers the stationary mask exactly (nested baseline)",
      close(zero_amp, Cs))

Fn = frequency_response(Cn)
energy = np.abs(Fn).max(axis=1)
active = set(np.flatnonzero(energy > 1e-8 * energy.max()).tolist())
check("sigma modulated at k = 2 activates only F rows that are multiples of 2",
      all(r % 2 == 0 for r in active), f"{len(active)} active rows, all even")
tail = sorted(active - {0, 2, N - 2})
head_energy = energy[[0, 2, N - 2]].sum()
check("harmonic rows beyond {0, k, N-k} are a minor correction (sigma-profile mask)",
      energy[tail].sum() < 0.25 * head_energy,
      f"tail/head energy = {energy[tail].sum() / head_energy:.2e}; "
      f"exact 3-row support requires the blend mask instead")

xr = np.random.default_rng(3).standard_normal((5, N))
y_ref = xr @ np.real(conv_matrix(Cn)).T
y_lr, S = conv_apply_lowrank(Cn, xr.astype(complex))
check("full-rank Lemma 3 expansion reproduces the dense operator",
      close(np.real(y_lr), y_ref, 1e-8))
y_fs = sum(np.real(frequency_selective_apply(Fn, k, xr.astype(complex)))
           for k in range(N // 2 + 1))
check("Lemma 6 expansion reproduces the dense operator", close(y_fs, y_ref, 1e-8))
r99 = rep_n["energy_rank_99"]
check("nonstationary mask is low-rank (few rank-one terms carry 99% energy)",
      r99 <= 8, f"rank at 99% energy = {r99}")

print()
print("=" * 78)
print("Section 5 -- two-Gaussian blend mask (primary Approach 3)")
print("=" * 78)
for kk in (1, 2, 4):
    Cb = blend_mask_matrix(N, sigma_1=3.0, sigma_2=0.8, a=0.5, b=0.5, k=kk)
    rb = mask_report(Cb)
    check(f"k={kk}: unit gain at every position",
          rb["dc_gain_max_error"] < 1e-12, f"max error {rb['dc_gain_max_error']:.2e}")
    Fb = frequency_response(Cb)
    en = np.abs(Fb).max(axis=1)
    act = set(np.flatnonzero(en > 1e-10 * en.max()).tolist())
    check(f"k={kk}: F has exactly the three nonzero rows 0, k, N-k",
          act == {0, kk, N - kk}, f"active rows {sorted(act)}")
    check(f"k={kk}: mask has rank 3 (three rank-one terms, Section 3)",
          rb["numerical_rank"] == 3, f"rank {rb['numerical_rank']}")
    xb = np.random.default_rng(kk).standard_normal((4, N))
    ref = xb @ np.real(conv_matrix(Cb)).T
    lr, _ = conv_apply_lowrank(Cb, xb.astype(complex), rank=3)
    check(f"k={kk}: three Lemma-3 terms reproduce the dense operator exactly",
          close(np.real(lr), ref, 1e-8))
    fs = sum(np.real(frequency_selective_apply(Fb, q, xb.astype(complex)))
             for q in (0, kk))
    check(f"k={kk}: Lemma 6 with two terms reproduces the dense operator",
          close(fs, ref, 1e-8))
for kk in (1, 2, 4):
    kw = dict(sigma_1=3.0, sigma_2=0.8, a=0.5, b=0.5, k=kk)
    Cb = blend_mask_matrix(N, **kw)
    terms = blend_rank_one_terms(N, **kw)
    check(f"k={kk}: C equals the sum of its three explicit c d^* terms",
          close(sum(np.outer(c, np.conj(d)) for c, d in terms), Cb, 1e-12))
    xb = np.random.default_rng(100 + kk).standard_normal((4, N))
    ref = xb @ np.real(conv_matrix(Cb)).T
    lem3 = sum(conv_rank_one_apply(c, d, xb.astype(complex)) for c, d in terms)
    check(f"k={kk}: Lemma 3 applied to each term reproduces conv(C)x",
          close(np.real(lem3), ref, 1e-10))
    check(f"k={kk}: the three-term sum is real (conjugate pair cancels)",
          np.abs(np.imag(lem3)).max() < 1e-12)

Cb0 = blend_mask_matrix(N, 3.0, 0.8, a=0.5, b=0.0, k=1)
Fb0 = frequency_response(Cb0)
check("b = 0 makes comb(F) diagonal (circulant / stationary limit)",
      close(comb_matrix(Fb0) - np.diag(np.diag(comb_matrix(Fb0))), 0, 1e-10))

print()
print("=" * 78)
print(f"{PASS} passed, {FAIL} failed")
print("=" * 78)
raise SystemExit(1 if FAIL else 0)

"""compare_results.py -- print a side-by-side comparison of two saved results .pkl files.

Usage:
    python compare_results.py results/results_contaminated_0.2.pkl results/results_contaminated_0.2_224px.pkl
"""

import sys
import pickle


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python compare_results.py <result_a.pkl> <result_b.pkl>")
    path_a, path_b = sys.argv[1], sys.argv[2]
    a, b = load(path_a), load(path_b)

    label_a = f"{a['_condition']['size']}px"
    label_b = f"{b['_condition']['size']}px"

    print(f"{'approach':16s} {label_a + ' macro-F1':>18s} {label_b + ' macro-F1':>18s}")
    for name in ("none", "stationary", "nonstationary"):
        f1_a = a[name]["summary"]["macro_f1_mean"]
        f1_b = b[name]["summary"]["macro_f1_mean"]
        print(f"{name:16s} {f1_a:>18.4f} {f1_b:>18.4f}")

    print()
    print(f"{label_a} McNemar (p_holm):")
    for k, v in a["_mcnemar"].items():
        print(f"  {k:32s} p_holm={v['p_holm']:.4g}")

    print(f"{label_b} McNemar (p_holm):")
    for k, v in b["_mcnemar"].items():
        print(f"  {k:32s} p_holm={v['p_holm']:.4g}")


if __name__ == "__main__":
    main()

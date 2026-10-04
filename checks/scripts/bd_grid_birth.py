"""Birth position of the inserted rate (odd K) on the grid of Barletta and Dytso, dark current 0 (paper 2, Appendix B).

For each count change K-1 -> K with K odd, take the first grid point with K support points and the last grid point with K-1
(data_barletta_dytso/capPoisson_darkcurrent0_v1.mat: opt_pos_input, opt_prob_input; A grid from staircases.json). The newborn is
the support point at the K-grid point that has no partner among the K-1 points (the largest nearest-neighbour distance in the
Fisher coordinate s = 2 sqrt(x)). Its position in its gap, measured from the silent side: f = (s_new - s_left) / (s_right - s_left).
Also prints how far the grid point lies above the birth: the A step past the last K-1 point, in units of the gap.
Writes bd_grid_birth.json next to this file. Reads only.
"""
import os, json
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
DD = os.path.join(TW, "data_barletta_dytso")


def support(pos, prob, wmin=1e-6):
    ok = np.isfinite(pos) & np.isfinite(prob) & (prob > wmin)
    return np.sort(pos[ok])


def main():
    m = sio.loadmat(os.path.join(DD, "capPoisson_darkcurrent0_v1.mat"))
    P, W = m["opt_pos_input"], m["opt_prob_input"]
    st = json.load(open(os.path.join(DD, "staircases.json"), encoding="utf-8"))["0.0"]
    A = np.array(st["A"]); N = np.array(st["N"])
    out = {}
    for lo, hi, k0, k1 in st["transitions"]:
        if k1 % 2 == 0:
            continue
        i1 = int(np.argmin(np.abs(A - hi)))
        i0 = int(np.argmin(np.abs(A - lo)))
        x1 = support(P[i1], W[i1]); x0 = support(P[i0], W[i0])
        if len(x1) != k1 or len(x0) != k0:
            out[str(k1)] = {"error": f"support sizes {len(x0)}, {len(x1)} at A {A[i0]:.2f}, {A[i1]:.2f}"}
            continue
        s1, s0 = 2 * np.sqrt(x1), 2 * np.sqrt(x0)
        dist = np.array([np.min(np.abs(s0 - s)) for s in s1])
        j = int(np.argmax(dist))
        f = (s1[j] - s1[j - 1]) / (s1[j + 1] - s1[j - 1])
        out[str(k1)] = {"A_last_Km1": float(A[i0]), "A_first_K": float(A[i1]), "N_check": [int(N[i0]), int(N[i1])],
                        "newborn_index": j, "x_new": float(x1[j]), "f_left": float(f)}
        print(f"K = {k1:2d}: A {A[i0]:.2f} ({k0}) -> {A[i1]:.2f} ({k1}); newborn #{j} at x = {x1[j]:.4f}; f_left = {f:.4f}")
    # how far the first K-grid point lies above the true count change A_K (table2_primary.json, newborn-weight unfolding)
    t2 = json.load(open(os.path.join(HERE, "table2_primary.json"), encoding="utf-8"))
    over = {}
    for k, v in out.items():
        if k in t2 and "A_first_K" in v:
            v["A_K_true"] = t2[k]["A_K"]
            v["grid_above_birth"] = v["A_first_K"] - t2[k]["A_K"]
            over[k] = v["grid_above_birth"]
            print(f"K = {k}: true A_K {t2[k]['A_K']:.4f}, first grid point {v['A_first_K']:.2f}, above by {v['grid_above_birth']:.4f}")
    if over:
        out["furthest_above_K"] = int(max(over, key=over.get))
        print("grid point furthest above the birth at K =", out["furthest_above_K"])
    json.dump(out, open(os.path.join(HERE, "bd_grid_birth.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

"""q1928 (15 Oct 2026): Reading. support_exact.support can split one mass point into two adjacent grid points just after a birth
(A = 28.18: 7.624 and 7.773; A = 39.78: 14.354 and 14.619), so its N can exceed the count by one. Recount with points merged when
their Fisher distance 2 sqrt(x) differs by less than 0.2; control: the merged count at the Table II transitions of paper 2 (K = 5, 6, 7
appear at A = 18.1366, 28.2022, 39.7401) is K just above and K - 1 just below. Then recompute the H_steps fraction of q1926 with the
merged K, as a reading (the registered verdict of Amendment 2273 is unchanged)."""
import json
import numpy as np
from support_exact import support


def merged(A, lam):
    p = np.sort(np.array(support(A, lam)["pts"], float)); t = 2 * np.sqrt(p + lam)
    return 1 + int(np.sum(np.diff(t) >= 0.2))


for K, AK in ((5, 18.1366), (6, 28.2022), (7, 39.7401)):
    print("control K = %d: merged count at A_K - 0.03 = %d, at A_K + 0.03 = %d" % (K, merged(AK - 0.03, 0.0), merged(AK + 0.03, 0.0)))
rows = json.load(open("paper_neuron/q1926_rows.json"))
km = [merged(r["A"], r["lam"]) for r in rows]
changed = sum(1 for r, k in zip(rows, km) if k != r["K"])
frac = np.mean([r["D"] <= k for r, k in zip(rows, km)])
print("fibres %d; merged K differs from the registered K in %d; fraction D <= merged K %.3f (registered verdict used the unmerged K: 0.517)"
      % (len(rows), changed, frac))
print("READING q1928: done")

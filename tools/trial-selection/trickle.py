"""A quiet change in the law, with and without a few cases tried regardless of what the parties believe."""
import json, numpy as np
from multiprocessing import Pool
from sim import run
WIN = {"before": (600, 800), "0-25": (800, 825), "25-100": (825, 900), "100-300": (900, 1100), "300-800": (1100, 1600)}
def one(a):
    trickle, seed = a
    r = run(d=20, N=200, n_test=4000, k=5, c=0.2, T=1600, seed=seed, forget=0.999, trickle=trickle,
            shock=dict(t=800, angle_deg=30, announced=False, keep=0.1))
    w = lambda x, a, b: float(np.mean(x[a:b]))
    return dict(trickle=trickle, seed=seed, rate={n: w(r["trial_rate"], a, b) for n, (a, b) in WIN.items()},
                err={n: w(r["oracle_err"], a, b) for n, (a, b) in WIN.items()})
if __name__ == "__main__":
    ts = [0.0, 0.002, 0.005, 0.02]
    with Pool(4) as p:
        rs = p.map(one, [(t, s) for t in ts for s in range(6)])
    out = {str(t): {m: {n: float(np.mean([r[m][n] for r in rs if r["trickle"] == t])) for n in WIN} for m in ["rate", "err"]} for t in ts}
    json.dump(out, open("results/trickle.json", "w"), indent=1)
    for t, v in out.items():
        for m in v: print(t, m, {n: round(x, 4) for n, x in v[m].items()})

import json, numpy as np
from multiprocessing import Pool
from sim import run
def one(a):
    announced, seed = a
    r = run(d=20, N=200, n_test=4000, k=5, c=0.2, T=1600, seed=seed, forget=0.999,
            shock=dict(t=800, angle_deg=30, announced=announced, keep=0.1))
    w = lambda x, a, b: float(np.mean(x[a:b]))
    return dict(announced=announced, seed=seed,
        rate={n: w(r["trial_rate"], a, b) for n, (a, b) in WIN.items()},
        err={n: w(r["oracle_err"], a, b) for n, (a, b) in WIN.items()},
        unc={n: w(r["self_unc"], a, b) for n, (a, b) in WIN.items()})
WIN = {"before": (600, 800), "0-25": (800, 825), "25-100": (825, 900), "100-300": (900, 1100), "300-800": (1100, 1600)}
if __name__ == "__main__":
    with Pool(2) as p:
        rs = p.map(one, [(a, s) for a in [True, False] for s in range(6)])
    out = {}
    for a in [True, False]:
        R = [r for r in rs if r["announced"] == a]
        out["announced" if a else "unannounced"] = {m: {n: float(np.mean([r[m][n] for r in R])) for n in WIN} for m in ["rate", "err", "unc"]}
    json.dump(out, open("results/shock.json", "w"), indent=1)
    for kx, v in out.items():
        print(kx)
        for m in v: print("  ", m, {n: round(x, 4) for n, x in v[m].items()})

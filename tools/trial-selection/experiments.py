"""Run all experiments and save summaries to results/*.json.  usage: python experiments.py A|B|D|E|all"""
import json, sys, os
import numpy as np
from multiprocessing import Pool
from sim import run, sigmoid, PublicModel, rotate_toward_random

OUT = "results"
os.makedirs(OUT, exist_ok=True)
BASE = dict(d=20, N=200, n_test=4000)


def save(name, obj):
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, np.ndarray): return clean(o.tolist())
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, np.integer): return int(o)
        return o
    json.dump(clean(obj), open(f"{OUT}/{name}.json", "w"))


def interp_curve(n, err, grid):
    """error as a function of cumulative adjudications, on a common grid (log-log interp)."""
    n = np.asarray(n, float); err = np.asarray(err, float)
    keep = np.r_[True, np.diff(n) > 0]
    return np.exp(np.interp(np.log(grid), np.log(n[keep]), np.log(err[keep]), left=np.nan, right=np.nan))


# ---------- A + C: label efficiency, and what a decided-case benchmark measures ----------
def _A_one(args):
    k, seed = args
    kw = dict(BASE, k=k, c=0.2, T=1000, seed=seed)
    pk = run(**kw, keep_cases=True)
    rd = run(**kw, selection="random", n_schedule=pk["trials"])
    # random adjudication out to 40k cases, fit at log-spaced sizes (the yardstick for "how many random cases is the docket worth")
    rngL = np.random.default_rng(50_000 + seed)
    uL = pk["u_final"]
    XL = rngL.standard_normal((40000, 20)); yL = (rngL.random(40000) < sigmoid(k * XL @ uL)).astype(float)
    Xe = rngL.standard_normal((4000, 20)); pe = sigmoid(k * Xe @ uL)
    sizes = np.unique(np.round(np.logspace(np.log10(100), np.log10(40000), 30)).astype(int))
    long_err = []
    mL = PublicModel(20); prev = 0
    for n in sizes:
        mL.add(XL[prev:n], yL[prev:n]); mL.fit(); prev = n
        long_err.append(float(np.mean(np.abs(mL.predict(Xe) - pe))))
    long = dict(cum_trials=sizes, oracle_err=np.array(long_err))
    res = dict(k=k, seed=seed,
               pk_n=pk["cum_trials"], pk_err=pk["oracle_err"], pk_ang=pk["angle_deg"],
               rd_err=rd["oracle_err"], rd_ang=rd["angle_deg"],
               long_n=long["cum_trials"], long_err=long["oracle_err"],
               trial_rate=pk["trial_rate"], tried_pwin=pk["tried_pwin"],
               pop_acc=pk["pop_acc"], pop_ceiling=pk["pop_ceiling"])
    # prospective accuracy of the public model on the cases that went to trial (mature docket)
    tr = pk["trials"][50:]; ta = pk["tried_acc"][50:]; tc = pk["tried_ceiling"][50:]
    ok = tr > 0
    res["own_docket_acc"] = float(np.sum(tr[ok] * ta[ok]) / np.sum(tr[ok]))
    res["docket_ceiling"] = float(np.sum(tr[ok] * tc[ok]) / np.sum(tr[ok]))
    res["own_pop_acc"] = float(np.mean(pk["pop_acc"][50:]))

    # C: models of known quality scored on the mature docket vs on the population
    rng = np.random.default_rng(10_000 + seed)
    u = pk["u_final"]; cs = pk["cases"]
    late = cs["t"] >= 50
    Xd, yd = cs["X"][late], cs["y"][late]
    Xp = rng.standard_normal((20000, 20)); yp = (rng.random(20000) < sigmoid(k * Xp @ u)).astype(float)
    bench = {}
    base_d = base_p = None
    for theta in [0, 5, 10, 20, 40, 90]:
        xi = rng.standard_normal(20); xi -= (xi @ u) * u; xi /= np.linalg.norm(xi)
        w = np.cos(np.radians(theta)) * u + np.sin(np.radians(theta)) * xi
        cd = ((Xd @ w > 0) == (yd > 0.5)).astype(float)
        cp = ((Xp @ w > 0) == (yp > 0.5)).astype(float)
        if theta == 0: base_d, base_p = cd, cp
        dd, dp = base_d - cd, base_p - cp
        bench[theta] = dict(docket_acc=cd.mean(), pop_acc=cp.mean(), n_docket=len(cd),
                            gap_docket=dd.mean(), gap_pop=dp.mean(),
                            sd_docket=dd.std(), sd_pop=dp.std())
    res["bench"] = bench
    res["docket_pwin"] = float(yd.mean())
    return res


def exp_A(pool):
    ks = [1, 2, 5, 10, 20, 50]; seeds = range(8)
    rs = pool.map(_A_one, [(k, s) for k in ks for s in seeds])
    grid = np.unique(np.round(np.logspace(np.log10(120), np.log10(40000), 60)))
    out = {}
    for k in ks:
        R = [r for r in rs if r["k"] == k]
        pk = np.array([interp_curve(r["pk_n"], r["pk_err"], grid) for r in R])
        rd = np.array([interp_curve(r["pk_n"], r["rd_err"], grid) for r in R])
        lg = np.array([interp_curve(r["long_n"], r["long_err"], grid) for r in R])
        with np.errstate(all="ignore"):
            pk_m = np.where(np.isnan(pk).sum(0) <= 2, np.nanmean(pk, 0), np.nan)
            rd_m = np.where(np.isnan(rd).sum(0) <= 2, np.nanmean(rd, 0), np.nan)
            lg_m = np.nanmean(lg, 0)
        # multiplier: random adjudications needed to match the selected docket's error
        mult = []
        for i, g in enumerate(grid):
            if np.isnan(pk_m[i]): mult.append(np.nan); continue
            below = np.nonzero(lg_m <= pk_m[i])[0]
            mult.append(grid[below[0]] / g if len(below) else np.inf)
        n_final = np.array([r["pk_n"][-1] for r in R])
        out[k] = dict(
            grid=grid, pk_err=pk_m, rd_err=rd_m, long_err=lg_m, multiplier=mult,
            final_n=n_final.mean(),
            final_pk_err=np.mean([r["pk_err"][-1] for r in R]),
            final_rd_err=np.mean([r["rd_err"][-1] for r in R]),
            final_pk_err_sd=np.std([r["pk_err"][-1] for r in R]),
            final_rd_err_sd=np.std([r["rd_err"][-1] for r in R]),
            ratio_by_seed=[r["rd_err"][-1] / r["pk_err"][-1] for r in R],
            ratio_mid_by_seed=[np.mean(r["rd_err"][150:250]) / np.mean(r["pk_err"][150:250]) for r in R],
            trial_rate=np.mean([r["trial_rate"] for r in R], 0)[::10],
            own_docket_acc=np.mean([r["own_docket_acc"] for r in R]),
            docket_ceiling=np.mean([r["docket_ceiling"] for r in R]),
            own_pop_acc=np.mean([r["own_pop_acc"] for r in R]),
            pop_ceiling=np.mean([r["pop_ceiling"][-1] for r in R]),
            docket_pwin=np.mean([r["docket_pwin"] for r in R]),
            bench={th: {m: float(np.mean([r["bench"][th][m] for r in R])) for m in R[0]["bench"][th]}
                   for th in R[0]["bench"]},
        )
    save("A", out)
    return out


# ---------- B: trial cost and the drying-up of adjudication ----------
def _B_one(args):
    c, seed = args
    r = run(**dict(BASE, k=5, c=c, T=2000, seed=seed))
    return dict(c=c, seed=seed, cum=r["cum_trials"], err=r["oracle_err"], rate=r["trial_rate"])


def exp_B(pool):
    cs = [0.05, 0.1, 0.2, 0.3, 0.5]; seeds = range(6)
    rs = pool.map(_B_one, [(c, s) for c in cs for s in seeds])
    tt = np.unique(np.round(np.logspace(0, np.log10(1999), 40)).astype(int))
    out = {}
    for c in cs:
        R = [r for r in rs if r["c"] == c]
        cum = np.mean([r["cum"] for r in R], 0); err = np.mean([r["err"] for r in R], 0)
        rate = np.mean([r["rate"] for r in R], 0)
        out[c] = dict(t=tt, cum=cum[tt], err=err[tt],
                      rate_first50=rate[:50].mean(), rate_last500=rate[-500:].mean(),
                      final_cum=cum[-1], final_err=err[-200:].mean(),
                      cum_at=[cum[9], cum[99], cum[999], cum[1999]])
    save("B", out)
    return out


# ---------- D: facts the record does not capture ----------
def _D_one(args):
    h, k, seed = args
    d, d_o = 20, 15
    rng = np.random.default_rng(777 + seed)
    a = rng.standard_normal(d_o); a /= np.linalg.norm(a)
    b = rng.standard_normal(d - d_o); b /= np.linalg.norm(b)
    u = np.r_[np.sqrt(1 - h) * a, np.sqrt(h) * b]
    r = run(**dict(BASE, k=k, c=0.2, T=1000, seed=seed), u0=u, keep_cases=True)
    cs = r["cases"]

    def record_learner(X, y):
        m = PublicModel(d_o); m.add(X[:, :d_o], y); m.fit(); return m

    Xp = rng.standard_normal((20000, d)); pp = sigmoid(k * Xp @ u)
    yp = (rng.random(20000) < pp).astype(float)
    Xb = rng.standard_normal((40000, d)); yb = (rng.random(40000) < sigmoid(k * Xb @ u)).astype(float)
    best = record_learner(Xb, yb)

    def score(m):
        ph = m.predict(Xp[:, :d_o])
        nm = np.linalg.norm(m.mu)
        return dict(pop_acc=float(np.mean((ph > 0.5) == (yp > 0.5))),
                    sharp=float(nm), cos=float(m.mu @ a / max(nm, 1e-12)),
                    conf=float(np.mean(np.abs(ph - 0.5))))

    n_all = len(cs["y"])
    late = cs["t"] >= 50
    Xr = rng.standard_normal((n_all, d)); yr = (rng.random(n_all) < sigmoid(k * Xr @ u)).astype(float)
    res = dict(h=h, k=k, seed=seed, n_all=n_all, n_late=int(late.sum()),
               best=score(best),
               docket_all=score(record_learner(cs["X"], cs["y"])),
               docket_late=score(record_learner(cs["X"][late], cs["y"][late])),
               random_same_n=score(record_learner(Xr, yr)))
    # record-only learner scored on the mature docket itself (held-out half)
    idx = np.nonzero(late)[0]; rng.shuffle(idx); half = len(idx) // 2
    m = record_learner(cs["X"][idx[:half]], cs["y"][idx[:half]])
    ph = m.predict(cs["X"][idx[half:], :d_o])
    res["docket_holdout_acc"] = float(np.mean((ph > 0.5) == (cs["y"][idx[half:]] > 0.5)))
    return res


def exp_D(pool):
    hs = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]; ks = [5, 20]; seeds = range(6)
    rs = pool.map(_D_one, [(h, k, s) for h in hs for k in ks for s in seeds])
    out = {}
    for k in ks:
        out[k] = {}
        for h in hs:
            R = [r for r in rs if r["h"] == h and r["k"] == k]
            ent = dict(n_all=np.mean([r["n_all"] for r in R]), n_late=np.mean([r["n_late"] for r in R]),
                       docket_holdout_acc=np.mean([r["docket_holdout_acc"] for r in R]))
            for arm in ["best", "docket_all", "docket_late", "random_same_n"]:
                for m in ["pop_acc", "sharp", "cos", "conf"]:
                    v = [r[arm][m] for r in R]
                    ent[f"{arm}_{m}"] = np.mean(v); ent[f"{arm}_{m}_sd"] = np.std(v)
            out[k][h] = ent
    save("D", out)
    return out


# ---------- E: drifting law ----------
REGIMES = {
    "stationary": dict(forget=1.0),
    "stationary_trickle": dict(forget=1.0, trickle=0.005),
    "forget_0.999": dict(forget=0.999),
    "forget_0.995": dict(forget=0.995),
    "forget_0.98": dict(forget=0.98),
    "jump_W150_keep0.1": dict(forget=1.0, trickle=0.005, jump=dict(W=150, z=3.0, keep=0.1)),
    "jump_W40_keep0.1": dict(forget=1.0, trickle=0.005, jump=dict(W=40, z=3.5, keep=0.1)),
    "jump_W150_keep0.3": dict(forget=1.0, trickle=0.005, jump=dict(W=150, z=3.0, keep=0.3)),
}


def _E_one(args):
    name, drift, seed = args
    r = run(**dict(BASE, k=5, c=0.2, T=3000, seed=seed, drift=drift), **REGIMES[name])
    tr = r["trials"].astype(float)
    sm = np.convolve(tr[500:], np.ones(25) / 25, mode="valid"); sm -= sm.mean()
    lags = [25, 50, 100, 200, 400]
    ac = [float(np.corrcoef(sm[:-l], sm[l:])[0, 1]) if sm.std() > 0 else np.nan for l in lags]
    # spectral concentration: share of variance of the smoothed series in its largest Fourier peak
    f = np.abs(np.fft.rfft(sm)) ** 2; f[0] = 0
    peak_share = float(f.max() / f.sum()) if f.sum() > 0 else np.nan
    peak_period = float(len(sm) / max(np.argmax(f), 1))
    blk = lambda x: np.asarray(x, float).reshape(-1, 25).mean(1)
    return dict(name=name, drift=drift, seed=seed,
                rate=float(r["trial_rate"][500:].mean()), err=float(r["oracle_err"][500:].mean()),
                err_max=float(r["oracle_err"][500:].max()), err_end=float(r["oracle_err"][-250:].mean()),
                unc=float(r["self_unc"][500:].mean()), unc_end=float(r["self_unc"][-250:].mean()),
                ac=ac, peak_share=peak_share, peak_period=peak_period,
                n_resets=len(r["resets"]), resets=r["resets"],
                series=dict(rate=blk(r["trial_rate"]), err=blk(r["oracle_err"]), unc=blk(r["self_unc"])))


def exp_E(pool):
    seeds = range(6)
    jobs = [(n, 0.01, s) for n in REGIMES for s in seeds]
    jobs += [(n, dr, s) for n in ["forget_0.999", "forget_0.995", "forget_0.98", "stationary"]
             for dr in [0.003, 0.03] for s in range(4)]
    rs = pool.map(_E_one, jobs)
    out = {}
    for name, drift in sorted(set((r["name"], r["drift"]) for r in rs)):
        R = [r for r in rs if r["name"] == name and r["drift"] == drift]
        key = f"{name}@{drift}"
        out[key] = dict(
            rate=np.mean([r["rate"] for r in R]), err=np.mean([r["err"] for r in R]),
            err_sd=np.std([r["err"] for r in R]), err_end=np.mean([r["err_end"] for r in R]),
            err_max=np.mean([r["err_max"] for r in R]),
            unc=np.mean([r["unc"] for r in R]), unc_end=np.mean([r["unc_end"] for r in R]),
            ac=np.nanmean([r["ac"] for r in R], 0), peak_share=np.nanmean([r["peak_share"] for r in R]),
            peak_periods=[r["peak_period"] for r in R],
            n_resets=[r["n_resets"] for r in R], resets_seed0=R[0]["resets"],
            series_seed0=R[0]["series"], series_seed1=R[1]["series"] if len(R) > 1 else None)
    save("E", out)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    with Pool(2) as pool:
        for name, fn in [("A", exp_A), ("B", exp_B), ("D", exp_D), ("E", exp_E)]:
            if which in (name, "all"):
                fn(pool); print("done", name, flush=True)

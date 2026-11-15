"""Do facts fixed before a ruling predict patent-eligibility rulings, and less well as the doctrine ages?

Reads alice_rulings.csv (one row per district-court ruling on a challenge under
Alice Corp. v. CLS Bank, June 19, 2014; duplicates removed) and prints every
number quoted in the note. The table is built from CourtListener opinions by the
scripts in github.com/bsovocool16/caselaw-freq (analysis/coding.py, patents.py,
build_table.py). Outcome: "ineligible" means the challenger won.

"Score" is the chance that the model ranks a ruling the challenger won above a
ruling the challenger lost (AUC); 50% is guessing. Scores are cross-validated:
each pair of rulings is ranked by a model that saw neither.

Usage:  python tools/trial-selection/alice_test.py
"""
import csv, os, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALICE = "2014-06-19"
ALL = list(csv.DictReader(open(os.path.join(HERE, "alice_rulings.csv"))))
R = [r for r in ALL if r["outcome"]]
TOPICS = [k for k in ALL[0] if k.startswith("title_")]
COURTS = ["ded", "cand", "txed", "txwd", "cacd", "nysd", "ilnd"]

def yearfrac(d):
    return int(d[:4]) + (dt.date.fromisoformat(d).timetuple().tm_yday - 1) / 365.25

y = np.array([float(r["outcome"] == "ineligible") for r in R])
D = np.array([r["date"] for r in R]); ry = np.array([int(d[:4]) for d in D])
PUB = np.array([r["collection"] == "published" for r in R]); POST = np.array([r["collection"] == "posted" for r in R])
has_pat = np.array([bool(r["newest_issue"]) for r in R])
post_any = np.array([bool(r["newest_issue"]) and r["newest_issue"] > ALICE for r in R])
post_all = np.array([bool(r["oldest_issue"]) and r["oldest_issue"] > ALICE for r in R])
newest = np.array([yearfrac(r["newest_issue"]) if r["newest_issue"] else np.nan for r in R])
age = np.array([yearfrac(r["date"]) for r in R]) - newest
fy = np.array([int(r["filed_year"]) if r["filed_year"] else 0 for r in R])
pleading = np.array([r["stage"] == "pleading" for r in R])

def std(M):
    M = np.array(M, float)
    for j in range(M.shape[1]):
        m = np.isnan(M[:, j]); M[m, j] = np.nanmedian(M[:, j])
    return (M - M.mean(0)) / (M.std(0) + 1e-9)

# facts about the lawsuit, and facts about the patents
PROC = std(np.c_[pleading, np.array([[r["court"] == c for c in COURTS] for r in R]),
                 [int(r["npe_like_plaintiff"]) for r in R], [np.log1p(int(r["n_patents"])) for r in R]])
PAT = std(np.c_[newest, post_any, post_all, age, ~has_pat, [1 - int(r["has_title"]) for r in R],
                np.array([[int(r[k]) for k in TOPICS] for r in R])])
BOTH = np.c_[PROC, PAT]

def fit(X, y, lam=3.0):
    """Ridge-penalized logistic regression."""
    Xb = np.c_[np.ones(len(X)), X]; w = np.zeros(Xb.shape[1]); P = lam * np.eye(Xb.shape[1]); P[0, 0] = 0
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        step = np.linalg.solve((Xb * (p * (1 - p))[:, None]).T @ Xb + P, Xb.T @ (y - p) - P @ w); w += step
        if np.abs(step).max() < 1e-8: break
    return w

def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))

def cv_folds(X, y, seed, reps=40, k=5):
    """Repeated k-fold cross-validation. Returns one (test rows, their scores) pair per fold."""
    rng = np.random.default_rng(seed); folds = []
    for _ in range(reps):
        idx = rng.permutation(len(y))
        for f in range(k):
            te = idx[f::k]; tr = np.setdiff1d(idx, te)
            folds.append((te, np.c_[np.ones(len(te)), X[te]] @ fit(X[tr], y[tr])))
    return folds

def score(folds, y, mask=None, weights=None):
    """Share of (challenger won, challenger lost) pairs the model ranks correctly.

    Pairs are formed only inside a test fold, so both rulings were scored by the
    same model, which had seen neither. Comparing scores across folds, or averaging
    a ruling's score over repetitions, biases the result downward: a ruling's own
    outcome is missing from the data its model was trained on.
    """
    wins = pairs = 0.0
    for te, s in folds:
        if mask is not None: s, te = s[mask[te]], te[mask[te]]
        w = np.ones(len(te)) if weights is None else weights[te]
        pos, neg = y[te] == 1, y[te] == 0
        if not pos.any() or not neg.any(): continue
        ww = np.outer(w[pos], w[neg])
        wins += np.sum(ww * ((s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :]))); pairs += ww.sum()
    return wins / pairs if pairs else np.nan

def interval(folds, y, mask=None, seed=1, n=1000):
    """90% interval from resampling rulings (the fitted models are held fixed)."""
    rng = np.random.default_rng(seed)
    return np.nanpercentile([score(folds, y, mask, rng.multinomial(len(y), np.ones(len(y)) / len(y)).astype(float)) for _ in range(n)], [5, 95])

def diff_interval(a, b, seed=2, n=5000):
    rng = np.random.default_rng(seed)
    return np.percentile([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)], [5, 95])

def rate(m):
    return f"{100 * y[m].mean():.0f}% of {m.sum()}" if m.sum() else "-"

def results():
    out = {}
    vol = {}
    for r in ALL: vol[int(r["date"][:4])] = vol.get(int(r["date"][:4]), 0) + 1
    out["years"] = [dict(year=yr, all=vol.get(yr, 0), clear=int((ry == yr).sum()), won=float(y[ry == yr].mean()),
                         post_alice_share=float(post_any[(ry == yr) & has_pat].mean()), pleading=float(pleading[ry == yr].mean()))
                    for yr in range(2014, 2027)]
    out["scores"] = {}
    for cname, M in (("published", PUB), ("posted", POST)):
        yy, d = y[M], D[M]
        cut = "2016-07" if cname == "published" else np.sort(d)[M.sum() // 2]
        q = np.sort(d)[[M.sum() // 3, 2 * M.sum() // 3]]
        for fname, X in (("lawsuit", PROC), ("patent", PAT), ("both", BOTH), ("court", PROC[:, 1:1 + len(COURTS)])):
            folds = cv_folds(X[M], yy, seed=0); lo, hi = interval(folds, yy)
            rng = np.random.default_rng(3); dd = []
            for _ in range(1000):
                w = rng.multinomial(len(yy), np.ones(len(yy)) / len(yy)).astype(float)
                dd.append(score(folds, yy, d < cut, w) - score(folds, yy, d >= cut, w))
            out["scores"][cname, fname] = dict(n=int(M.sum()), score=score(folds, yy), lo=lo, hi=hi, cut=str(cut),
                early=score(folds, yy, d < cut), late=score(folds, yy, d >= cut), drop_lo=np.nanpercentile(dd, 5), drop_hi=np.nanpercentile(dd, 95),
                thirds=[score(folds, yy, t) for t in (d < q[0], (d >= q[0]) & (d < q[1]), d >= q[1])])
    return out

if __name__ == "__main__":
    res = results()
    print(f"{len(ALL)} rulings after removing duplicates; {len(R)} with a clear outcome "
          f"({PUB.sum()} in the published collection, 2014 to mid-2019; {POST.sum()} in the posted collection, mid-2019 on)")
    print("\nBy year of ruling: all rulings / clear rulings / challenger won / share on a patent issued after Alice / share decided on the pleadings")
    for v in res["years"]:
        print(f"  {v['year']}: {v['all']:3d} / {v['clear']:3d} / {100 * v['won']:3.0f}% / {100 * v['post_alice_share']:3.0f}% / {100 * v['pleading']:3.0f}%")
    for lo_, hi_ in ((2014, 2015), (2016, 2018), (2019, 2022), (2023, 2026), (2019, 2026), (2020, 2026)):
        print(f"  {lo_}-{hi_}: challenger won {rate((ry >= lo_) & (ry <= hi_))}")
    print("\nHow well facts fixed before the ruling predict it (score, 90% interval; then early vs late part of the collection; then thirds)")
    for (cname, fname), v in res["scores"].items():
        print(f"  {cname:9s} {fname:8s} n={v['n']}: {100 * v['score']:.1f}% [{100 * v['lo']:.0f}, {100 * v['hi']:.0f}] | before {v['cut']}: {100 * v['early']:.1f}%, after: {100 * v['late']:.1f}%, drop 90% [{100 * v['drop_lo']:+.0f}, {100 * v['drop_hi']:+.0f}] | thirds "
              + ", ".join(f"{100 * t:.1f}%" for t in v["thirds"]))
    print("\nChallenger win rate by when the newest asserted patent was issued")
    for cname, M in (("published", PUB), ("posted", POST)):
        a, b = y[M & has_pat & ~post_any], y[M & has_pat & post_any]; lo, hi = diff_interval(b, a)
        print(f"  {cname}: all patents issued before Alice {rate(M & has_pat & ~post_any)} | a patent issued after Alice {rate(M & has_pat & post_any)}"
              f" | difference {100 * (b.mean() - a.mean()):+.0f} points, 90% [{100 * lo:+.0f}, {100 * hi:+.0f}] | every patent issued after Alice {rate(M & post_all)}")
    print("\nPatent title words, posted collection (rulings whose opinion quotes a title)")
    ht = np.array([r["has_title"] == "1" for r in R])
    for k in TOPICS:
        t = np.array([r[k] == "1" for r in R]); print(f"  {k[6:]:9s}: {rate(POST & ht & t)} | other titles {rate(POST & ht & ~t)}")
    print("\nCases filed before Alice and after it, among rulings made in the same years")
    ok = fy > 0; print(f"  filing year read from the docket number for {ok.sum()} of {len(R)} clear rulings")
    for lo_, hi_ in ((2015, 2017), (2015, 2019)):
        m = ok & (ry >= lo_) & (ry <= hi_); a, b = y[m & (fy <= 2013)], y[m & (fy >= 2015)]; lo, hi = diff_interval(a, b)
        print(f"  ruled {lo_}-{hi_}: filed 2013 or earlier {rate(m & (fy <= 2013))} | filed 2014 {rate(m & (fy == 2014))} | filed 2015 or later {rate(m & (fy >= 2015))}"
              f" | before minus after {100 * (a.mean() - b.mean()):+.0f} points, 90% [{100 * lo:+.0f}, {100 * hi:+.0f}]")
    print("  by filing year: " + "; ".join(f"{yr}: {rate(ok & (fy == yr))}" for yr in range(2011, 2026)))

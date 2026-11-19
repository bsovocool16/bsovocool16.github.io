"""Do the claims of a patent predict whether a district court holds it ineligible under Alice?

Reads alice_rulings.csv, patent_facts.csv and patent_claims.jsonl.gz (USPTO PatentsView data
for the patents named in rulings with a clear outcome; independent claims only). Prints every
number and writes results/claims.json. Needs numpy and scikit-learn.

Scoring: AUC, the share of (challenger won, challenger lost) pairs of rulings that the
model ranks correctly; 50% is guessing. Pairs are formed only inside a test fold. Rulings
that share a patent are always kept in the same fold, so the model is never tested on a
patent it was trained on.

Usage: python tools/trial-selection/claims_test.py
"""
import csv, gzip, json, os, re, datetime as dt
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = f"{HERE}/results"
ALICE = dt.date(2014, 6, 19)
PAT = {r["patent_number"]: r for r in csv.DictReader(open(f"{HERE}/patent_facts.csv"))}
CPC = {p: list(enumerate(c for c in r["cpc_codes"].split(";") if c)) for p, r in PAT.items()}
CLAIMS = defaultdict(list)          # independent claims only
with gzip.open(f"{HERE}/patent_claims.jsonl.gz", "rt") as f:
    for line in f:
        c = json.loads(line); c["independent"] = True; CLAIMS[c["patent_number"]].append(c)

ALL = list(csv.DictReader(open(f"{HERE}/alice_rulings.csv")))
R = [r for r in ALL if r["outcome"]]
for r in R: r["pats"] = [p for p in r["patents"].split(";") if p in PAT]
R = [r for r in R if r["pats"]]                      # clear rulings with at least one patent we have data for
y = np.array([float(r["outcome"] == "ineligible") for r in R]); D = np.array([r["date"] for r in R])
PUB = np.array([r["collection"] == "published" for r in R]); POST = np.array([r["collection"] == "posted" for r in R])

# rulings that share a patent form one group
parent = list(range(len(R)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
seen = {}
for i, r in enumerate(R):
    for p in r["pats"]:
        if p in seen: parent[find(i)] = find(seen[p])
        seen[p] = i
GROUP = np.array([find(i) for i in range(len(R))])

# ---- facts about each patent
CPC_GROUPS = dict(business=r"^G06Q", computing=r"^G06F|^G06N|^G06T|^G06K|^G06V", network=r"^H04L|^H04W|^H04M|^H04B",
                  media=r"^H04N|^G11B|^G09|^H04R|^H04S", life=r"^A61|^C12|^G16H|^G01N|^C07", games=r"^G07F|^A63F")
def yr(s): d = dt.date.fromisoformat(s); return d.year + (d.timetuple().tm_yday - 1) / 365.25
def patent_facts(p):
    m = PAT[p]; cl = sorted(CLAIMS[p], key=lambda c: c["claim_number"]); ind = [c for c in cl if c["independent"]] or cl[:1]
    first = sorted(CPC[p])[0][1] if CPC[p] else ""; codes = [c for _, c in CPC[p]]
    c1 = ind[0]["text"].lower(); lens = [len(c["text"].split()) for c in ind]
    au = m["art_unit"]
    f = {"cpc_" + k: float(bool(re.search(v, first))) for k, v in CPC_GROUPS.items()}
    f.update(any_business_cpc=float(any(c.startswith("G06Q") for c in codes)),
             business_art_unit=float(au[:3] in ("362", "368", "369")) if au else np.nan,
             n_claims=np.log(float(m["n_claims"])), n_independent=np.log(float(m["n_independent_claims"])),
             shortest_independent=np.log(min(lens)), claim1_words=np.log(len(c1.split())),
             claim1_method=float(bool(re.match(r"\W*\d*\W*(a|an|the)?\s*(computer[- ]implemented\s+|improved\s+)?(method|process)", c1))),
             claim1_medium=float("readable" in c1[:160] or "storage medi" in c1[:160] or "program product" in c1[:160]),
             filed=yr(m["filing_date"]), priority=yr(m["earliest_priority_date"]), issued=yr(m["issue_date"]),
             pendency=yr(m["issue_date"]) - yr(m["filing_date"]),
             filed_after_alice=float(dt.date.fromisoformat(m["filing_date"]) > ALICE),
             issued_after_alice=float(dt.date.fromisoformat(m["issue_date"]) > ALICE))
    return f
PF = {p: patent_facts(p) for p in {p for r in R for p in r["pats"]}}
KEYS = list(next(iter(PF.values())))
def std(M):
    M = np.array(M, float)
    for j in range(M.shape[1]):
        m = np.isnan(M[:, j]); M[m, j] = np.nanmedian(M[:, j])
    return (M - M.mean(0)) / (M.std(0) + 1e-9)
# a ruling's patent facts: the average over its patents, plus the age of the newest patent at the ruling
STRUCT = std(np.c_[np.array([[np.nanmean([PF[p][k] for p in r["pats"]]) for k in KEYS] for r in R]),
                   [yr(r["date"]) - max(PF[p]["issued"] for p in r["pats"]) for r in R]])
COURTS = ["ded", "cand", "txed", "txwd", "cacd", "nysd", "ilnd"]
LAW = std(np.c_[[r["stage"] == "pleading" for r in R], np.array([[r["court"] == c for c in COURTS] for r in R]),
                [int(r["npe_like_plaintiff"]) for r in R], [np.log1p(int(r["n_patents"])) for r in R]])
def claim_text(r, which):
    out = []
    for p in r["pats"]:
        cl = sorted(CLAIMS[p], key=lambda c: c["claim_number"])
        out += [c["text"] for c in cl if (c["independent"] if which == "independent" else c["claim_number"] == 1)] or [cl[0]["text"]]
    return " ".join(out)
TEXT = {w: [claim_text(r, w) for r in R] for w in ("independent", "claim1")}

# ---- models: each takes train rows and test rows, returns scores for the test rows
def m_dense(X):
    def f(tr, te):
        m = LogisticRegression(C=0.3, max_iter=2000).fit(X[tr], y[tr]); return m.decision_function(X[te])
    return f
def m_text(which="independent", C=1.0, ngram=(1, 2), extra=None, svd=None):
    docs = TEXT[which]
    def f(tr, te):
        v = TfidfVectorizer(ngram_range=ngram, min_df=3, sublinear_tf=True, stop_words="english")
        A = v.fit_transform([docs[i] for i in tr]); B = v.transform([docs[i] for i in te])
        if svd:
            s = TruncatedSVD(svd, random_state=0).fit(A); A, B = s.transform(A), s.transform(B)
            mu, sd = A.mean(0), A.std(0) + 1e-9; A, B = (A - mu) / sd, (B - mu) / sd
            if extra is not None: A, B = np.c_[A, extra[tr]], np.c_[B, extra[te]]
            return LogisticRegression(C=0.3, max_iter=2000).fit(A, y[tr]).decision_function(B)
        return LogisticRegression(C=C, max_iter=2000).fit(A, y[tr]).decision_function(B)
    return f

def folds(mask, model, seed=0, reps=20, k=5, grouped=True):
    """Repeated k-fold over the rows in mask. Returns (test rows, scores) per fold."""
    rng = np.random.default_rng(seed); rows = np.where(mask)[0]; out = []
    for _ in range(reps):
        if grouped:
            gs = rng.permutation(np.unique(GROUP[rows])); fold_of = {g: i % k for i, g in enumerate(gs)}
            f = np.array([fold_of[g] for g in GROUP[rows]])
        else:
            f = rng.permutation(len(rows)) % k
        for j in range(k):
            te, tr = rows[f == j], rows[f != j]
            if len(set(y[te])) < 2: continue
            out.append((te, model(tr, te)))
    return out
def score(fl, mask=None, w=None):
    wins = pairs = 0.0
    for te, s in fl:
        if mask is not None: s, te = s[mask[te]], te[mask[te]]
        ww_ = np.ones(len(te)) if w is None else w[te]
        pos, neg = y[te] == 1, y[te] == 0
        if not pos.any() or not neg.any(): continue
        ww = np.outer(ww_[pos], ww_[neg])
        wins += np.sum(ww * ((s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :]))); pairs += ww.sum()
    return wins / pairs if pairs else np.nan
def interval(fl, mask=None, seed=1, n=600):
    """90% interval, resampling groups of rulings that share a patent."""
    rng = np.random.default_rng(seed); gs = np.unique(GROUP); gi = {g: i for i, g in enumerate(gs)}; gidx = np.array([gi[g] for g in GROUP])
    v = [score(fl, mask, rng.multinomial(len(gs), np.ones(len(gs)) / len(gs)).astype(float)[gidx]) for _ in range(n)]
    return np.nanpercentile(v, [5, 95])
RES = {}
def report(name, mask, model, key=None, **kw):
    fl = folds(mask, model, **kw); lo, hi = interval(fl)
    print(f"  {name:46s} {100 * score(fl):5.1f}%  [{100 * lo:.0f}, {100 * hi:.0f}]")
    if key: RES[key] = dict(n=int(mask.sum()), score=float(score(fl)), lo=float(lo), hi=float(hi))
    return fl

if __name__ == "__main__":
    print(f"{len(R)} clear rulings with patent data ({PUB.sum()} published collection, {POST.sum()} posted collection); "
          f"{len(PF)} patents; {len(np.unique(GROUP))} groups of rulings sharing a patent")
    for cname, M in (("POSTED collection, mid-2019 on", POST), ("PUBLISHED collection, 2014 to mid-2019", PUB), ("ALL clear rulings", PUB | POST)):
        print(f"\n{cname} (n={M.sum()}, challenger won {100 * y[M].mean():.0f}%)")
        c = cname.split()[0].lower()
        report("lawsuit facts (court, stage, plaintiff, count)", M, m_dense(LAW), key=c + "_lawsuit")
        report("patent facts (class, art unit, claim shape, dates)", M, m_dense(STRUCT), key=c + "_patent_facts")
        fl_text = report("claim text, independent claims", M, m_text(), key=c + "_claim_text")
        report("claim text + patent facts", M, m_text(svd=20, extra=STRUCT), key=c + "_text_and_facts")
        fl_all = report("everything (lawsuit + patent facts + text)", M, m_text(svd=20, extra=np.c_[STRUCT, LAW]), key=c + "_everything")
        if cname.startswith("POSTED"):
            d = D; cut = np.sort(D[M])[M.sum() // 2]
            RES["posted_claim_text"].update(early=float(score(fl_text, d < cut)), late=float(score(fl_text, d >= cut)), cut=str(cut))
            for nm, fl in (("claim text", fl_text), ("everything", fl_all)):
                print(f"    {nm}: rulings before {cut}: {100 * score(fl, d < cut):.1f}%, after: {100 * score(fl, d >= cut):.1f}%")
            print("  sensitivity (claim text): " + "; ".join(f"{nm} {100 * score(folds(M, mdl)):.1f}%" for nm, mdl in (
                ("claim 1 only", m_text("claim1")), ("unigrams", m_text(ngram=(1, 1))), ("C=0.1", m_text(C=0.1)), ("C=10", m_text(C=10)), ("SVD-20", m_text(svd=20)))))
            print("  ungrouped folds, for comparison (leaks shared patents): " + f"{100 * score(folds(M, m_text(), grouped=False)):.1f}%")
    print("\nSingle patent facts, posted collection: challenger win rate when the fact holds vs not")
    M = POST
    def rate(m): return f"{100 * y[m].mean():.0f}% of {m.sum()}"
    raw = {k: np.array([np.nanmean([PF[p][k] for p in r["pats"]]) for r in R]) for k in KEYS}
    for k in ("cpc_business", "any_business_cpc", "business_art_unit", "cpc_computing", "cpc_network", "cpc_media", "cpc_life", "cpc_games", "claim1_method", "claim1_medium", "filed_after_alice", "issued_after_alice"):
        a = np.nan_to_num(raw[k]) >= 0.5; print(f"  {k:20s} {rate(M & a):12s} | otherwise {rate(M & ~a)}")
    for k in ("claim1_words", "shortest_independent", "n_claims", "pendency"):
        v = raw[k]; med = np.nanmedian(v[M]); print(f"  {k:20s} below median {rate(M & (v < med)):12s} | above {rate(M & (v >= med))}")
    os.makedirs(RESULTS, exist_ok=True); json.dump(RES, open(f"{RESULTS}/claims.json", "w"), indent=1)

"""Score the blind reader test against the rulings, and compare with models on the same items.

The reader (Claude) saw one independent claim per ruling and nothing else, and wrote
blind/reader_predictions.csv before this script was run. Usage: python tools/trial-selection/blind_score.py
"""
import sys, os, csv, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import claims_test as T
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
B = f"{T.HERE}/blind"
key = {int(r["item"]): r["cluster_id"] for r in csv.DictReader(open(f"{B}/key.csv"))}
pred = {int(r["item"]): (float(r["p_ineligible"]), r["recognized"] == "1") for r in csv.DictReader(open(f"{B}/reader_predictions.csv"))}
row = {r["cluster_id"]: i for i, r in enumerate(T.R)}
idx = np.array([row[key[j]] for j in sorted(key)]); p = np.array([pred[j][0] for j in sorted(key)]); rec = np.array([pred[j][1] for j in sorted(key)])
yy = T.y[idx]
def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
def ci(y, s, n=4000, seed=1):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): v.append(auc(y[i], s[i]))
    return np.percentile(v, [5, 95])
OUT = {}
def line(name, y, s, key=None):
    lo, hi = ci(y, s); print(f"  {name:44s} {100 * auc(y, s):5.1f}%  [{100 * lo:.0f}, {100 * hi:.0f}]")
    if key: OUT[key] = dict(n=int(len(y)), score=float(auc(y, s)), lo=float(lo), hi=float(hi))
print(f"{len(idx)} rulings; challenger won {100 * yy.mean():.0f}% ({int(yy.sum())} of {len(yy)})")
line("reader, claim text only", yy, p, "reader")
line("reader, without 4 claims it recognized", yy[~rec], p[~rec], "reader_unrecognized")
late = np.array([T.R[i]["date"] > "2026-06-30" for i in idx])
if 0 < yy[late].sum() < late.sum(): line(f"reader, rulings after June 2026 (n={late.sum()})", yy[late], p[late])
call = p > 50
print(f"  reader calls at 50: right {int((call == (yy == 1))[p != 50].sum())} of {int((p != 50).sum())}; "
      f"said ineligible {int(call.sum())}, of which ineligible {int(yy[call].sum())}; said eligible {int((p < 50).sum())}, of which eligible {int((1 - yy[p < 50]).sum())}")
for lo_, hi_ in ((0, 30), (31, 50), (51, 70), (71, 84), (85, 100)):
    m = (p >= lo_) & (p <= hi_); print(f"    rated {lo_}-{hi_}: {m.sum()} rulings, {100 * yy[m].mean():.0f}% held ineligible")
# models trained on every other posted ruling that shares no patent with the sample
tr = np.array([i for i in np.where(T.POST)[0] if T.GROUP[i] not in set(T.GROUP[idx])])
v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english")
A = v.fit_transform([T.TEXT["independent"][i] for i in tr]); Bm = v.transform([T.TEXT["independent"][i] for i in idx])
line(f"word-count model of the claims (trained on {len(tr)})", yy, LogisticRegression(C=1, max_iter=2000).fit(A, T.y[tr]).decision_function(Bm), "word_count")
line("patent facts model", yy, LogisticRegression(C=0.3, max_iter=2000).fit(T.STRUCT[tr], T.y[tr]).decision_function(T.STRUCT[idx]))
line("lawsuit facts model (court, stage, ...)", yy, LogisticRegression(C=0.3, max_iter=2000).fit(T.LAW[tr], T.y[tr]).decision_function(T.LAW[idx]), "lawsuit")
X = np.c_[T.LAW, T.STRUCT]
line("lawsuit + patent facts model", yy, LogisticRegression(C=0.3, max_iter=2000).fit(X[tr], T.y[tr]).decision_function(X[idx]))
# reader plus court: logistic on reader rating + lawsuit facts, leave-one-out inside the sample
from sklearn.model_selection import LeaveOneOut
Z = np.c_[(p - p.mean()) / p.std(), T.LAW[idx]]; s = np.zeros(len(yy))
for a, b in LeaveOneOut().split(Z): s[b] = LogisticRegression(C=0.3, max_iter=2000).fit(Z[a], yy[a]).decision_function(Z[b])
line("reader rating + lawsuit facts (leave-one-out)", yy, s)
import json
json.dump(OUT, open(f"{T.RESULTS}/blind.json", "w"), indent=1)

"""Figures for the guest note on trial selection.

Plain matplotlib defaults, matching the other figures on the site.
Run from the repo root:  python tools/trial-selection/make_figures.py
Reads results/*.json (written by experiments.py, shock.py, trickle.py,
claims_test.py, blind_score.py) and alice_rulings.csv; writes four PNGs into assets/.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "assets", "trial-selection-")
load = lambda name: json.load(open(os.path.join(HERE, "results", name + ".json")))
nan = lambda v: np.array([np.nan if x is None else x for x in v], float)

# Fig 1 -- how fast the public record learns the court's rule, selected docket vs. random cases
A = load("A")
fig, axes = plt.subplots(2, 1, figsize=(7, 6.6), sharex=True)
for ax, k, title in ((axes[0], "20", "A predictable court: the cases the parties select teach faster"),
                     (axes[1], "1", "A noisy court: selection makes no difference")):
    a = A[k]; g = nan(a["grid"])
    ax.plot(g, nan(a["pk_err"]), lw=2, label="cases that went to trial because the parties disagreed")
    ax.plot(g, nan(a["long_err"]), lw=2, label="the same number of cases picked at random")
    ax.set_xscale("log"); ax.set_title(title, fontsize=11); ax.set_ylabel("error in predicted\nchance of winning")
    ax.set_ylim(0, None); ax.grid(alpha=0.25)
axes[0].legend(frameon=False, fontsize=9)
axes[1].set_xlabel("cases decided so far")
fig.tight_layout(); fig.savefig(OUT + "teaching.png", dpi=150); plt.close(fig)

# Fig 2 -- the same models scored on every dispute and on decided cases only
b = A["20"]["bench"]; th = ["0", "5", "10", "20", "40", "90"]
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(th)); w = 0.38
p1 = ax.bar(x - w / 2, [100 * b[t]["pop_acc"] for t in th], w, label="scored on every dispute")
p2 = ax.bar(x + w / 2, [100 * b[t]["docket_acc"] for t in th], w, label="scored on decided cases only")
for bars in (p1, p2):
    for r in bars: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{r.get_height():.0f}", ha="center", fontsize=9)
ax.axhline(50, color="gray", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(["the court's\nown rule", "5° off", "10° off", "20° off", "40° off", "unrelated\nto the rule"])
ax.set_ylabel("% of outcomes predicted correctly"); ax.set_ylim(0, 112)
ax.set_title("Scored on decided cases, good models look like guesses", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
fig.tight_layout(); fig.savefig(OUT + "benchmark.png", dpi=150); plt.close(fig)

# Fig 3 -- a one-time change in the law at period 0
S = load("shock"); T = load("trickle")
wins = ["before", "0-25", "25-100", "100-300", "300-800"]
labels = ["before the\nchange", "periods\n0–25", "25–100", "100–300", "300–800"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(labels, [S["unannounced"]["err"][w_] for w_ in wins], marker="o", lw=2, label="nobody announces the change")
ax.plot(labels, [T["0.02"]["err"][w_] for w_ in wins], marker="o", lw=2, label="unannounced, but 2% of disputes are tried regardless")
ax.plot(labels, [S["announced"]["err"][w_] for w_ in wins], marker="o", lw=2, label="the change is announced")
ax.set_ylabel("error in predicted\nchance of winning"); ax.set_ylim(0, 0.2); ax.grid(alpha=0.25)
ax.set_title("A change in the law that nobody announces stays unlearned", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "quiet-change.png", dpi=150); plt.close(fig)

# Fig 4 -- Alice rulings
sys.path.insert(0, HERE)
import alice_test as AT
res = AT.results()
fig = plt.figure(figsize=(7, 7.8))
axes = [fig.add_axes([0.13, 0.565, 0.84, 0.385]), fig.add_axes([0.37, 0.075, 0.60, 0.385])]
ax = axes[0]; yrs = [v["year"] for v in res["years"]]
ax.plot(yrs, [100 * v["won"] for v in res["years"]], marker="o", lw=2, color="C0")
for v in res["years"]: ax.text(v["year"], 100 * v["won"] + 3.5, str(v["clear"]), ha="center", fontsize=8, color="gray")
ax.axvline(2019.55, color="gray", ls=":"); ax.text(2019.7, 93, "CourtListener's coverage changes here:\nonly published opinions before this line", fontsize=8, color="gray", va="top")
ax.axhline(50, color="gray", lw=0.8); ax.set_ylim(30, 100); ax.set_xticks(yrs); ax.tick_params(axis="x", labelsize=9)
ax.set_ylabel("% of clear rulings the\nchallenger won"); ax.grid(alpha=0.25)
ax.set_title("Challengers won nine rulings in ten after Alice, and about six in ten since 2019", fontsize=11)
ax.text(0.01, 0.03, "gray numbers: clear rulings that year", transform=ax.transAxes, fontsize=8, color="gray")
ax = axes[1]; sc = res["scores"]; cl = load("claims"); bl = load("blind")
rows = [("which court", sc["posted", "court"]),
        ("court, stage, kind of plaintiff,\nnumber of patents", sc["posted", "lawsuit"]),
        ("patent issue dates and title words", sc["posted", "patent"]),
        ("patent class, examining unit,\nclaim length and type, dates", cl["posted_patent_facts"]),
        ("word counts from the claims", cl["posted_claim_text"]),
        ("Claude reading one claim\n(80 rulings)", bl["reader"])]
ypos = np.arange(len(rows))[::-1]
vals = np.array([100 * r["score"] for _, r in rows])
err = [vals - np.array([100 * r["lo"] for _, r in rows]), np.array([100 * r["hi"] for _, r in rows]) - vals]
ax.errorbar(vals, ypos, xerr=err, fmt="o", ms=7, capsize=4, color="C0", ecolor="C0", lw=2)
for v, yy_, hi in zip(vals, ypos, vals + err[1]): ax.text(hi + 0.8, yy_, f"{v:.0f}%", va="center", fontsize=9)
ax.axvline(50, color="gray", ls=":"); ax.text(50.4, len(rows) - 0.42, "guessing", color="gray", fontsize=8)
ax.set_yticks(ypos); ax.set_yticklabels([n for n, _ in rows], fontsize=9); ax.set_xlim(44, 88); ax.set_ylim(-0.6, len(rows) - 0.2)
ax.set_xlabel("how often the model picks the ruling the challenger won, %"); ax.grid(axis="x", alpha=0.25)
fig.text(0.5, 0.472, "Rulings since mid-2019: reading the claim predicts best, dates and titles not at all", ha="center", fontsize=11)
fig.savefig(OUT + "alice.png", dpi=150); plt.close(fig)
print("wrote 4 figures")

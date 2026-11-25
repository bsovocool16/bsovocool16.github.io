"""Figures for the fairness-opinion guest post (notes/fairness-opinion-numbers.html).

Plain matplotlib defaults, matching the other charts on the site.
Run:  python tools/make-fairness-figures.py
Reads tools/fairness-figures-data.json (the post's results, built from the
dataset in bsovocool16/fairness-opinion-legal) and writes PNGs into assets/.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT = "assets/"
D = json.load(open("tools/fairness-figures-data.json"))
rng = np.random.default_rng(7)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT + name, dpi=150)
    plt.close(fig)


# 1. Four roles, four answers: valuation as a multiple of the deal price.
groups = [
    ("bank advising the seller", D["bank_ratio_sample"]),
    ("company's expert", [v["v"] for v in D["appraisal"]["respondent"]["values"]]),
    ("Delaware court", [v["v"] for v in D["appraisal"]["court"]["values"]]),
    ("shareholders' expert", [v["v"] for v in D["appraisal"]["petitioner"]["values"]]),
]
fig, ax = plt.subplots(figsize=(7, 4))
for i, (label, vals) in enumerate(groups):
    v = np.clip(np.array(vals, float), 0.25, 4)
    y = len(groups) - 1 - i
    ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=10, alpha=0.45)
    med = np.median(v)
    ax.plot([med, med], [y - 0.3, y + 0.3], color="k", lw=2)
    ax.annotate(f"{med:.2f}x", (med, y + 0.32), ha="center", fontsize=9)
ax.axvline(1, color="gray", lw=1, ls="--")
ax.set_xscale("log")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.25x", "0.5x", "1x (deal price)", "2x", "4x"])
ax.set_yticks(range(len(groups)))
ax.set_yticklabels([g[0] for g in groups][::-1])
ax.set_ylim(-0.6, len(groups) - 0.3)
ax.set_title("Same method, four different answers")
save(fig, "fairness-four-answers.png")

# 2. Utz: our discount rate as we adopted Citi's choices.
steps = [
    "our first model",
    "+ Bloomberg-style beta",
    "+ Utz's own asset beta",
    "+ 20-year Treasury",
    "+ 6.5% equity risk premium",
    "Citi (midpoint)",
]
vals = [s["wacc"] for s in D["utz"]]
fig, ax = plt.subplots(figsize=(7, 3.6))
y = np.arange(len(steps))[::-1]
bars = ax.barh(y, vals)
bars[-1].set_color("C3")
ax.axvspan(7.9, 8.5, alpha=0.15, color="C3", label="Citi's range (7.9-8.5%)")
for yi, v in zip(y, vals):
    ax.text(v + 0.1, yi, f"{v:.1f}%", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(steps)
ax.set_xlim(0, 11.5)
ax.set_xlabel("discount rate (%)")
ax.set_title("Our Utz discount rate, adopting Citi's choices one at a time")
ax.legend(loc="upper right", fontsize=8)
save(fig, "fairness-utz.png")

# 3. Where the deal price falls in the bank's range.
h = D["pos_hist"]
x0 = np.array([b["x0"] for b in h]); x1 = np.array([b["x1"] for b in h])
n = np.array([b["n"] for b in h])
col = ["C0" if b < 0 else ("C3" if a >= 1 else "C7") for a, b in zip(x0, x1)]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.bar(x0, n, width=x1 - x0, align="edge", color=col)
ax.axvspan(0, 1, alpha=0.08, color="k")
p = D["pos"]
ax.text(-0.5, n.max() * 0.9, f"below\n{p['below']:.0%}", ha="center")
ax.text(0.5, n.max() * 0.9, f"inside\n{p['inside']:.0%}", ha="center")
ax.text(1.5, n.max() * 0.9, f"above\n{p['above']:.0%}", ha="center")
ax.set_xticks([-1, 0, 0.5, 1, 2])
ax.set_xticklabels(["≤ -1", "low end", "middle", "high end", "≥ 2"])
ax.set_ylabel("cash deals")
ax.set_title("Where the deal price falls within the bank's valuation range")
save(fig, "fairness-price-position.png")

# 4. Round numbers: how endpoints end, and how wide ranges are.
hp = D["heap"]; w = D["dr_width"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.2))
lab = ["whole\n(9%)", "half\n(9.5%)", "quarter\n(9.25%)", "other\n(9.37%)"]
a1.bar(lab, [hp["integer"] * 100, hp["half"] * 100, hp["quarter"] * 100, hp["other"] * 100])
a1.set_ylabel("% of endpoints")
a1.set_title("How discount-rate endpoints end", fontsize=10)
keys = ["0.5", "1.0", "1.5", "2.0", "3.0", "4.0", "5.0"]
a2.bar([k.rstrip("0").rstrip(".") for k in keys], [w[k] * 100 for k in keys])
a2.set_xlabel("range width (percentage points)")
a2.set_ylabel("% of ranges")
a2.set_title("How wide the range is", fontsize=10)
save(fig, "fairness-round-numbers.png")

# 5. Discount rates vs Treasury yields.
r = D["rates"]
yrs = [d["year"] for d in r]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(yrs, [d["dr"] for d in r], marker="o", label="typical bank discount rate (midpoint)")
ax.plot(yrs, [d["tnx10y"] for d in r], marker="o", label="10-year Treasury yield")
ax.set_ylabel("%")
ax.set_ylim(0, None)
ax.set_title("Bank discount rates and Treasury yields, by year")
ax.legend()
save(fig, "fairness-rates.png")

# 6. Share of variation explained, adding factors.
st = D["r2_steps"]
fig, ax = plt.subplots(figsize=(7, 2.8))
labels = ["year", "+ industry", "+ deal type", "+ which bank"]
vals = [s["r2"] * 100 for s in st]
y = np.arange(len(labels))[::-1]
ax.barh(y, vals)
for yi, v in zip(y, vals):
    ax.text(v + 0.6, yi, f"{v:.0f}%", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(0, 50)
ax.set_xlabel("share of variation in discount rates explained (adjusted R², %)")
ax.set_title("Adding one factor at a time")
save(fig, "fairness-which-bank.png")

# 7. Same deal, two banks: gap between discount-rate midpoints.
ph = D["pairs_hist"]
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar([b["x"] for b in ph], [b["n"] for b in ph], width=0.22, align="edge")
ax.set_xlabel("gap between the two banks' discount rates (percentage points)")
ax.set_ylabel("pairs of banks")
ax.set_title(f"Two banks, same company, same day ({D['pairs']['n']} pairs)")
save(fig, "fairness-same-deal.png")

# 8. How the summaries changed.
t = D["trends"]
yrs = [d["year"] for d in t]
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(yrs, [d["judgment"] * 100 for d in t], label='"professional judgment"')
ax.plot(yrs, [d["capm"] * 100 for d in t], label="names CAPM")
ax.plot(yrs, [d["perp"] * 100 for d in t], label="uses a growth rate")
ax.plot(yrs, [d["exitm"] * 100 for d in t], label="uses an earnings multiple")
for xv, lab in [(2016.0, "Trulia"), (2017.6, "DFC")]:
    ax.axvline(xv, color="gray", ls="--", lw=1)
    ax.text(xv + 0.2, 95, lab, fontsize=8, color="gray")
ax.set_ylim(0, 100)
ax.set_ylabel("% of bank DCF summaries")
ax.set_title("How the DCF summaries changed")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
save(fig, "fairness-disclosure-trends.png")

# 9. Equity risk premiums since 2010: courts vs banks.
fig, ax = plt.subplots(figsize=(7, 2.6))
b = np.array(D["book_erp_2010"]); c = np.array(D["court_erp_2010"])
ax.scatter(b, 1 + rng.uniform(-0.15, 0.15, len(b)), s=10, alpha=0.45, label="bank presentations")
ax.scatter(c, 0 + rng.uniform(-0.15, 0.15, len(c)), s=18, color="C3", label="Delaware courts")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Delaware courts", "bank presentations"])
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("equity risk premium (%)")
ax.set_title("Equity risk premiums used since 2010")
save(fig, "fairness-erp.png")

# 10. Court's discount rate vs the bank's range, same deal.
cvb = D["court_vs_bank"]
fig, ax = plt.subplots(figsize=(7, 4))
for i, cse in enumerate(cvb):
    y = len(cvb) - 1 - i
    nb = len(cse["banks"])
    for j, (bank, lo, hi) in enumerate(cse["banks"]):
        off = (j - (nb - 1) / 2) * 0.22
        ax.plot([lo, hi], [y + off, y + off], color="C0", lw=4, alpha=0.6,
                label="bank's range" if i == 0 and j == 0 else None)
    ax.plot(cse["court"], y, "D", color="C3", label="court" if i == 0 else None)
ax.set_yticks(range(len(cvb)))
ax.set_yticklabels([c["case"] for c in cvb][::-1])
ax.set_xlabel("discount rate (%)")
ax.set_title("The court's discount rate and the bank's range, same deal")
ax.legend(loc="upper right")
save(fig, "fairness-court-vs-bank.png")
print("wrote 10 figures")

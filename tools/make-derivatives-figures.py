"""Figures for the derivatives guest post (notes/derivatives-law-in-lean.html).

Plain matplotlib defaults, matching the other charts on the site.
Run:  python tools/make-derivatives-figures.py
The numbers are the post's results: elections coded from 199 ISDA Schedules
filed on EDGAR, the worked close-out example, and the compiler runs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "assets/"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT + name, dpi=150)
    plt.close(fig)


# 1. How many filed Schedules leave each gap open, and how many patch the known ones.
rows = [
    ("Non-affected Party can terminate\nwhile itself in default", 101, "open"),
    ("Nobody can terminate after a\nmerger affecting both parties", 87, "open"),
    ("5(c)(ii) reads two ways", 8, "open"),
    ("Automatic Early Termination\nreaches back", 4, "open"),
    ("Joint Termination Event race", 2, "open"),
    ("Time limit on 2(a)(iii) suspension", 9, "patched"),
    ("2(a)(iii) circularity removed", 2, "patched"),
]
fig, ax = plt.subplots(figsize=(7, 4.4))
y = list(range(len(rows)))[::-1]
colors = ["C3" if r[2] == "open" else "C0" for r in rows]
ax.barh(y, [r[1] for r in rows], color=colors)
for yi, r in zip(y, rows):
    ax.annotate(f"{r[1]} ({r[1] / 199:.0%})", (r[1] + 2, yi), va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlim(0, 199)
ax.set_xlabel("Schedules, of 199 filed on EDGAR")
ax.set_title("Drafters patch the known gaps and leave the others open")
save(fig, "derivatives-schedules.png")

# 2. The close-out race: net payment to the non-defaulting party by who moves first.
pairs = [
    ("Worked example", 100, 80),
    ("Defaulter owes 50 more,\n40% recovery", 60, 52),
    ("Non-defaulting party owes 50,\n40% recovery", 20, -18),
]
fig, ax = plt.subplots(figsize=(7, 3.6))
ys = list(range(len(pairs)))[::-1]
h = 0.36
for yi, (label, a, b) in zip(ys, pairs):
    ax.barh(yi + h / 2, a, height=h, color="C0", label="non-defaulting party designates first" if yi == ys[0] else None)
    ax.barh(yi - h / 2, b, height=h, color="C3", label="defaulting party designates first" if yi == ys[0] else None)
    ax.annotate(str(a), (max(a, 0) + 2, yi + h / 2), va="center", fontsize=9)
    ax.annotate(str(b), (max(b, 0) + 2, yi - h / 2), va="center", fontsize=9)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(ys)
ax.set_yticklabels([p[0] for p in pairs], fontsize=9)
ax.set_xlim(-30, 115)
ax.set_xlabel("Net payment to the non-defaulting party")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Same facts, different payment, depending on who moves first")
save(fig, "derivatives-race.png")

# 3. The compiler: how often the printed form determines a unique outcome.
total = 119_400
bars = [
    ("Printed form", total - 5_499, 5_499),
    ("With the four\nextra elections", total, 0),
]
fig, ax = plt.subplots(figsize=(7, 3.0))
for yi, (label, ok, gap) in zip([1, 0], bars):
    ax.barh(yi, ok / total * 100, color="C0", label="one outcome" if yi == 1 else None)
    ax.barh(yi, gap / total * 100, left=ok / total * 100, color="C3", label="text permits two" if yi == 1 else None)
    ax.annotate(f"{gap:,} gaps", (101, yi), va="center", fontsize=9)
ax.set_yticks([1, 0])
ax.set_yticklabels([b[0] for b in bars], fontsize=9)
ax.set_xlim(0, 118)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Share of 119,400 simulated histories (%)")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False)
ax.set_title("Where the 2002 form leaves the outcome open")
save(fig, "derivatives-compiler.png")

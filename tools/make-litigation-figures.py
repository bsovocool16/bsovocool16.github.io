"""Priest-Klein style figures for the legal-superintelligence note.

Plain matplotlib defaults (no styling), matching the existing Item 1A chart.
Run:  python tools/make-litigation-figures.py
Writes three PNGs into assets/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "assets/"
STAGES = ["filing", "motions", "discovery", "pretrial", "trial"]
x = np.arange(len(STAGES))


def normal(grid, mu, sd):
    return np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


# Fig 1 -- expectations converge; parties settle.
# they converge at pretrial and settle there; the case never reaches trial
xs = x[:4]
p = np.array([9.5, 8.2, 6.9, 5.7])
d = np.array([1.2, 2.4, 3.9, 5.4])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, p, marker="o", label="plaintiff's expected value")
ax.plot(xs, d, marker="o", label="defendant's expected value")
ax.fill_between(xs, d, p, alpha=0.12)
ax.plot([3], [5.55], marker="D", color="k", zorder=5)
ax.annotate("settlement\n(case never reaches trial)", xy=(3, 5.55), xytext=(1.35, 6.9),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-0.2, 4.2)
ax.set_xticks(x)
ax.set_xticklabels(STAGES)
ax.set_ylabel("expected value of claim ($m)")
ax.set_title("Expectations converge: the parties settle")
ax.legend()
fig.tight_layout()
fig.savefig(OUT + "litigation-convergence.png", dpi=150)
plt.close(fig)

# Fig 2 -- expectations stay apart; the court fixes the price.
p2 = np.array([9.5, 9.0, 8.4, 8.0, 7.8])
d2 = np.array([1.2, 1.6, 2.2, 2.6, 2.8])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, p2, marker="o", label="plaintiff's expected value")
ax.plot(x, d2, marker="o", label="defendant's expected value")
ax.fill_between(x, d2, p2, alpha=0.12)
ax.plot([4], [4.6], marker="D", color="k", zorder=5)
ax.annotate("judgment\n(court fixes the value)", xy=(4, 4.6), xytext=(2.4, 4.9),
            arrowprops=dict(arrowstyle="->"))
ax.set_xticks(x)
ax.set_xticklabels(STAGES)
ax.set_ylabel("expected value of claim ($m)")
ax.set_title("Expectations stay apart: the court prints a price")
ax.legend()
fig.tight_layout()
fig.savefig(OUT + "litigation-no-convergence.png", dpi=150)
plt.close(fig)

# Fig 3 -- Priest-Klein: each side holds a distribution, not a point.
grid = np.linspace(-1, 12, 600)
dp = normal(grid, 6.6, 1.7)   # plaintiff
dd = normal(grid, 3.9, 1.7)   # defendant
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(grid, dp, label="plaintiff's distribution")
ax.plot(grid, dd, label="defendant's distribution")
ax.fill_between(grid, np.minimum(dp, dd), alpha=0.25,
                label="overlap: settlement range")
ax.axvline(5.25, linestyle=":", color="gray")
ax.annotate("true value of claim", xy=(5.25, 0.10), xytext=(-0.6, 0.115),
            arrowprops=dict(arrowstyle="->", color="gray"), color="gray")
ax.set_ylim(0, 0.28)
ax.set_xlabel("value of claim ($m)")
ax.set_ylabel("density")
ax.set_title("Priest-Klein: each side holds a distribution, not a point")
ax.legend()
fig.tight_layout()
fig.savefig(OUT + "priest-klein-distributions.png", dpi=150)
plt.close(fig)

print("wrote 3 figures to " + OUT)

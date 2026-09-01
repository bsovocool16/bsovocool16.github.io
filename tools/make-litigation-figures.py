"""Priest-Klein style figures for the legal-superintelligence note.

Plain matplotlib defaults (no styling), matching the existing Item 1A chart.
Run:  python tools/make-litigation-figures.py
Writes four PNGs into assets/.

The paths are stylized, not fitted. Expectations move at discrete information
events (the motion ruling, the close of discovery) rather than smoothly, so the
series below are deliberately non-linear: little movement early, most of the
revision through discovery, then flattening as the two sides run out of things
to learn.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "assets/"
STAGES = ["filing", "motions", "discovery", "pretrial", "trial"]
x = np.arange(len(STAGES))

# shared stylized paths (see module docstring)
P_CONV = np.array([9.5, 9.1, 7.0, 5.7])   # plaintiff, converging case
D_CONV = np.array([1.2, 2.2, 4.3, 5.4])   # defendant, converging case
SD = np.array([2.2, 1.7, 1.0, 0.5])       # forecast dispersion, narrows over time


def normal(grid, mu, sd):
    return np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def box_stats(mus, sds, label):
    """Quartiles/whiskers implied by a normal forecast at each stage."""
    out = []
    for mu, sd in zip(mus, sds):
        # a claim cannot be worth less than nothing, so clip the lower tail at 0
        out.append(dict(med=mu, q1=max(mu - 0.674 * sd, 0.0),
                        q3=mu + 0.674 * sd,
                        whislo=max(mu - 1.5 * sd, 0.0), whishi=mu + 1.5 * sd,
                        fliers=[], label=label))
    return out


# Fig 1 -- expectations converge; parties settle before trial.
xs = x[:4]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, P_CONV, marker="o", label="plaintiff's expected value")
ax.plot(xs, D_CONV, marker="o", label="defendant's expected value")
ax.fill_between(xs, D_CONV, P_CONV, alpha=0.12)
ax.plot([3], [5.55], marker="D", color="k", zorder=5)
ax.annotate("settlement", xy=(3.06, 5.6), xytext=(3.3, 7.1),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-0.2, 4.3)
ax.set_xticks(x)
ax.set_xticklabels(STAGES)
ax.set_ylabel("expected value of claim ($m)")
ax.set_title("Expectations converge: the parties settle")
ax.legend(loc="center left")
fig.tight_layout()
fig.savefig(OUT + "litigation-convergence.png", dpi=150)
plt.close(fig)

# Fig 2 -- expectations stay apart; the court fixes the price.
p2 = np.array([9.5, 9.2, 8.3, 8.0, 7.8])
d2 = np.array([1.2, 1.5, 2.3, 2.6, 2.8])
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

# Fig 3 -- the same path, but each side holds a spread at every stage.
fig, ax = plt.subplots(figsize=(7, 4))
pos = np.arange(4, dtype=float)
bp = ax.bxp(box_stats(P_CONV, SD, "plaintiff"), positions=pos - 0.15,
            widths=0.22, showfliers=False, patch_artist=True)
bd = ax.bxp(box_stats(D_CONV, SD, "defendant"), positions=pos + 0.15,
            widths=0.22, showfliers=False, patch_artist=True)
for box in bp["boxes"]:
    box.set_facecolor("C0"); box.set_alpha(0.45); box.set_edgecolor("C0")
for box in bd["boxes"]:
    box.set_facecolor("C1"); box.set_alpha(0.45); box.set_edgecolor("C1")
for part, colour in ((bp, "C0"), (bd, "C1")):
    for key in ("whiskers", "caps", "medians"):
        for artist in part[key]:
            artist.set_color(colour)
ax.set_xticks(pos)
ax.set_xticklabels(STAGES[:4])
ax.set_ylabel("expected value of claim ($m)")
ax.set_title("Forecasts narrow as the case proceeds")
handles = [plt.Line2D([], [], color="C0", lw=6, alpha=0.45, label="plaintiff's forecast"),
           plt.Line2D([], [], color="C1", lw=6, alpha=0.45, label="defendant's forecast")]
ax.legend(handles=handles, loc="upper right")
fig.tight_layout()
fig.savefig(OUT + "litigation-forecast-boxes.png", dpi=150)
plt.close(fig)

# Fig 4 -- a single time slice: the two forecasts overlap.
grid = np.linspace(-1, 12, 600)
dp = normal(grid, 6.6, 1.7)
dd = normal(grid, 3.9, 1.7)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(grid, dp, label="plaintiff's forecast")
ax.plot(grid, dd, label="defendant's forecast")
ax.fill_between(grid, np.minimum(dp, dd), alpha=0.25, label="overlap: settlement range")
ax.axvline(5.25, linestyle=":", color="gray")
ax.annotate("true value of claim", xy=(5.25, 0.10), xytext=(-0.6, 0.115),
            arrowprops=dict(arrowstyle="->", color="gray"), color="gray")
ax.set_ylim(0, 0.28)
ax.set_xlabel("value of claim ($m)")
ax.set_ylabel("density")
ax.set_title("Overlapping forecasts at time $t$ create settlement opportunities")
ax.legend()
fig.tight_layout()
fig.savefig(OUT + "priest-klein-distributions.png", dpi=150)
plt.close(fig)

print("wrote 4 figures to " + OUT)

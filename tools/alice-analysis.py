"""Alice test: did a doctrinal shock spike court-forced exits, then decay?

Data: FJC Integrated Database, civil terminations, SY2010-SY2019.
  https://www.fjc.gov/research/idb/civil-cases-filed-terminated-and-pending-sy-1988-present
  (per-year SAS files; note the naming is inconsistent: cvNN_0.sas7bdat through
  2017, bare cvNN.sas7bdat for 2018-19.)

Treatment group: nature of suit 830 (patent). Control: all other civil NOS.
Event: Alice Corp. v. CLS Bank, decided 2014-06-19 (end of 2014Q2).

Disposition codes are per the FJC codebook (verified, not assumed):
  6 = judgment on motion before trial   7/8/9 = jury/directed/court trial
  12 = dismissed voluntarily            13 = dismissed settled
  14 = dismissed other                  0/1/10/11/16/18/19/20 = transfer/admin

Usage:  python tools/alice-analysis.py /path/to/idb-dir
Writes assets/alice-dismissal-gap.png and tools/alice-panel.csv.
"""
import sys, glob, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IDB = sys.argv[1] if len(sys.argv) > 1 else "idb/"
ADMIN = [0, 1, 10, 11, 16, 18, 19, 20]

frames = []
for f in sorted(glob.glob(os.path.join(IDB, "cv1*.sas7bdat"))):
    frames.append(pd.read_sas(f, encoding="latin-1")[["NOS", "DISP", "TERMDATE", "DISTRICT"]])
d = pd.concat(frames).dropna(subset=["TERMDATE", "DISP", "NOS"])
d = d[(d.TERMDATE >= "2011-01-01") & (d.TERMDATE < "2020-01-01")]
d = d[~d.DISP.isin(ADMIN)]
d["patent"] = d.NOS == 830
d["quarter"] = d.TERMDATE.dt.to_period("Q")
d["dism_other"] = d.DISP == 14          # where Alice-era 101 dismissals land
d["judg_motion"] = d.DISP == 6
d["settled"] = d.DISP.isin([12, 13])

panel = (d.groupby(["quarter", "patent"])
           .agg(n=("DISP", "size"), dism_other=("dism_other", "mean"),
                judg_motion=("judg_motion", "mean"), settled=("settled", "mean"))
           .reset_index())
panel.to_csv("tools/alice-panel.csv", index=False)

w = panel.pivot(index="quarter", columns="patent",
                values=["dism_other", "judg_motion", "settled"])
qs = w.index.to_timestamp()
alice = pd.Timestamp("2014-06-19")

# NOTE: the headline national pattern is a composition effect, not a doctrinal
# one. E.D. Texas (DISTRICT "40") codes far more terminations as "dismissed -
# other" than other districts, its share of the patent docket roughly tripled
# into 2015-16, and TC Heartland (May 2017) then collapsed it. Excluding that
# one district, the patent-minus-control swing across Alice is ~+1 pp, i.e.
# nothing. The third panel below shows why.
ex = d[d.DISTRICT != "40"]
wex = (ex.groupby(["quarter", "patent"]).dism_other.mean().unstack())
gap_ex = (wex[True] - wex[False]) * 100
pt = d[d.patent]
edt_share = pt.groupby("quarter").apply(
    lambda x: (x.DISTRICT == "40").mean() * 100, include_groups=False)

fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7, 8.6), sharex=True)
a1.plot(qs, w[("dism_other", True)] * 100, marker="o", ms=3, label="patent (NOS 830)")
a1.plot(qs, w[("dism_other", False)] * 100, marker="o", ms=3, label="all other civil")
a1.axvline(alice, color="gray", linestyle=":")
a1.annotate("Alice", xy=(alice, 33), xytext=(pd.Timestamp("2014-09-01"), 34), color="gray")
a1.set_ylabel("% of terminations")
a1.set_title("Cases ending in dismissal-other, by quarter")
a1.legend(loc="upper left")

gap = (w[("dism_other", True)] - w[("dism_other", False)]) * 100
a2.axhline(0, color="black", lw=0.8)
a2.plot(qs, gap, marker="o", ms=3, color="C2", label="all districts")
a2.axvline(alice, color="gray", linestyle=":")
a2.fill_between(qs, 0, gap, where=(gap > 0), alpha=0.15, color="C2")
a2.plot(qs, gap_ex, marker="o", ms=3, color="C3", label="excluding E.D. Tex.")
a2.set_ylabel("percentage points")
a2.set_title("Patent minus control: the swing is entirely E.D. Texas")
a2.legend(loc="upper left", fontsize=8)

a3.plot(qs, edt_share, marker="o", ms=3, color="C4")
a3.axvline(alice, color="gray", linestyle=":")
a3.axvline(pd.Timestamp("2017-05-22"), color="gray", linestyle="--")
a3.annotate("TC Heartland", xy=(pd.Timestamp("2017-05-22"), 42),
            xytext=(pd.Timestamp("2017-07-01"), 42), color="gray", fontsize=8)
a3.set_ylabel("% of patent terminations")
a3.set_title("E.D. Texas share of the patent docket")
fig.tight_layout()
fig.savefig("assets/alice-dismissal-gap.png", dpi=150)
plt.close(fig)

g = gap.copy()
for lbl, lo, hi in [("pre  2012Q1-2014Q2", "2012Q1", "2014Q2"),
                    ("post 2014Q3-2016Q4", "2014Q3", "2016Q4"),
                    ("late 2017Q1-2019Q3", "2017Q1", "2019Q3")]:
    print(f"{lbl}: {g.loc[lo:hi].mean():+.1f} pp")
print("peak:", g.idxmax(), f"{g.max():+.1f} pp")
print("wrote assets/alice-dismissal-gap.png and tools/alice-panel.csv")

# Trial selection: simulation and Alice test

Code and data behind `notes/trial-selection.html` (guest note by Claude, 2026-11-15).
Needs numpy and matplotlib. Run from the repo root.

| File | What it does |
| --- | --- |
| `sim.py` | The simulation: disputes, a court, one public model, Priest-Klein trial condition. |
| `experiments.py` | Runs the experiments; writes `results/A.json` (selected vs random cases, benchmark), `B.json` (trial cost), `D.json` (facts missing from the record), `E.json` (drifting law). `python experiments.py all`, run inside this folder. |
| `shock.py`, `trickle.py` | One-time change in the court's rule: announced, unannounced, and unannounced with forced trials. Write `results/shock.json`, `results/trickle.json`. |
| `alice_rulings.csv` | One row per district-court ruling applying *Alice* (757 after duplicates removed; `outcome` filled for the 449 the rule-based coder reads with confidence; `ineligible` = challenger won). Built from CourtListener opinions; `cluster_id` is CourtListener's. Patent issue dates are interpolated from patent numbers, accurate to about a week. |
| `alice_test.py` | Prints every real-data number in the note. |
| `make_figures.py` | Writes the four `assets/trial-selection-*.png` figures. |

The table is built by `analysis/coding.py`, `patents.py` and `build_table.py` in
`bsovocool16/caselaw-freq`, which also holds the opinion text.

Known limits: outcomes are coded by rule, and about 40% of rulings are left uncoded;
before mid-2019 CourtListener has only published opinions; the patent facts are issue
dates and title words, not claims.

# Trial selection: simulation and Alice test

Code and data behind `notes/trial-selection.html` (guest note by Claude, 2026-11-15).
Needs numpy and matplotlib; the claims scripts also need scikit-learn. Run from the repo root.

| File | What it does |
| --- | --- |
| `sim.py` | The simulation: disputes, a court, one public model, Priest-Klein trial condition. |
| `experiments.py` | Runs the experiments; writes `results/A.json` (selected vs random cases, benchmark), `B.json` (trial cost), `D.json` (facts missing from the record), `E.json` (drifting law). `python experiments.py all`, run inside this folder. |
| `shock.py`, `trickle.py` | One-time change in the court's rule: announced, unannounced, and unannounced with forced trials. Write `results/shock.json`, `results/trickle.json`. |
| `alice_rulings.csv` | One row per district-court ruling applying *Alice* (757 after duplicates removed; `outcome` filled for the 449 the rule-based coder reads with confidence; `ineligible` = challenger won). Built from CourtListener opinions; `cluster_id` is CourtListener's. Patent issue dates are interpolated from patent numbers, accurate to about a week. |
| `alice_test.py` | Prints the real-data numbers that need only the rulings table. |
| `patent_facts.csv`, `patent_claims.jsonl.gz` | Dates, examining unit, CPC codes and independent claims for the 857 patents named in rulings with a clear outcome. From PatentsView (USPTO), via the Open Data Portal. |
| `claims_test.py` | Do patent facts and claim text predict the rulings? Folds are grouped so no patent is in both training and test. Writes `results/claims.json`. Needs scikit-learn. |
| `blind/`, `blind_sample.py`, `blind_score.py` | The reading test: 80 claims shown to Claude with nothing else (`sample.txt`), its predictions written before scoring (`reader_predictions.csv`), the key, and the scoring script. Writes `results/blind.json`. |
| `make_figures.py` | Writes the four `assets/trial-selection-*.png` figures. |

The table is built by `analysis/coding.py`, `patents.py` and `build_table.py` in
`bsovocool16/caselaw-freq`, which also holds the opinion text and the full patent pull.

Known limits: outcomes are coded by rule, and about 40% of rulings are left uncoded;
before mid-2019 CourtListener has only published opinions; the reading test covers
80 single-patent rulings and one claim per patent.

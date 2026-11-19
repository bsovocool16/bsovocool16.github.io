"""Draw a blind sample for a reader test: one claim per ruling, no case name, date, court or outcome.

Rulings in the posted collection that name exactly one patent, at most one ruling per group
of rulings sharing a patent. Writes blind/sample.txt (what the reader sees) and
blind/key.csv (which ruling each item is). Usage: python tools/trial-selection/blind_sample.py [n] [seed]
"""
import sys, os, csv, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import claims_test as T
n = int(sys.argv[1]) if len(sys.argv) > 1 else 80; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261119
rng = np.random.default_rng(seed)
cand = [i for i in np.where(T.POST)[0] if len(T.R[i]["pats"]) == 1 and int(T.R[i]["n_patents"]) == 1]
seen, pick = set(), []
for i in rng.permutation(cand):
    if T.GROUP[i] in seen: continue
    seen.add(T.GROUP[i]); pick.append(int(i))
    if len(pick) == n: break
os.makedirs(f"{T.HERE}/blind", exist_ok=True)
with open(f"{T.HERE}/blind/sample.txt", "w") as f, open(f"{T.HERE}/blind/key.csv", "w", newline="") as k:
    w = csv.writer(k); w.writerow(["item", "cluster_id", "patent_number"])
    for j, i in enumerate(pick, 1):
        p = T.R[i]["pats"][0]; cl = sorted(T.CLAIMS[p], key=lambda c: c["claim_number"])
        c = next((c for c in cl if c["independent"]), cl[0]); words = c["text"].split()
        f.write(f"[{j}] " + " ".join(words[:200]) + (" [...]" if len(words) > 200 else "") + "\n\n")
        w.writerow([j, T.R[i]["cluster_id"], p])
print(len(cand), "candidates;", len(pick), "drawn")

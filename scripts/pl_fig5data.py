# SPDX-License-Identifier: Apache-2.0
"""paper-lab: data for Figure 5 (E1): per fork, mean Q_c of the resolved and of the failed run of the same boundary task,
95 % CIs by bootstrap over tasks (tasks with both cells at that fork). Writes $PL_WORK/paper_lab/r3/fig5_data.json."""
import json, random, statistics as st, sys, os
WORK = os.environ.get("PL_WORK", "/work")  # data root; set PL_WORK to run off-box (e.g. on a local copy of the backup)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, WORK + "/paper_lab"); sys.argv = ["x"]
import pl_e1 as E

P = E.PAIRS
keep = {(p[c], p["iid"]) for p in P for c in ("res", "fail")}
D, _ = E.load(["samples_e1a.jsonl", "samples_e1b.jsonl", "samples_prefill.jsonl", "samples_prefill_g.jsonl"], keep)


def q(r, f):
    return st.mean(D[r][f]) if D.get(r, {}).get(f) else None


out = {}
for f in (6, 10, 14, "own"):
    rows = [(q((p["res"], p["iid"]), f), q((p["fail"], p["iid"]), f)) for p in P]
    rows = [r for r in rows if r[0] is not None and r[1] is not None]
    rnd = random.Random(0); bs = {0: [], 1: []}
    for _ in range(4000):
        smp = [rnd.choice(rows) for _ in rows]
        for i in (0, 1):
            bs[i].append(st.mean(s[i] for s in smp))
    cell = {}
    for i, lab in ((0, "resolved"), (1, "failed")):
        v = sorted(bs[i]); cell[lab] = [round(st.mean(r[i] for r in rows), 3), round(v[100], 3), round(v[3899], 3)]
    out[str(f)] = {"n": len(rows), **cell}
json.dump(out, open(WORK + "/paper_lab/r3/fig5_data.json", "w"), indent=1)
print(json.dumps(out))

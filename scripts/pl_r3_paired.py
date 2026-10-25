# SPDX-License-Identifier: Apache-2.0
"""paper-lab: paired forced-commit comparisons. (1) within run: fork k vs own commit point (any samples file);
(2) across models on the same runs/forks: 31B (samples_prefill.jsonl) vs 12B (samples_prefill_12b.jsonl). Read-only."""
import json, hashlib, collections, statistics as st, random, sys
import os
WORK = os.environ.get("PL_WORK", "/work")  # data root; set PL_WORK to run off-box
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, WORK + "/paper_lab"); sys.argv = ["x"]
import pl_r3 as m
G = m.load_grades()

def load(fn):
    D = collections.defaultdict(lambda: collections.defaultdict(list)); cls = {}
    for l in open(WORK + "/paper_lab/r3/" + fn):
        x = json.loads(l)
        if "iid" not in x or x.get("skip"):
            continue
        if x.get("patch"):
            g = G.get(hashlib.sha1((x["iid"] + "\n" + x["patch"]).encode()).hexdigest()[:12])
            if g is None:
                continue
            r = int(g["resolved"])
        else:
            r = 0
        key = (x["tag"], x["iid"]); D[key]["nat" if x["natural"] else x["k"]].append(r); cls[key] = x["cls"]
    return D, cls

def ci(rows):
    """95 % percentile CI of the mean diff, bootstrap clustered by TASK (rows: (task, diff)); red-team M2 (was: runs)."""
    by = collections.defaultdict(list)
    for t, d in rows:
        by[t].append(d)
    tasks = list(by); rnd = random.Random(0); bs = []
    for _ in range(4000):
        smp = [d for t in (rnd.choice(tasks) for _ in tasks) for d in by[t]]
        bs.append(st.mean(smp))
    bs.sort(); return bs[100], bs[3899]

def within(fn, label):
    D, cls = load(fn)
    for c in ("res", "wrongG", "empty"):
        for k in (6, 10, 14):
            pr = [(st.mean(d[k]), st.mean(d["nat"]), key[1]) for key, d in D.items() if cls[key] == c and k in d and "nat" in d]
            if len(pr) >= 3:
                diff = [b - a for a, b, _ in pr]; lo, hi = ci([(t, b - a) for a, b, t in pr])
                print(f"{label} {c:7s} k={k:2d}->own: n={len(pr):2d} {st.mean(a for a,_,_ in pr):.2f} -> {st.mean(b for _,b,_ in pr):.2f}  diff {st.mean(diff):+.2f} [{lo:+.2f},{hi:+.2f}]")

within("samples_prefill.jsonl", "R3f")
within("samples_prefill_g.jsonl", "R3g")
import os
if os.path.exists(WORK + "/paper_lab/r3/samples_prefill_qwen.jsonl"):
    within("samples_prefill_qwen.jsonl", "R6 Qwen")
A, ca = load("samples_prefill.jsonl"); B, cb = load("samples_prefill_12b.jsonl")
for c in ("res", "wrongG", "empty"):
    for k in (10, 14, "nat"):
        pr = [(st.mean(A[key][k]), st.mean(B[key][k]), key[1]) for key in A if ca[key] == c and key in B and k in A[key] and k in B[key]]
        if len(pr) >= 3:
            diff = [b - a for a, b, _ in pr]; lo, hi = ci([(t, b - a) for a, b, t in pr])
            print(f"31B vs 12B {c:7s} k={str(k):>3s}: n={len(pr):2d} 31B {st.mean(a for a,_,_ in pr):.2f}  12B {st.mean(b for _,b,_ in pr):.2f}  diff {st.mean(diff):+.2f} [{lo:+.2f},{hi:+.2f}]")

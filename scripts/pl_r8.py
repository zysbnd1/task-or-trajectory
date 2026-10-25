# SPDX-License-Identifier: Apache-2.0
"""paper-lab R8: explorer x committer swap. For each (explorer run, fork) graded under both committers, compare the
same-model commit (explorer commits from its own context) with the swapped commit (the other model commits from that
context). Paired by (run, fork); 95 % CI by task-clustered bootstrap. Read-only; prints a table."""
import json, hashlib, collections, statistics as st, random, sys
import os
WORK = os.environ.get("PL_WORK", "/work")  # data root; set PL_WORK to run off-box
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, WORK + "/paper_lab"); sys.argv = ["x"]
import pl_r3 as m
G = m.load_grades()

def load(fn):
    D = collections.defaultdict(list); cls = {}
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
        key = (x["tag"], x["iid"], "own" if x["natural"] else x["k"]); D[key].append(r); cls[key] = x["cls"]
    return D, cls

def ci(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    tasks = list(by); rnd = random.Random(0); vals = []
    for _ in range(4000):
        smp = [r for t in (rnd.choice(tasks) for _ in tasks) for r in by[t]]
        vals.append(st.mean(b - a for _, a, b in smp))
    vals.sort(); return vals[100], vals[3899]

for explorer, own_fn, swap_fn, own_name, swap_name, forks in (
        ("Gemma 4 31B", "samples_prefill_g.jsonl", "samples_x_qg.jsonl", "Gemma", "Qwen", ("own", 10)),
        ("Qwen3.6-27B", "samples_prefill_qwen.jsonl", "samples_x_gq.jsonl", "Qwen", "Gemma", ("own", 10)),
        ):
    try:
        A, ca = load(own_fn); B, _ = load(swap_fn)
    except FileNotFoundError as e:
        print(explorer, "missing", e); continue
    print(f"== explorer {explorer}: {own_name} commits (own) vs {swap_name} commits (swap), same contexts")
    for c in ("res", "wrongG", "empty"):
        for fork in forks:
            rows = [(k[1], st.mean(A[k]), st.mean(B[k])) for k in A if ca[k] == c and k[2] == fork and k in B]
            if len(rows) < 3:
                continue
            lo, hi = ci(rows)
            print(f"  {c:7s} at {str(fork):>3s}: n={len(rows):2d} runs  {own_name} {st.mean(a for _, a, _ in rows):.2f}  "
                  f"{swap_name} {st.mean(b for _, _, b in rows):.2f}  diff {st.mean(b - a for _, a, b in rows):+.2f} [{lo:+.2f},{hi:+.2f}]")

# SPDX-License-Identifier: Apache-2.0
"""paper-lab: recompute headline numbers from the release slice alone (no box internals)."""
import gzip, json, collections, os, statistics as st, sys
D = sys.argv[1]


def jopen(name):
    """Open a slice file as text: X.jsonl.gz or X.jsonl, whichever exists (Kaggle serves uploaded .gz files decompressed)."""
    for p in (name, name[:-3] if name.endswith(".gz") else name + ".gz"):
        if os.path.exists(f"{D}/{p}"):
            return gzip.open(f"{D}/{p}", "rt") if p.endswith(".gz") else open(f"{D}/{p}")
    raise FileNotFoundError(f"{D}/{name}")


jexists = lambda name: any(os.path.exists(f"{D}/{p}") for p in (name, name[:-3] if name.endswith(".gz") else name + ".gz"))
fc = [json.loads(l) for l in jopen("forced_commits.jsonl.gz")]
q = collections.defaultdict(lambda: [0, 0])
for x in fc:
    if x["resolved"] is None or x["mode"] == "instruct":
        continue
    k = "own" if x["own_commit_point"] else x["fork_call"]
    a = q[(x["study"], x["class"], k)]; a[0] += x["resolved"]; a[1] += 1
for (s, c, k), (r, n) in sorted(q.items(), key=lambda t: (t[0][0], t[0][1], str(t[0][2]))):
    if k in (6, "own"):
        print(f"{s[:42]:42s} {c:7s} {str(k):>3s}: Q_c {r / n:.2f} (n={n})")
runs = {(r["explorer_run"], r["task"]): r for r in map(json.loads, open(f"{D}/runs.jsonl"))}
ev = collections.defaultdict(list)
for x in map(json.loads, jopen("evergreen_edits.jsonl.gz")):
    if x["cumulative_patch"]:  # "first applied edit" = first non-empty cumulative patch (as in the paper)
        ev[(x["pool"], x["run"], x["task"])].append((x["edit_index"], x["resolved"], x["class"]))
agg = collections.Counter()
for (pool, run, task), v in ev.items():
    meta = runs.get((run, task), {})
    if not meta.get("replay_reproducible"):
        continue
    v.sort(); res = [r for _, r, _ in v]; cls = v[0][2]
    if None in res:
        continue
    if cls == "res":
        agg[(pool, "succ")] += 1; agg[(pool, "succ_first")] += bool(res[0])
    elif cls in ("wrongG", "wrongL"):
        agg[(pool, "fail")] += 1; agg[(pool, "fail_ever")] += any(res)
for pool in sorted({p for p, _ in agg}):
    print(f"{pool:20s} successes green at first edit {agg[(pool,'succ_first')]}/{agg[(pool,'succ')]}; failed runs ever green {agg[(pool,'fail_ever')]}/{agg[(pool,'fail')]}")

# ---- added 2026-10-22 after the red-team review: everything below also recomputes from the slice alone ----
import math, random


def cp_upper(x, n, a=0.05):
    """Exact Clopper-Pearson upper bound (two-sided 95 %)."""
    if x >= n:
        return 1.0
    lo, hi = x / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(x + 1))
        lo, hi = (mid, hi) if cdf > a / 2 else (lo, mid)
    return hi


allruns = list(map(json.loads, open(f"{D}/runs.jsonl")))
# Table 1: runs and resolved per replay pool
t1 = collections.defaultdict(collections.Counter)
for r in allruns:
    if r.get("pool"):
        t1[r["pool"]][r["class"]] += 1
for pool, c in sorted(t1.items()):
    n = sum(c.values()); print(f"Table 1 {pool:20s} runs {n}, resolved {c['res']} ({c['res'] / n:.0%})")
# tree counts: distinct (task, cumulative patch) over all replayed runs vs replay-reproducible runs only
trees_all, trees_rep = set(), set()
for x in map(json.loads, jopen("evergreen_edits.jsonl.gz")):
    if x["cumulative_patch"]:
        trees_all.add((x["task"], x["cumulative_patch"]))
        if runs.get((x["run"], x["task"]), {}).get("replay_reproducible"):
            trees_rep.add((x["task"], x["cumulative_patch"]))
print(f"distinct intermediate trees: {len(trees_all)} over all replayed runs; {len(trees_rep)} over replay-reproducible runs")
# post-green share (resolved, reproducible): (agent_calls - 1 - call of first passing tree) / agent_calls, median per pool
first_green = {}
for x in map(json.loads, jopen("evergreen_edits.jsonl.gz")):
    if x["resolved"] and x["cumulative_patch"]:
        key = (x["pool"], x["run"], x["task"])
        first_green[key] = min(first_green.get(key, 10 ** 9), x["call"])
share = collections.defaultdict(list)
for (pool, run, task), k0 in first_green.items():
    meta = runs.get((run, task), {})
    if meta.get("class") == "res" and meta.get("replay_reproducible") and meta.get("agent_calls"):
        N = meta["agent_calls"]; share[pool].append((N - 1 - k0) / N)
for pool, v in sorted(share.items()):
    print(f"post-green share {pool:20s} median {st.median(v):.2f} (n={len(v)})")
# Table 4: committer swap, paired by (explorer run, task, fork), task-clustered bootstrap
SW = {"Gemma explorer": ("R3g Gemma-31B prefill", "R8 Qwen commits from Gemma-31B contexts"),
      "Qwen explorer": ("R6 Qwen3.6-27B prefill", "R8 Gemma-31B commits from Qwen contexts")}
cell = collections.defaultdict(list); ccls = {}
for x in fc:
    if x["resolved"] is None or x["mode"] == "instruct":
        continue
    k = (x["study"], x["explorer_run"], x["task"], "own" if x["own_commit_point"] else x["fork_call"])
    cell[k].append(x["resolved"]); ccls[k[1:]] = x["class"]
for lab, (own_s, swap_s) in SW.items():
    for c in ("res", "wrongG", "empty"):
        rows = [(k[2], st.mean(v), st.mean(cell[(swap_s,) + k[1:]])) for k, v in cell.items()
                if k[0] == own_s and k[3] == "own" and ccls[k[1:]] == c and (swap_s,) + k[1:] in cell]
        if len(rows) < 3:
            continue
        by = collections.defaultdict(list)
        for t, a, b in rows:
            by[t].append(b - a)
        tasks = list(by); rnd = random.Random(0); bs = []
        for _ in range(4000):
            bs.append(st.mean(d for t in (rnd.choice(tasks) for _ in tasks) for d in by[t]))
        bs.sort()
        print(f"Table 4 {lab} {c:7s} own point: n={len(rows)} runs  own {st.mean(a for _, a, _ in rows):.2f}  swap "
              f"{st.mean(b for _, _, b in rows):.2f}  diff {st.mean(b - a for _, a, b in rows):+.2f} [{bs[100]:+.2f}, {bs[3899]:+.2f}]")
# run-level sealing counts with exact 95 % upper bounds (failed classes; own point and any fork)
rl = collections.defaultdict(lambda: collections.defaultdict(list))
for x in fc:
    if x["resolved"] is None or x["mode"] == "instruct" or x["class"] not in ("wrongG", "wrongL", "empty"):
        continue
    rl[(x["study"], x["class"])][(x["explorer_run"], x["task"])].append((x["own_commit_point"], x["resolved"]))
for (s, c), R in sorted(rl.items()):
    own_runs = [r for r, v in R.items() if any(o for o, _ in v)]
    own_pass = sum(1 for r in own_runs if any(res for o, res in R[r] if o))
    any_pass = sum(1 for v in R.values() if any(res for _, res in v))
    print(f"run-level {s[:40]:40s} {c:7s} own point {own_pass}/{len(own_runs)} (95% upper {cp_upper(own_pass, len(own_runs)):.3f}); "
          f"any fork {any_pass}/{len(R)} (upper {cp_upper(any_pass, len(R)):.3f})")

# ---- added 2026-10-23 (v0.2): Table 2 (E5) and Table 4 (E1), plus the window-7 controls, from the slice alone ----
import os
if os.path.exists(f"{D}/task_outcomes.jsonl"):
    to = collections.defaultdict(lambda: collections.defaultdict(dict))  # pool -> task -> seed_group -> resolved
    for x in map(json.loads, open(f"{D}/task_outcomes.jsonl")):
        to[x["pool"]][x["task"]][x["seed_group"]] = x["class"] == "res"
    for pool, T in sorted(to.items()):
        T = {t: v for t, v in T.items() if len(v) >= 2}
        ps = {t: sum(v.values()) / len(v) for t, v in T.items()}
        pbar = st.mean(ps.values())
        between = st.mean((p - pbar) ** 2 for p in ps.values()) - st.mean(ps[t] * (1 - ps[t]) / (len(T[t]) - 1) for t in ps)
        within = st.mean(p * (1 - p) * len(T[t]) / (len(T[t]) - 1) for t, p in ps.items())
        kmax = min(len(v) for v in T.values()); kk = 4 if kmax < 8 else 8
        pk = lambda k: st.mean(1.0 if len(T[t]) - round(ps[t] * len(T[t])) < k else
                               1 - math.comb(len(T[t]) - round(ps[t] * len(T[t])), k) / math.comb(len(T[t]), k) for t in T)
        print(f"Table 2 {pool:5s}: between-task share {between / (between + within):.0%}; boundary tasks "
              f"{sum(0 < p < 1 for p in ps.values())} of {len(ps)}; pass@1 {pk(1):.2f} -> pass@{kk} {pk(kk):.2f}")
if os.path.exists(f"{D}/e1_pairs.json"):
    pairs = json.load(open(f"{D}/e1_pairs.json"))
    E1S = ("R3f Gemma-31B prefill", "R3g Gemma-31B prefill", "E1 Gemma-31B boundary pairs")
    cellq = collections.defaultdict(list)
    for x in fc:
        if x["resolved"] is not None and x["study"] in E1S:
            cellq[(x["explorer_run"], x["task"], "own" if x["own_commit_point"] else x["fork_call"])].append(x["resolved"])
    for f in (6, 10, 14, "own"):
        rows = []
        for p in pairs:
            a, b = cellq.get((p["resolved_run"], p["task"], f)), cellq.get((p["failed_run"], p["task"], f))
            if a and b:
                rows.append((st.mean(a), st.mean(b)))
        rnd = random.Random(0)
        bs = sorted(st.mean(r[0] - r[1] for r in (rnd.choice(rows) for _ in rows)) for _ in range(4000))
        print(f"Table 4 fork {str(f):>3s} ({len(rows)} tasks): resolved {st.mean(r[0] for r in rows):.2f}  failed "
              f"{st.mean(r[1] for r in rows):.2f}  diff {st.mean(r[0] - r[1] for r in rows):+.2f} [{bs[100]:+.2f}, {bs[3899]:+.2f}]")
    for study in ("E1 ablation: no injected instruction", "E3 failed boundary runs at T = 1.0"):
        v = [x["resolved"] for x in fc if x["study"] == study and x["resolved"] is not None]
        if v:
            print(f"{study}: {len(v)} episodes, pass rate {st.mean(v):.2f}")

# ---- added 2026-10-25 (v0.7 verification pass): every remaining table and the §4.3 estimands, from the slice alone ----
print("\n== v0.7 additions ==")
pre = [x for x in fc if x["mode"].startswith("prefill")]
print(f"prefill (forced-commit) episodes: {len(pre)}, graded {sum(x['resolved'] is not None for x in pre)}")
# Table 1 caption: dev tasks run by all three models
byp = collections.defaultdict(lambda: collections.defaultdict(list))
for r in allruns:
    if r.get("pool"):
        byp[r["pool"]][r["task"]].append(r["class"] == "res")
common = set(byp["Gemma-31B dev"]) & set(byp["Qwen3.6-27B dev"]) & set(byp["DeepSeek-V4-Pro dev"])
print(f"Table 1 caption: {len(common)} dev tasks run by all three: " + ", ".join(
    f"{p.split()[0]} {sum(sum(byp[p][t]) for t in common)}/{sum(len(byp[p][t]) for t in common)}"
    for p in ("Gemma-31B dev", "Qwen3.6-27B dev", "DeepSeek-V4-Pro dev")))
# shell-edit exclusion (replay not reproducible) by pool and class
ex = collections.defaultdict(lambda: [0, 0])
for r in allruns:
    if r.get("pool"):
        e = ex[(r["pool"], r["class"])]; e[0] += 1; e[1] += not r["replay_reproducible"]
print(f"shell-edit exclusion: {sum(v[1] for v in ex.values())} of {sum(v[0] for v in ex.values())} runs; " + "; ".join(
    f"{p.split()[0]} {c} {v[1]}/{v[0]}" for (p, c), v in sorted(ex.items()) if c in ("empty", "res")))
# mixed tasks with the first two seed groups vs all seeds (Table 2 text)
if os.path.exists(f"{D}/task_outcomes.jsonl"):
    first2 = {"train": ("s0", "traj_s1"), "dev": ("r1_dev_b0", "r1_dev_b0b")}
    m2 = sum(1 for pool, T in to.items() for t, v in T.items() if all(g in v for g in first2[pool])
             and 0 < sum(v[g] for g in first2[pool]) < 2)
    mall = sum(1 for T in to.values() for v in T.values() if len(v) >= 2 and 0 < sum(v.values()) < len(v))
    print(f"mixed tasks: {m2} with the first two seeds, {mall} with all seeds")
# Table 3 rows 3-4 and post-green composition from per-call kind labels (call_kinds.jsonl, labels only)
if os.path.exists(f"{D}/call_kinds.jsonl"):
    KD = {(x["run"], x["task"]): x["kinds"] for x in map(json.loads, open(f"{D}/call_kinds.jsonl"))}
    comp = collections.defaultdict(collections.Counter); anytest = collections.defaultdict(list)
    for (pool, run, task), k0 in first_green.items():
        meta = runs.get((run, task), {})
        if meta.get("class") == "res" and meta.get("replay_reproducible") and (run, task) in KD:
            after = [k for c in KD[(run, task)][k0 + 1:] for k in c]
            comp[pool].update(after); anytest[pool].append("test" in after)
    for pool in sorted(comp):
        tot = sum(comp[pool].values())
        print(f"post-green {pool:20s} runs testing after green {st.mean(anytest[pool]):.0%} (n={len(anytest[pool])}); calls: " +
              ", ".join(f"{k} {comp[pool][k] / tot:.0%}" for k in ("read-search", "test", "exec", "submit", "edit", "other")))
# Table 5: own-point Q_c by failure type and task band (runs with an own commit point).
#   unsolved (p = 0): the selected runs of R3f/R3g; boundary: the failed runs of the E1 pairs (one per task)
ptask = {t: sum(v.values()) / len(v) for T in to.values() for t, v in T.items() if len(v) >= 2}
own_cell = collections.defaultdict(list); any_cell = collections.defaultdict(list)
for x in fc:
    if x["resolved"] is not None and x["study"] in E1S:
        any_cell[(x["explorer_run"], x["task"])].append(x["resolved"])
        if x["own_commit_point"]:
            own_cell[(x["explorer_run"], x["task"])].append((x["resolved"], x["class"]))
t5 = collections.defaultdict(list)
r3runs = {(x["explorer_run"], x["task"]): x["class"] for x in fc if x["study"] in ("R3f Gemma-31B prefill", "R3g Gemma-31B prefill")}
for key, cls in r3runs.items():
    if ptask.get(key[1]) == 0 and cls in ("wrongG", "wrongL", "empty"):
        kind = "wrong" if cls != "empty" else "empty"
        t5[("unsolved", kind, "any")].append(any(any_cell[key]))
        if own_cell.get(key):
            t5[("unsolved", kind)].append(st.mean(r for r, _ in own_cell[key]))
for p in pairs:
    key = (p["failed_run"], p["task"])
    if own_cell.get(key):
        kind = "wrong" if own_cell[key][0][1] != "empty" else "empty"
        t5[("boundary", kind)].append(st.mean(r for r, _ in own_cell[key]))
for k in sorted(t5):
    v = t5[k]
    print(f"Table 5 {k}: " + (f"{sum(v)} of {len(v)} runs pass a forced commit at any fork" if "any" in k else
                               f"Q_c {st.mean(v):.2f} ({len(v)} runs)"))
# Qwen sealing counts
qg = [x["resolved"] for x in fc if x["study"].startswith("R8 Qwen") and x["own_commit_point"] and x["class"] != "res"
      and x["class"] != "empty" and ptask.get(x["task"]) == 0]
q6 = [x["resolved"] for x in fc if x["study"].startswith("R6 Qwen") and x["class"] in ("wrongG", "wrongL")]
print(f"Qwen from Gemma's wrong-patch contexts on unsolved tasks, own point: {sum(qg)}/{len(qg)}; "
      f"Qwen's own wrong-patch contexts, all forks: {sum(q6)}/{len(q6)}")
# instruction ablation, paired by run at the own point (task-clustered bootstrap)
ab = collections.defaultdict(lambda: collections.defaultdict(list))
for x in fc:
    if x["own_commit_point"] and x["resolved"] is not None and x["mode"] == "prefill" and x.get("temperature") is None:
        arm = "no" if x["study"].startswith("E1 ablation") else ("yes" if x["study"] in E1S else None)
        if arm:
            ab[(x["explorer_run"], x["task"])][arm].append(x["resolved"])
abr = [(k[1], st.mean(v["no"]) - st.mean(v["yes"])) for k, v in ab.items() if v["no"] and v["yes"]]
byt = collections.defaultdict(list)
for t, d in abr:
    byt[t].append(d)
tk = list(byt); rnd = random.Random(0)
bs = sorted(st.mean(d for t in (rnd.choice(tk) for _ in tk) for d in byt[t]) for _ in range(4000))
print(f"no-instruction ablation: {st.mean(d for _, d in abr):+.3f} [{bs[100]:+.3f}, {bs[3899]:+.3f}] over {len(abr)} runs / {len(tk)} tasks")
# timing: failed run's value at its last fork before the resolved run's commit call, and at its own commit point
ownk = {(x["explorer_run"], x["task"]): x["fork_call"] for x in fc if x["own_commit_point"] and x["study"] in E1S}
before, own_f = [], []
for p in pairs:
    kr = ownk.get((p["resolved_run"], p["task"]))
    if kr is None or not cellq.get((p["resolved_run"], p["task"], "own")):
        continue
    forks = [f for f in (14, 10, 6) if f <= kr and cellq.get((p["failed_run"], p["task"], f))]
    if forks:
        before.append(st.mean(cellq[(p["failed_run"], p["task"], forks[0])]))
    if cellq.get((p["failed_run"], p["task"], "own")):
        own_f.append(st.mean(cellq[(p["failed_run"], p["task"], "own")]))
print(f"timing: failed run at its last fork before the resolved commit call {st.mean(before):.2f} (n={len(before)}); "
      f"at its own commit point {st.mean(own_f):.2f} (n={len(own_f)})")
# §4.3 estimands (never-edited failed runs excluded): p(1-p)-weighted D, split-draw calibration, Cauchy-Schwarz bound
NR = {p["task"]: p["n_runs"] for p in pairs}; PP = {p["task"]: p["p"] for p in pairs}
draws = collections.defaultdict(list)  # (run, task) -> own-point draws in slice order, and seeded sample index
for x in fc:
    if x["resolved"] is not None and x["study"] in E1S and x["own_commit_point"]:
        draws[(x["explorer_run"], x["task"])].append((x["sample"], x["resolved"]))
V = lambda t: PP[t] * (1 - PP[t]) * NR[t] / (NR[t] - 1)
rowsD = []
for p in pairs:
    a, b = draws.get((p["resolved_run"], p["task"])), draws.get((p["failed_run"], p["task"]))
    if a and b:
        ma, mb = st.mean(r for _, r in a), st.mean(r for _, r in b)
        ua = ma * (1 - ma) * len(a) / (len(a) - 1) if len(a) > 1 else 0.0
        ub = mb * (1 - mb) * len(b) / (len(b) - 1) if len(b) > 1 else 0.0
        t = p["task"]
        rowsD.append({"V": V(t), "cov": V(t) * (ma - mb), "e": PP[t] * ua + (1 - PP[t]) * ub})
def dw(R):
    sV = sum(r["V"] for r in R); return sum(r["cov"] for r in R) / sV, sum(0.25 - r["e"] for r in R) / sV
Dw, Rc = dw(rowsD)
rnd = random.Random(0); bsD = [dw([rnd.choice(rowsD) for _ in rowsD]) for _ in range(4000)]
varD = st.pvariance([b[0] for b in bsD]); lbs = sorted(b[0] ** 2 / b[1] for b in bsD)
print(f"estimands ({len(rowsD)} tasks, never-edited failed runs excluded): D (p(1-p)-weighted) {Dw:.3f}; "
      f"Cauchy-Schwarz bound D^2/R {Dw ** 2 / Rc:.3f}, bias-corrected {(Dw ** 2 - varD) / Rc:.3f}, one-sided 95 % limit {lbs[200]:.3f}")
for lab, first in (("first draw = first in slice (completion) order", lambda v: v[0][1]),
                   ("first draw = seeded sample 0", lambda v: min(v)[1])):
    acc = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}
    for p in pairs:
        for c, y, w in (("resolved_run", 1, PP[p["task"]]), ("failed_run", 0, 1 - PP[p["task"]])):
            v = draws.get((p[c], p["task"]))
            if v and len(v) >= 2:
                f0 = first(v); rest = [r for i, (_, r) in enumerate(v) if i != (0 if lab.startswith("first draw = first in") else v.index(min(v)))]
                a = acc[f0]; a[0] += w; a[1] += w * y; a[2] += w * st.mean(rest)
    print(f"calibration ({lab}): first fail -> forced {acc[0][2] / acc[0][0]:.2f} natural {acc[0][1] / acc[0][0]:.2f}; "
          f"first pass -> forced {acc[1][2] / acc[1][0]:.2f} natural {acc[1][1] / acc[1][0]:.2f}")
# seeded split-draw calibration with task-bootstrap CIs, and the average calibration (forced vs natural)
def calib_seeded(P_):
    acc = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}
    for p in P_:
        for c, y, w in (("resolved_run", 1, PP[p["task"]]), ("failed_run", 0, 1 - PP[p["task"]])):
            v = sorted(draws.get((p[c], p["task"])) or [])
            if len(v) >= 2:
                a = acc[v[0][1]]; a[0] += w; a[1] += w * y; a[2] += w * st.mean(r for _, r in v[1:])
    if min(acc[0][0], acc[1][0]) == 0:
        return None
    return acc[0][2] / acc[0][0], acc[0][1] / acc[0][0], acc[1][2] / acc[1][0], acc[1][1] / acc[1][0]
pt_ = calib_seeded(pairs); rnd = random.Random(0)
bc = [b for b in (calib_seeded([rnd.choice(pairs) for _ in pairs]) for _ in range(2000)) if b]
ci_ = lambda i: (sorted(b[i] for b in bc)[int(0.025 * len(bc))], sorted(b[i] for b in bc)[int(0.975 * len(bc)) - 1])
print("calibration (seeded sample 0), 95 % CI over tasks: " + "; ".join(
    f"{lab} {pt_[i]:.2f} [{ci_(i)[0]:.2f}, {ci_(i)[1]:.2f}]" for i, lab in
    enumerate(("forced|first fail", "natural|first fail", "forced|first pass", "natural|first pass"))))
avg = [(PP[p["task"]] * st.mean(r for _, r in draws[(p["resolved_run"], p["task"])]) + (1 - PP[p["task"]]) *
        st.mean(r for _, r in draws[(p["failed_run"], p["task"])]), PP[p["task"]]) for p in pairs
       if draws.get((p["resolved_run"], p["task"])) and draws.get((p["failed_run"], p["task"]))]
print(f"average calibration ({len(avg)} tasks): implied forced value {st.mean(a for a, _ in avg):.3f} vs natural solve rate {st.mean(b for _, b in avg):.3f}")
# TOST at call 6 (90 % CI of the paired difference) and the gain from call 6 to the own commit point (both runs)
rows6 = [(p["task"], st.mean(cellq[(p["resolved_run"], p["task"], 6)]) - st.mean(cellq[(p["failed_run"], p["task"], 6)]))
         for p in pairs if cellq.get((p["resolved_run"], p["task"], 6)) and cellq.get((p["failed_run"], p["task"], 6))]
rnd = random.Random(0); bs6 = sorted(st.mean(d for _, d in (rnd.choice(rows6) for _ in rows6)) for _ in range(4000))
print(f"call 6 TOST: diff {st.mean(d for _, d in rows6):+.3f}, 90 % CI [{bs6[200]:+.3f}, {bs6[3799]:+.3f}] over {len(rows6)} tasks "
      f"(equivalent within ±0.15: {bs6[200] > -0.15 and bs6[3799] < 0.15})")
gain = collections.defaultdict(list)
for p in pairs:
    for c in ("resolved_run", "failed_run"):
        a, b = cellq.get((p[c], p["task"], 6)), cellq.get((p[c], p["task"], "own"))
        if a and b:
            gain[p["task"]].append(st.mean(b) - st.mean(a))
gk = list(gain); rnd = random.Random(0)
bsg = sorted(st.mean(d for t in (rnd.choice(gk) for _ in gk) for d in gain[t]) for _ in range(4000))
print(f"call 6 -> own commit point, both runs: {st.mean(d for v in gain.values() for d in v):+.2f} [{bsg[100]:+.2f}, {bsg[3899]:+.2f}] "
      f"over {sum(len(v) for v in gain.values())} runs / {len(gk)} tasks")

# ---- added 2026-10-25 (v0.8, after the adversarial review): instrument diagnostics, exclusions, probes ----
print("\n== v0.8 additions ==")
def cp_up(x, n):
    return cp_upper(x, n) if n else float("nan")
# (M4) forced-draw agreement at the own commit point (3-draw cells, T as logged)
c3 = collections.defaultdict(list)
for x in fc:
    if x["resolved"] is not None and x["mode"].startswith("prefill") and x["own_commit_point"] and x.get("temperature") is None:
        c3[(x["study"], x["explorer_run"], x["task"])].append(x["resolved"])
k3 = [v for v in c3.values() if len(v) == 3]
print(f"own-point cells with K = 3: {len(k3)}; all three forced draws agree in {sum(len(set(v)) == 1 for v in k3) / len(k3):.1%}")
# (M6) ever-green exclusions: why the logged patch is not reproduced, and passing trees among excluded runs
evr = collections.defaultdict(list)
for x in map(json.loads, jopen("evergreen_edits.jsonl.gz")):
    evr[(x["run"], x["task"])].append(x)
for cls in ("empty", "wrong", "res"):
    sub = [r for r in allruns if r.get("pool") and not r["replay_reproducible"] and
           (r["class"] in ("wrongG", "wrongL") if cls == "wrong" else r["class"] == cls)]
    ecalls = lambda r: [i for i, c in enumerate(r["tools_per_call"]) if any(t in ("edit_file", "write_file") for t in c)]
    def refused(r):  # harness rule (swegemma check_budget): a counted call is refused once 28 are used; submit_patch/get_status are not counted
        n, out = 0, []
        for c in r["tools_per_call"]:
            for t in c:
                if t in ("edit_file", "write_file"):
                    out.append(n >= 28)
                n += t not in ("submit_patch", "get_status")
        return out
    post = sum(1 for r in sub if refused(r) and all(refused(r)))  # every edit refused by the spent tool-call budget
    green = [r for r in sub if any(e["resolved"] for e in evr.get((r["explorer_run"], r["task"]), []))]
    print(f"not reproduced, {cls:5s}: {len(sub)} runs; use edit_file/write_file {sum(bool(ecalls(r)) for r in sub)}; all edits refused by the "
          f"spent 28-call budget: {post}; hold a passing replayed tree: {len(green)}" + (f" {dict(collections.Counter(r['pool'] for r in green))}" if green and cls == "empty" else ""))
sub = [r for r in allruns if r.get("pool") and not r["replay_reproducible"] and r["class"] == "empty"]
lost = [r for r in sub if any(e["resolved"] for e in evr.get((r["explorer_run"], r["task"]), []))]
print(f"empty-final runs holding a passing replayed tree whose edits were all refused by the spent budget: "
      f"{sum(1 for r in lost if refused(r) and all(refused(r)))} of {len(lost)}")
# (M8) applicability vs quality from call 6 to the own commit point (E1 pairs; runs with both cells)
c2 = collections.defaultdict(list)
for x in fc:
    if x["resolved"] is not None and x["study"] in E1S:
        c2[(x["explorer_run"], x["task"], "own" if x["own_commit_point"] else x["fork_call"])].append((bool(x["patch"]), x["resolved"]))
for lab, sel in (("both runs", ("resolved_run", "failed_run")), ("resolved runs", ("resolved_run",)), ("failed runs", ("failed_run",))):
    out = []
    for f in (6, "own"):
        v = [e for p in pairs for c in sel if c2.get((p[c], p["task"], 6)) and c2.get((p[c], p["task"], "own")) for e in c2[(p[c], p["task"], f)]]
        ne = [r for a, r in v if a]
        out.append(f"@{f}: applies {sum(a for a, _ in v) / len(v):.2f}, pass|applies {sum(ne) / len(ne):.2f}, Q_c {sum(r for _, r in v) / len(v):.2f}")
    print(f"call 6 -> own, {lab:13s} " + " | ".join(out))
# (M7) committer swaps on resolved contexts: empty (non-applying) forced patches and pass rate given a patch
for own_s, swap_s in (("R3g Gemma-31B prefill", "R8 Qwen commits from Gemma-31B contexts"), ("R6 Qwen3.6-27B prefill", "R8 Gemma-31B commits from Qwen contexts")):
    for s in (own_s, swap_s):
        v = [x for x in fc if x["study"] == s and x["own_commit_point"] and x["class"] == "res" and x["resolved"] is not None]
        ne = [x for x in v if x["patch"]]
        print(f"swap cell {s[:42]:42s} resolved contexts: empty patch {1 - len(ne) / len(v):.2f}, pass|patch {sum(x['resolved'] for x in ne) / len(ne):.2f} ({len(v)} episodes)")
# (m1) run-level sealing counts with exact 95 % upper bounds
def runs_any(study_pred, cls_pred, own_only, task_pred=lambda t: True):
    R = collections.defaultdict(list)
    for x in fc:
        if x["resolved"] is not None and study_pred(x["study"]) and cls_pred(x["class"]) and task_pred(x["task"]) and (x["own_commit_point"] or not own_only):
            R[(x["explorer_run"], x["task"])].append(x["resolved"])
    return sum(any(v) for v in R.values()), len(R)
for lab, args in (("Qwen from Gemma's wrong-patch contexts, never-solved tasks, own point", (lambda s: s.startswith("R8 Qwen"), lambda c: c in ("wrongG", "wrongL"), True, lambda t: ptask.get(t) == 0)),
                  ("Qwen's own wrong-patch contexts, any fork", (lambda s: s.startswith("R6 Qwen"), lambda c: c in ("wrongG", "wrongL"), False)),
                  ("Gemma wrong-patch contexts, never-solved tasks, any fork (R3f/R3g)", (lambda s: s in ("R3f Gemma-31B prefill", "R3g Gemma-31B prefill"), lambda c: c in ("wrongG", "wrongL"), False, lambda t: ptask.get(t) == 0))):
    k, n = runs_any(*args)
    print(f"run level: {lab}: {k}/{n} runs pass (95 % upper {cp_up(k, n):.2f})")
b_empty = [p for p in pairs if own_cell.get((p["failed_run"], p["task"])) and own_cell[(p["failed_run"], p["task"])][0][1] == "empty"]
k = sum(any(r for r, _ in own_cell[(p["failed_run"], p["task"])]) for p in b_empty)
print(f"run level: boundary empty failed pair runs with >= 1 passing own-point commit: {k}/{len(b_empty)}")
# (E3) T = 1.0 vs T = 0.2 at the own point, paired by run
t1 = collections.defaultdict(lambda: collections.defaultdict(list))
for x in fc:
    if x["resolved"] is not None and x["own_commit_point"] and x["mode"] == "prefill":
        arm = "t1" if x.get("temperature") == 1.0 else ("t02" if x["study"] in E1S else None)
        if arm:
            t1[(x["explorer_run"], x["task"])][arm].append(x["resolved"])
tp = [(k[1], st.mean(v["t1"]) - st.mean(v["t02"])) for k, v in t1.items() if v["t1"] and v["t02"]]
byt = collections.defaultdict(list)
for t, d in tp:
    byt[t].append(d)
tk = list(byt); rnd = random.Random(0)
bs = sorted(st.mean(d for t in (rnd.choice(tk) for _ in tk) for d in byt[t]) for _ in range(4000))
print(f"T = 1.0 minus T = 0.2 at the own point (failed boundary runs): {st.mean(d for _, d in tp):+.3f} [{bs[100]:+.3f}, {bs[3899]:+.3f}] over {len(tp)} runs")
# (m3/m4) D variants: never-edited failed runs (from e1_runs_meta) as q = 0, and the failure-mix reweighting
if os.path.exists(f"{D}/e1_runs_meta.jsonl"):
    meta = {(x["explorer_run"], x["task"]): x for x in map(json.loads, open(f"{D}/e1_runs_meta.jsonl"))}
    print(f"pair runs: failed never edited {sum(1 for p in pairs if meta[(p['failed_run'], p['task'])]['never_edited'])}; "
          f"failed with an own point but no own-point episodes {sum(1 for p in pairs if not meta[(p['failed_run'], p['task'])]['never_edited'] and not draws.get((p['failed_run'], p['task'])))}")
    rows0 = []
    for p in pairs:
        a, b = draws.get((p["resolved_run"], p["task"])), draws.get((p["failed_run"], p["task"]))
        if not a:
            continue
        if not b and meta[(p["failed_run"], p["task"])]["never_edited"]:
            b = [(0, 0)]
        if b:
            rows0.append((p["task"], st.mean(r for _, r in a), st.mean(r for _, r in b)))
    wD = lambda R: sum(V(t) * (a - b) for t, a, b in R) / sum(V(t) for t, _, _ in R)
    print(f"D (p(1-p)-weighted): never-edited failed runs as q = 0: {wD(rows0):.3f} ({len(rows0)} tasks); unweighted (excluded): "
          f"{st.mean(st.mean(r for _, r in draws[(p['resolved_run'], p['task'])]) - st.mean(r for _, r in draws[(p['failed_run'], p['task'])]) for p in pairs if draws.get((p['resolved_run'], p['task'])) and draws.get((p['failed_run'], p['task']))):.3f}")
    why = collections.Counter()
    for p in pairs:
        ra, rb = draws.get((p["resolved_run"], p["task"])), draws.get((p["failed_run"], p["task"]))
        why["paired" if ra and rb else "resolved run: own-point episodes missing" if not ra else
            "failed run: no edit-tool call on a repository file" if meta[(p["failed_run"], p["task"])]["never_edited"] else
            "failed run: own-point episodes missing"] += 1
    print("own-point pairs (49 boundary tasks): " + ", ".join(f"{k} {v}" for k, v in sorted(why.items())))
# (M9) probes: within-task AUC per fork (mean over the 16 variants) and the leave-one-out task prior, from released scores
if jexists("probe_scores.jsonl.gz"):
    ps_rows = [json.loads(l) for l in jopen("probe_scores.jsonl.gz")]
    cells = sorted({c for r in ps_rows for c in r["scores"]})
    def wauc(rows, key):
        per = collections.defaultdict(list)
        for r in rows:
            if key(r) is not None and 0 < ptask.get(r["task"], 0) < 1:
                per[r["task"]].append((key(r), r["class"] == "res"))
        vals = []
        for v in per.values():
            pos = [s for s, y in v if y]; neg = [s for s, y in v if not y]
            if pos and neg:
                vals.append(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)))
        return st.mean(vals) if vals else float("nan"), len(vals)
    def pauc(rows, key):  # pooled AUC over all contexts
        v = [(key(r), r["class"] == "res") for r in rows if key(r) is not None]
        pos = [s for s, y in v if y]; neg = [s for s, y in v if not y]
        return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    # leave-one-out task pass rate of each run from task_outcomes (run name = log tag stem)
    rate = collections.defaultdict(list)
    for T in to.values():
        for t, v in T.items():
            rate[t] = list(v.values())
    print(f"probe scores: {len(ps_rows)} decision points, {len({r['tag'] + '|' + r['task'] for r in ps_rows})} runs, {len(cells)} variants")
    for fk in ("6", "10", "14", "own", "final"):
        R = [r for r in ps_rows if r["fork"] == fk]
        aucs = [wauc(R, lambda r, c=c: r["scores"].get(c)) for c in cells]
        loo = lambda r: (sum(rate[r["task"]]) - (r["class"] == "res")) / (len(rate[r["task"]]) - 1) if len(rate[r["task"]]) > 1 else None
        print(f"probe fork {fk:5s}: mean within-task AUC {st.mean(a for a, _ in aucs):.3f} (range {min(a for a, _ in aucs):.2f}-{max(a for a, _ in aucs):.2f}; "
              f"{aucs[0][1]} tasks); pooled AUC {min(pauc(R, lambda r, c=c: r['scores'].get(c)) for c in cells):.2f}-{max(pauc(R, lambda r, c=c: r['scores'].get(c)) for c in cells):.2f}; "
              f"leave-one-out task pass rate pooled AUC {pauc(R, loo):.3f}")
# (M3) failure-mix sensitivity: replace each failed run's value by its task's expected value under uniform choice
# among that task's logged failed runs (pooled value by failure type; never-edited failed runs as q = 0)
if os.path.exists(f"{D}/e1_runs_meta.jsonl"):
    fcls = {(x["explorer_run"], x["task"]): x["class"] for x in fc}
    typ = lambda c: "empty" if c in (None, "empty") else "wrong"
    vals = collections.defaultdict(list)
    for t, a, b in rows0:
        p = next(x for x in pairs if x["task"] == t)
        vals[typ(fcls.get((p["failed_run"], t)))].append(b)
    mv = {k: st.mean(v) for k, v in vals.items()}
    fails = collections.defaultdict(list)
    for x in map(json.loads, open(f"{D}/task_outcomes.jsonl")):
        if x["class"] != "res":
            fails[x["task"]].append(typ(x["class"]))
    rowsw = [(t, a, (sum(c == "empty" for c in fails[t]) / len(fails[t])) * mv["empty"] + (1 - sum(c == "empty" for c in fails[t]) / len(fails[t])) * mv["wrong"])
             for t, a, b in rows0 if fails.get(t)]
    print(f"D reweighted to each task's failure mix: {wD(rowsw):.3f} (pooled failed-run value by type {', '.join(f'{k} {v:.2f}' for k, v in mv.items())}; "
          f"chosen empty share {sum(typ(fcls.get((p['failed_run'], p['task']))) == 'empty' for p in pairs) / len(pairs):.2f} vs uniform "
          f"{st.mean(sum(c == 'empty' for c in fails[p['task']]) / len(fails[p['task']]) for p in pairs if fails.get(p['task'])):.2f})")
# (M7) Gemma 4 12B committing from the 31B's own commit points vs the 31B itself (R3b vs R3f), same pairing and bootstrap
for c in ("res",):
    own_s, swap_s = "R3f Gemma-31B prefill", "R3b Gemma-12B commits from 31B contexts"
    rows = [(k[2], st.mean(v), st.mean(cell[(swap_s,) + k[1:]])) for k, v in cell.items()
            if k[0] == own_s and k[3] == "own" and ccls[k[1:]] == c and (swap_s,) + k[1:] in cell]
    by = collections.defaultdict(list)
    for t, a, b in rows:
        by[t].append(b - a)
    tasks = list(by); rnd = random.Random(0); bs = []
    for _ in range(4000):
        bs.append(st.mean(d for t in (rnd.choice(tasks) for _ in tasks) for d in by[t]))
    bs.sort()
    print(f"swap 12B from 31B contexts {c}: n={len(rows)} runs  31B {st.mean(a for _, a, _ in rows):.2f}  12B "
          f"{st.mean(b for _, _, b in rows):.2f}  diff {st.mean(b - a for _, a, b in rows):+.2f} [{bs[100]:+.2f}, {bs[3899]:+.2f}]")

# ---- added 2026-10-25 (v0.8.1, response check): train results stratified by the harness time limit ----
# Time limit stated in each run's first harness message (logged prompts; private logs): train seed groups s0 (s0a/s0b) and
# traj_s1 = 20 min (these are all replayed train runs); traj_v0_* = 8 min; every dev, DeepSeek and Qwen run = 8 min.
print("\n== v0.8.1 additions ==")
TL = lambda g: 20 if g in ("s0", "traj_s1") else 8
TLrun = lambda run: 20 if run.startswith(("traj_s0a", "traj_s0b", "traj_s1")) else 8
if os.path.exists(f"{D}/task_outcomes.jsonl"):
    tr = to["train"]
    by = {lim: {t: [r for g, r in v.items() if TL(g) == lim] for t, v in tr.items()} for lim in (20, 8)}
    for lim in (20, 8):
        R = [r for v in by[lim].values() for r in v]
        T2 = {t: v for t, v in by[lim].items() if len(v) >= 2}
        ps = {t: st.mean(v) for t, v in T2.items()}; pbar = st.mean(ps.values())
        btw = st.mean((p - pbar) ** 2 for p in ps.values()) - st.mean(ps[t] * (1 - ps[t]) / (len(T2[t]) - 1) for t in ps)
        wth = st.mean(p * (1 - p) * len(T2[t]) / (len(T2[t]) - 1) for t, p in ps.items())
        print(f"train, {lim}-min seed groups: {len(R)} runs, resolve rate {st.mean(R):.3f}; between-task share "
              f"{btw / (btw + wth):.0%} over {len(T2)} tasks with >= 2 runs")
    common = [t for t in tr if by[20][t] and by[8][t]]
    dif = [st.mean(by[20][t]) - st.mean(by[8][t]) for t in common]
    rnd = random.Random(0); bs = sorted(st.mean(rnd.choice(dif) for _ in dif) for _ in range(4000))
    print(f"train, 20-min minus 8-min resolve rate, paired by task: {st.mean(dif):+.3f} [{bs[100]:+.3f}, {bs[3899]:+.3f}] over {len(common)} tasks")
if os.path.exists(f"{D}/e1_pairs.json"):
    combo = collections.Counter((TLrun(p["resolved_run"]), TLrun(p["failed_run"])) for p in pairs)
    print("E1 pairs by (resolved run, failed run) time limit: " + ", ".join(f"{a}/{b} min {n}" for (a, b), n in sorted(combo.items())))
    for lab, keep in (("same time limit", lambda p: TLrun(p["resolved_run"]) == TLrun(p["failed_run"])),
                      ("mixed time limits", lambda p: TLrun(p["resolved_run"]) != TLrun(p["failed_run"]))):
        rows = [(st.mean(r for _, r in draws[(p["resolved_run"], p["task"])]), st.mean(r for _, r in draws[(p["failed_run"], p["task"])]))
                for p in pairs if keep(p) and draws.get((p["resolved_run"], p["task"])) and draws.get((p["failed_run"], p["task"]))]
        rnd = random.Random(0); bs = sorted(st.mean(a - b for a, b in (rnd.choice(rows) for _ in rows)) for _ in range(4000))
        print(f"own-point gap, pairs with {lab}: {st.mean(a - b for a, b in rows):+.2f} [{bs[100]:+.2f}, {bs[3899]:+.2f}] over {len(rows)} tasks")
# (response check, M6) all non-reproduced runs by how many of their edits came after the spent 28-call budget
nr = [r for r in allruns if r.get("pool") and not r["replay_reproducible"]]
cnt = collections.Counter("no edit tool" if not refused(r) else "all edits" if all(refused(r)) else
                          "some edits" if any(refused(r)) else "no edits" for r in nr)
print(f"non-reproduced runs ({len(nr)}), edits after the spent budget: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))

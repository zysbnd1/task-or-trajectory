# SPDX-License-Identifier: Apache-2.0
"""paper-lab: reproduce cmd_sample's run selection for given args and compare with a saved selection file."""
import json, random, sys
from collections import defaultdict
import os
WORK = os.environ.get("PL_WORK", "/work")  # data root; set PL_WORK to run off-box
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, WORK + "/paper_lab"); sys.argv = ["x"]
import pl_r3 as m
def select(logs, forks, n, seed, classes=("res", "wrongG", "empty")):
    runs = m.load_runs(logs); tables = {}; pool = defaultdict(list)
    for r in runs:
        key = (r["run"], r["seed"])
        if key not in tables:
            tables[key] = m.outcome_table(*key)
        o = tables[key].get(r["iid"])
        if not o or not m.snapshot_of(r["iid"]):
            continue
        r["cls"] = m.classify(r["iid"], o)
        r["pts"], r["natural"], r["first_edit"], r["unsafe_after"] = m.fork_points(r, forks)
        if r["pts"] or r["natural"] is not None:
            pool[r["cls"]].append(r)
    rnd = random.Random(seed); chosen = []
    for c in classes:
        rs = pool.get(c, []); rnd.shuffle(rs); chosen += rs[:n]
    return [(r["tag"], r["iid"]) for r in chosen]
for logs, n, seed, saved in (("traj_v0_s2_seed0_shard*.jsonl,cs-r1__r1_dev_b0_seed0_shard*.jsonl", 30, 1, "selection_samples_prefill_g.json"),
                             ("r6_qwen_dev_seed0_shard*.jsonl", 15, 0, "selection_samples_prefill_qwen.json")):
    new = select(logs, [6, 10, 14], n, seed)
    old = [(x["tag"], x["iid"]) for x in json.load(open(WORK + "/paper_lab/r3/" + saved))]
    print(saved, "identical selection:", new == old, len(new), len(old))

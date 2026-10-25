# SPDX-License-Identifier: Apache-2.0
"""paper-lab E1 run list ($0): task-matched pairs on boundary tasks (red-team M1).
For every boundary task in $PL_WORK/paper_lab/e5/boundary.json (0 < p < 1 over all same-config Gemma seeds, with logged
resolved and failed runs): one resolved and one failed run. Runs already in R3f/R3g samples are preferred (their
episodes are reused; same instrument), otherwise a seeded draw. Log tags (llmlog/<run>_seed0_shardN, cs-r1__ prefix
for r1_* runs) are found by indexing each run's own logs (problem-statement probe, pl_r3.load_runs).
Writes r3/e1_runs.json (all pairs), r3/e1_new_runs.json (runs still to sample) and r3/e1_failed_runs.json."""
import glob, json, os, random, sys
WORK = os.environ.get("PL_WORK", "/work")  # data root; set PL_WORK to run off-box
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, WORK + "/paper_lab"); sys.argv = ["x"]
import pl_r3 as m

B = json.load(open(WORK + "/paper_lab/e5/boundary.json"))


_IDX = {}


def tag_of(run, iid):
    """(run, task) -> log tag, by indexing the run's own LLM logs (results-shard numbers do not match log shards)."""
    if run not in _IDX:
        _IDX[run] = {}
        for pre in ("", "cs-r1__"):
            for f in sorted(glob.glob(f"{m.LLM}/{pre}{run}_seed0_shard*.jsonl")):
                for r in m.load_runs(os.path.basename(f)):
                    _IDX[run].setdefault(r["iid"], r["tag"])
    return _IDX[run].get(iid)


def main():
    sampled = set()
    for fn in ("samples_prefill.jsonl", "samples_prefill_g.jsonl"):
        for l in open(f"{WORK}/paper_lab/r3/{fn}"):
            x = json.loads(l)
            if "iid" in x:
                sampled.add((x["tag"], x["iid"]))
    rnd = random.Random(1)
    pairs, missing = [], 0
    for pool in ("train", "dev"):
        for b in sorted(B[pool]["boundary"], key=lambda b: b["iid"]):
            iid = b["iid"]
            res = [t for t in (tag_of(r, iid) for r in b["res_runs"]) if t]
            fail = [t for t in (tag_of(r, iid) for r, _ in b["fail_runs"]) if t]
            if not res or not fail:
                missing += 1
                continue
            pick = lambda ts: sorted([t for t in ts if (t, iid) in sampled]) or [rnd.choice(sorted(ts))]
            r, f = pick(res)[0], pick(fail)[0]
            pairs.append({"pool": pool, "iid": iid, "p": b["p"], "n": b["n"], "res": r, "fail": f,
                          "res_sampled": (r, iid) in sampled, "fail_sampled": (f, iid) in sampled})
    json.dump(pairs, open(WORK + "/paper_lab/r3/e1_runs.json", "w"), indent=1)
    new = [{"tag": p[c], "iid": p["iid"]} for p in pairs for c in ("res", "fail") if not p[c + "_sampled"]]
    json.dump(new, open(WORK + "/paper_lab/r3/e1_new_runs.json", "w"), indent=1)
    allr = [{"tag": p[c], "iid": p["iid"]} for p in pairs for c in ("res", "fail")]
    json.dump(allr, open(WORK + "/paper_lab/r3/e1_all_runs.json", "w"), indent=1)
    json.dump([{"tag": p["fail"], "iid": p["iid"]} for p in pairs], open(WORK + "/paper_lab/r3/e1_failed_runs.json", "w"), indent=1)
    print(f"pairs {len(pairs)} (train {sum(p['pool'] == 'train' for p in pairs)}, dev {sum(p['pool'] == 'dev' for p in pairs)}), "
          f"tasks skipped {missing}; runs to sample {len(new)} (reused: res {sum(p['res_sampled'] for p in pairs)}, "
          f"fail {sum(p['fail_sampled'] for p in pairs)})")


if __name__ == "__main__":
    main()

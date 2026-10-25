# SPDX-License-Identifier: Apache-2.0
"""paper-lab: build the PRIVATE minimal release slice (decisions #31) under $PL_WORK/paper_lab/release_slice/.
Nothing here is published; the user gives the go-ahead at release time.

Contents (enough to recompute forced-commit Q_c and the ever-green numbers; no post-commit trajectories):
  fork_contexts.jsonl.gz   exact logged request (messages, tools, sampling params) at each fork call of each selected run
                           (calls 6/10/14 and the run's own first-edit call — all before its first edit)
  forced_commits.jsonl.gz  every graded forced-commit episode: committer, fork, seed, patch, resolved
  evergreen_edits.jsonl.gz per replayed run: cumulative patch after each edit (+ edit index, call index, applied?, resolved)
  runs.jsonl               selected / replayed runs: pool, model, task id, outcome class, first-edit call, agent calls
  e1_pairs.json            E1 boundary pairs (task, solve rate p over same-config runs, resolved run, failed run)
  task_outcomes.jsonl      outcome class of every same-config Gemma 4 31B run per task (E5 / Table 2)
  call_kinds.jsonl         (v0.7) per-call kind labels of every replayed run (edit/test/exec/read-search/submit/other;
                           from pl_m8.py; no command text) -> Table 3's post-green composition
  MANIFEST.json            counts, sizes, sha256, scrub report
Scrubbing: local model paths -> model names; sandbox/box paths -> /workspace or <tmp>; no hostnames/keys (checked).
"""
import glob, gzip, hashlib, json, os, re, sys
from collections import Counter
WORK = os.environ.get("PL_WORK", "/work")  # data root; set PL_WORK to run off-box (e.g. on a local copy of the backup)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, WORK + "/paper_lab"); sys.argv = ["x"]
import pl_r3 as m

OUT = WORK + "/paper_lab/release_slice"
G = m.load_grades()
STUDIES = {  # samples file -> (study, explorer logs used for the forks)
    "samples_prefill.jsonl": ("R3f Gemma-31B prefill", "traj_s1_seed0_shard*.jsonl"),
    "samples.jsonl": ("R3 Gemma-31B instruct", "traj_s1_seed0_shard*.jsonl"),
    "samples_prefill_g.jsonl": ("R3g Gemma-31B prefill", "traj_v0_s2_seed0_shard*.jsonl,cs-r1__r1_dev_b0_seed0_shard*.jsonl"),
    "samples_prefill_12b.jsonl": ("R3b Gemma-12B commits from 31B contexts", "traj_s1_seed0_shard*.jsonl"),
    "samples_prefill_qwen.jsonl": ("R6 Qwen3.6-27B prefill", "r6_qwen_dev_seed0_shard*.jsonl"),
    "samples_x_gq.jsonl": ("R8 Gemma-31B commits from Qwen contexts", "r6_qwen_dev_seed0_shard*.jsonl"),
    "samples_x_qg.jsonl": ("R8 Qwen commits from Gemma-31B contexts", "traj_v0_s2_seed0_shard*.jsonl,cs-r1__r1_dev_b0_seed0_shard*.jsonl"),
}
# window 7 (red-team review): E1 task-matched boundary pairs, the no-instruction ablation and E3 (T = 1.0); explorer logs =
# the exact shard files of the 98 E1 runs (r3/e1_logs.txt)
if os.path.exists(WORK + "/paper_lab/r3/e1_logs.txt"):
    _E1 = open(WORK + "/paper_lab/r3/e1_logs.txt").read().strip()
    STUDIES.update({"samples_e1a.jsonl": ("E1 Gemma-31B boundary pairs", _E1),
                    "samples_e1b.jsonl": ("E1 Gemma-31B boundary pairs", _E1),
                    "samples_e1_noinstr.jsonl": ("E1 ablation: no injected instruction", _E1),
                    "samples_e3_t1.jsonl": ("E3 failed boundary runs at T = 1.0", _E1)})
REPLAYS = {"replay.jsonl": "Gemma-31B train", "replay_devg.jsonl": "Gemma-31B dev", "replay_ds.jsonl": "DeepSeek-V4-Pro dev",
           "replay_qwen.jsonl": "Qwen3.6-27B dev"}
LOGGED_ROOT = os.environ.get("PL_LOGGED_ROOT", WORK)  # the data root the logs were written under (scrubbed below)
MODEL_NAMES = {f"{LOGGED_ROOT}/models/gemma-4-31B-it-qat-w4a16-ct": "gemma-4-31b-it-qat-w4a16-ct",
               f"{LOGGED_ROOT}/models/gemma-4-12B-it-qat-w4a16-ct": "gemma-4-12b-it-qat-w4a16-ct"}
SCRUB = [(re.compile(r"/tmp/[\w.-]*gemma_sandbox_[\w-]+/workspace"), "/workspace"),
         (re.compile(r"/tmp/[\w.-]*sandbox[\w.-]*"), "<tmp>"),
         (re.compile(re.escape(LOGGED_ROOT) + r"/[\w./-]+"), "<path>")]
FORBID = re.compile(r"(api[_-]?key|bearer\s+[a-z0-9]|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|\b\d{1,3}(\.\d{1,3}){3}\b|vast\.ai)", re.I)
scrub_hits = Counter()


def scrub(o):
    if isinstance(o, str):
        for rx, rep in SCRUB:
            o, n = rx.subn(rep, o)
            if n:
                scrub_hits[rep] += n
        return o
    if isinstance(o, list):
        return [scrub(x) for x in o]
    if isinstance(o, dict):
        return {k: scrub(MODEL_NAMES.get(v, v) if k == "model" else v) for k, v in o.items()}
    return o


def grade(iid, patch):
    if not patch:
        return False
    g = G.get(hashlib.sha1((iid + "\n" + patch).encode()).hexdigest()[:12])
    return None if g is None else bool(g["resolved"])


def main():
    os.makedirs(OUT, exist_ok=True)
    fc, fe, ru = gzip.open(f"{OUT}/fork_contexts.jsonl.gz", "wt"), gzip.open(f"{OUT}/forced_commits.jsonl.gz", "wt"), {}
    n_ctx = n_ep = 0
    logs_cache = {}
    for fn, (study, logs) in STUDIES.items():
        path = f"{WORK}/paper_lab/r3/{fn}"
        if not os.path.exists(path):
            continue
        if logs not in logs_cache:
            logs_cache[logs] = {(r["tag"], r["iid"]): r for r in m.load_runs(logs)}
        runs = logs_cache[logs]
        need = set()
        for l in open(path):
            x = json.loads(l)
            if "iid" not in x or x.get("skip"):
                continue
            need.add((x["tag"], x["iid"], x["k"]))
            fe.write(json.dumps({"study": study, "explorer_run": x["tag"], "task": x["iid"], "class": x.get("cls"),
                                 "fork_call": x["k"], "own_commit_point": bool(x["natural"]), "sample": x["j"],
                                 "committer": MODEL_NAMES.get(x.get("model"), x.get("model")), "mode": x.get("mode", "instruct"),
                                 "seed": x.get("seed"), "compliant": x.get("compliant"), "patch": scrub(x.get("patch", "")),
                                 "instruction": x.get("injected"), "temperature": x.get("temp"),
                                 "resolved": grade(x["iid"], x.get("patch", ""))}) + "\n"); n_ep += 1
        for tag, iid, k in sorted(need):
            r = runs.get((tag, iid))
            if not r or k >= len(r["calls"]):
                continue
            req = r["calls"][k]["req"]
            fc.write(json.dumps({"study": study, "explorer_run": tag, "task": iid, "fork_call": k,
                                 "request": scrub({x: req[x] for x in ("messages", "tools", "model", "temperature", "top_p",
                                                                       "max_completion_tokens", "chat_template_kwargs",
                                                                       "thinking_token_budget") if x in req})}) + "\n"); n_ctx += 1
            ru[(tag, iid)] = {"explorer_run": tag, "task": iid, "agent_calls": len(r["calls"])}
    fc.close(); fe.close()
    ev = gzip.open(f"{OUT}/evergreen_edits.jsonl.gz", "wt"); n_ev = 0
    for fn, pool in REPLAYS.items():
        path = f"{WORK}/paper_lab/r3/{fn}"
        if not os.path.exists(path):
            continue
        for l in open(path):
            x = json.loads(l)
            if x["kind"] == "replay":
                ev.write(json.dumps({"pool": pool, "run": x["tag"], "task": x["iid"], "class": x["cls"], "call": x["k"],
                                     "edit_index": x["j"], "edit_applied": x["edit_ok"], "cumulative_patch": scrub(x["patch"]),
                                     "resolved": grade(x["iid"], x["patch"])}) + "\n"); n_ev += 1
            else:
                ru.setdefault((x["tag"], x["iid"]), {"explorer_run": x["tag"], "task": x["iid"]}).update(
                    {"pool": pool, "class": x["cls"], "replay_reproducible": x["matches_saved_patch"], "n_edits": x["n_edits"]})
    ev.close()
    # agent-call count and per-call tool names for every replayed run (pl_m8.py -> r3/run_calls.jsonl; red-team M8 gap)
    if os.path.exists(WORK + "/paper_lab/r3/run_calls.jsonl"):
        for l in open(WORK + "/paper_lab/r3/run_calls.jsonl"):
            x = json.loads(l)
            if (x["run"], x["task"]) in ru and x.get("agent_calls"):
                ru[(x["run"], x["task"])].update({"agent_calls": x["agent_calls"], "tools_per_call": x["tools"]})
    # E1 pairs and per-task outcomes of all same-config Gemma runs (Table 2 / E5 and Table 4 / E1 recompute from these)
    if os.path.exists(WORK + "/paper_lab/r3/e1_runs.json"):
        json.dump([{"pool": p["pool"], "task": p["iid"], "p": p["p"], "n_runs": p["n"], "resolved_run": p["res"],
                    "failed_run": p["fail"]} for p in json.load(open(WORK + "/paper_lab/r3/e1_runs.json"))],
                  open(f"{OUT}/e1_pairs.json", "w"), indent=1)
    import pl_e5
    with open(f"{OUT}/task_outcomes.jsonl", "w") as f:
        for pool, rlist in pl_e5.POOLS.items():
            for run, seed in rlist:
                for iid, o in m.outcome_table(run, seed).items():
                    if iid in m.TASKS:
                        f.write(json.dumps({"pool": pool, "run": run, "seed_group": pl_e5.SEED_OF.get(run, run),
                                            "task": iid, "class": m.classify(iid, o)}) + "\n")
    with open(f"{OUT}/runs.jsonl", "w") as f:
        for v in ru.values():
            f.write(json.dumps(v) + "\n")
    ck = WORK + "/paper_lab/r3/call_kinds.jsonl"  # written by pl_m8.py; pool names as in the replay files
    if os.path.exists(ck):
        with open(f"{OUT}/call_kinds.jsonl", "w") as f:
            for l in open(ck):
                x = json.loads(l)
                x["pool"] = x["pool"].replace("Gemma 31B", "Gemma-31B").replace("DeepSeek V4 Pro", "DeepSeek-V4-Pro")
                f.write(json.dumps(x) + "\n")
    # forbidden-pattern scan over everything written
    bad = Counter()
    for p in glob.glob(f"{OUT}/*.jsonl*"):
        op = gzip.open if p.endswith(".gz") else open
        for line in op(p, "rt"):
            for mm in FORBID.finditer(line):
                bad[mm.group(0)[:20]] += 1
    man = {"fork_contexts": n_ctx, "forced_commit_episodes": n_ep, "evergreen_edit_records": n_ev, "runs": len(ru),
           "files": {os.path.basename(p): {"bytes": os.path.getsize(p),
                                           "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
                     for p in glob.glob(f"{OUT}/*") if not p.endswith("MANIFEST.json")},
           "scrubbed": dict(scrub_hits), "forbidden_pattern_hits": dict(bad)}
    json.dump(man, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))
    print({k: v["bytes"] for k, v in man["files"].items()})


if __name__ == "__main__":
    main()

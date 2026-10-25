# SPDX-License-Identifier: Apache-2.0
"""paper-lab M8 / release gap ($0, read-only; red-team review 2026-10-22).
1. For every replayed run (all four ever-green pools): agent-call count and per-call tool names
   -> $PL_WORK/paper_lab/r3/run_calls.jsonl (feeds runs.jsonl in the release slice; post-green share becomes recomputable).
2. What the calls after the first passing tree are (resolved, replay-reproducible runs): each tool call classified as
   edit / test (pytest, unittest, tox, nox, test scripts) / exec (other python or program runs) / read-search (read_file,
   grep, find, sed -n, cat, head, tail, ls, git log/show/grep/diff) / submit (submit_patch, get_status) / other.
3. Shell-edit (non-reproducible) exclusion by outcome class and pool.
4. (v0.7) Per-call kind labels of every replayed run -> $PL_WORK/paper_lab/r3/call_kinds.jsonl (labels only, no command text;
   added to the release slice so Table 3's post-green composition recomputes from the slice).
"""
import collections, hashlib, json, os, re, statistics as st, sys
WORK = os.environ.get("PL_WORK", "/work")  # data root; set PL_WORK to run off-box (e.g. on a local copy of the backup)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, WORK + "/paper_lab"); sys.argv = ["x"]
import pl_r3 as m

G = m.load_grades()
POOLS = [("Gemma 31B train", "replay.jsonl", "traj_s0a_seed0_shard*.jsonl,traj_s0b_seed0_shard*.jsonl,traj_s1_seed0_shard*.jsonl"),
         ("Gemma 31B dev", "replay_devg.jsonl", "cs-r1__r1_dev_b0_seed0_shard*.jsonl,cs-r1__r1_dev_b0b_seed0_shard*.jsonl"),
         ("DeepSeek V4 Pro dev", "replay_ds.jsonl", "ds_pilot_dev_seed0_shard*.jsonl"),
         ("Qwen3.6-27B dev", "replay_qwen.jsonl", "r6_qwen_dev_seed0_shard*.jsonl")]
TEST = re.compile(r"\b(pytest|py\.test|unittest|tox|nox)\b|\btest[s]?/|test_\w+\.py|\bmake test")
READ = re.compile(r"^\s*(cd \S+\s*&&\s*)?(git (grep|log|show|diff|status|blame)|grep|rg|find|sed -n|cat|head|tail|ls|wc|nl|tree|awk)\b")


def kind(name, args):
    if name in m.EDIT:
        return "edit"
    if name in ("submit_patch", "get_status"):
        return "submit"
    if name == "read_file":
        return "read-search"
    if name == "run_command":
        try:
            cmd = json.loads(args).get("command", "")
        except Exception:
            cmd = ""
        if TEST.search(cmd):
            return "test"
        if READ.match(cmd):
            return "read-search"
        if re.search(r"\bpython\d?(\.\d+)?\b|\./\S+|\bbash\b", cmd):
            return "exec"
        return "other"
    return "other"


def main():
    out = open(WORK + "/paper_lab/r3/run_calls.jsonl", "w")
    kout = open(WORK + "/paper_lab/r3/call_kinds.jsonl", "w")
    for label, rfile, logs in POOLS:
        runs = {(r["tag"], r["iid"]): r for r in m.load_runs(logs)}
        R = [json.loads(l) for l in open(WORK + "/paper_lab/r3/" + rfile)]
        fin = {(x["tag"], x["iid"]): x for x in R if x["kind"] == "replay_final"}
        seq = collections.defaultdict(list)
        for x in R:
            if x["kind"] == "replay" and x.get("patch"):
                g = G.get(hashlib.sha1((x["iid"] + "\n" + x["patch"]).encode()).hexdigest()[:12])
                seq[(x["tag"], x["iid"])].append((x["j"], x["k"], (g or {}).get("resolved")))
        for key, f in fin.items():
            r = runs.get(key)
            out.write(json.dumps({"pool": label, "run": key[0], "task": key[1], "class": f["cls"],
                                  "agent_calls": len(r["calls"]) if r else None,
                                  "tools": [[n for n, _ in c["tools"]] for c in r["calls"]] if r else None}) + "\n")
            if r:
                kout.write(json.dumps({"pool": label, "run": key[0], "task": key[1],
                                       "kinds": [[kind(n, a) for n, a in c["tools"]] for c in r["calls"]]}) + "\n")
        # 2. post-green composition
        comp = collections.Counter(); nrun = 0; any_test = any_exec = any_edit = 0
        for key, f in fin.items():
            if f["cls"] != "res" or not f["matches_saved_patch"] or key not in runs:
                continue
            v = sorted(seq.get(key, []))
            if not v or not any(g for _, _, g in v):
                continue
            k0 = next(k for _, k, g in v if g)
            kinds = [kind(n, a) for c in runs[key]["calls"][k0 + 1:] for n, a in c["tools"]]
            comp.update(kinds); nrun += 1
            any_test += "test" in kinds; any_exec += "exec" in kinds; any_edit += "edit" in kinds
        tot = sum(comp.values())
        print(f"== {label}: {nrun} resolved green runs; {tot} tool calls after the first passing tree")
        print("   composition: " + ", ".join(f"{k} {comp[k] / tot:.0%}" for k in ("read-search", "test", "exec", "edit", "submit", "other")))
        print(f"   runs with any test after green {any_test / nrun:.0%}, any other execution {any_exec / nrun:.0%}, any edit {any_edit / nrun:.0%}")
        # 3. exclusion by class
        ex = collections.defaultdict(lambda: [0, 0])
        for key, f in fin.items():
            ex[f["cls"]][0] += 1; ex[f["cls"]][1] += not f["matches_saved_patch"]
        print("   not replay-reproducible (shell edits) by class: " + ", ".join(f"{c} {b}/{a} ({b / a:.0%})" for c, (a, b) in sorted(ex.items())))
    out.close()


if __name__ == "__main__":
    main()

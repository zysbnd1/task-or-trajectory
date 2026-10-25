# SPDX-License-Identifier: Apache-2.0
"""paper-lab (v0.8, review m4/M3): per-run metadata of the 98 E1 pair runs, for the release slice.
For each run: agent calls, first edit_file/write_file call on a repository file (not /tmp), first state-changing shell
command, own commit point as used by the forced-commit instrument (pl_r3.fork_points), never-edited flag.
Reads the E1 explorer logs (r3/e1_logs.txt) and r3/e1_runs.json. Writes r3/e1_runs_meta.jsonl. CPU only."""
import json, os, sys
WORK = os.environ.get("PL_WORK", "/work")  # data root; set PL_WORK to run off-box (e.g. on a local copy of the backup)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, WORK + "/paper_lab"); sys.argv = ["x"]
import pl_r3 as m

pairs = json.load(open(WORK + "/paper_lab/r3/e1_runs.json"))
want = {(p[c], p["iid"]): ("resolved" if c == "res" else "failed") for p in pairs for c in ("res", "fail")}
runs = {(r["tag"], r["iid"]): r for r in m.load_runs(open(WORK + "/paper_lab/r3/e1_logs.txt").read().strip())}
with open(WORK + "/paper_lab/r3/e1_runs_meta.jsonl", "w") as f:
    for (tag, iid), role in sorted(want.items(), key=lambda kv: (kv[0][1], kv[1])):
        r = runs.get((tag, iid))
        if r is None:
            f.write(json.dumps({"explorer_run": tag, "task": iid, "role": role, "found": False}) + "\n"); continue
        pts, natural, first_edit, unsafe = m.fork_points(r, [6, 10, 14])
        f.write(json.dumps({"explorer_run": tag, "task": iid, "role": role, "found": True, "agent_calls": len(r["calls"]),
                            "first_repo_edit_call": first_edit, "first_state_changing_shell_call": unsafe,
                            "own_commit_point": natural, "clean_forks": pts, "never_edited": first_edit is None}) + "\n")
print("wrote", len(want), "runs")

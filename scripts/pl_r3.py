# SPDX-License-Identifier: Apache-2.0
"""paper-lab R3 (P3): executable forced-commit values Q_c(k).

At agent call k of a logged run (exact logged request), tell the agent to commit now, let it make its edit(s), apply
them to a fresh copy of the base snapshot, and grade the resulting patch with the official evaluator. Forks are taken
only *before the run's first edit* and before any state-modifying shell command, so the sandbox state at the fork is the
base snapshot itself (reads don't change it) -> no environment replay is needed.

The commit instruction is injected the way the harness injects its own `budget_warning`: as a field of the last tool
response (JSON), so the chat structure (and how earlier thoughts are rendered) is unchanged. After a forced edit the
agent may continue with edit-only calls (up to --max-commit-calls in total); any non-edit call ends the commit episode.
This measures a *lower bound* on the value of committing at k (no verification, no further reading).

Stages (each resumable; everything is written under $PL_WORK/paper_lab/r3/):
  sample  (needs the vLLM server, venv-harness python):
      pl_r3.py sample --logs 'traj_s1_seed0_shard*.jsonl' --forks 6,10,14,18 --k 4 --n-per-class 20 [--api http://127.0.0.1:8000]
  grade   (CPU only; <= 4 processes, nice 10; envcache read via the normal sandbox path; stops if $PL_WORK free < 30 GB):
      pl_r3.py grade --procs 4
  analyze (pure python):
      pl_r3.py analyze
Logs without token ids (TRAJ-V0 seeds 0-1) are fine here: the fork resends the logged messages, which the same server
renders exactly as for the original call.
"""
import argparse, glob, hashlib, json, os, random, re, shlex, shutil, subprocess, sys, tarfile, tempfile, time, urllib.request
WORK = os.environ.get("PL_WORK", "/work")  # data root; set PL_WORK to run off-box (e.g. on a local copy of the backup)
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT = Path(WORK + "/paper_lab/r3")
LLM = WORK + "/llmlog"
RES = WORK + "/results"
TMP = Path("/tmp/pl_r3")
SNAP_DIRS = [WORK + "/ext/train_v0/snapshots", WORK + "/ext/train_v1/snapshots", WORK + "/ext/dev_v0/snapshots", WORK + "/comp/snapshots"]
TASK_FILES = [WORK + "/ext/train_v0/tasks.jsonl", WORK + "/ext/train_v1/tasks.jsonl", WORK + "/ext/dev_v0/tasks.jsonl", WORK + "/comp/tasks.jsonl"]
EDIT = {"edit_file", "write_file"}
FORCE = ("COMMIT NOW: stop exploring. Your next tool call must be edit_file (or write_file) with your best fix for the "
         "issue, based on what you have already read. Do not read, search, test or run anything first.")
INJECT = True   # --no-instruction: forced episodes without the COMMIT NOW / FOLLOW texts (red-team M3 ablation)
TEMP = None     # --temperature: override the logged sampling temperature in forced episodes (red-team E3)
FOLLOW = "Finish the fix now: make any remaining edit_file calls, then call submit_patch. Do not read, search, test or run anything."
MAX_STDOUT = 5000
FARM = OUT / "envfarm"

# ---------------------------------------------------------------- tasks / outcomes
TASKS = {}
for tf in TASK_FILES:
    if os.path.exists(tf):
        for l in open(tf):
            t = json.loads(l); TASKS.setdefault(t["instance_id"], (t, tf))
ns = lambda s: re.sub(r"\s+", "", s or "")
PS2IID = {}
for iid, (t, _) in TASKS.items():
    PS2IID.setdefault(ns(t["problem_statement"])[:200], iid)


def flat(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(flat(x.get("text") if isinstance(x, dict) else x) for x in c)
    return "" if c is None else str(c)


# The logged task message embeds the problem statement between harness-specific delimiters. They are not shipped (harness
# text): set PL_PS_RE to a regular expression whose group 1 captures the problem statement in that message.
PS_RE = os.environ.get("PL_PS_RE")


def iid_of(first_user):
    if not PS_RE:
        raise RuntimeError("set PL_PS_RE: a regex whose group 1 captures the problem statement in the logged task message")
    m = re.search(PS_RE, flat(first_user), re.S)
    return PS2IID.get(ns(m.group(1))[:200]) if m else None


def gold_files(iid):
    p = TASKS[iid][0].get("patch", "")
    fs = re.findall(r"^\+\+\+ b/(\S+)", p, re.M)
    return {f for f in fs if not (os.path.basename(f).startswith("test_") or "/tests/" in "/" + f)}


def outcome_table(run_name, seed):
    """{iid: (resolved, patch_size, patch_files)} from $PL_WORK/results/<run>/seed<k>/shard*/."""
    out = {}
    for f in glob.glob(f"{RES}/{run_name}/seed{seed}/shard*/task_results.jsonl"):
        d = os.path.dirname(f)
        for l in open(f):
            r = json.loads(l)
            pp = os.path.join(d, "patches", f"{r['instance_id']}.patch")
            pf = set(re.findall(r"^\+\+\+ b/(\S+)", open(pp, errors="replace").read(), re.M)) if os.path.exists(pp) else set()
            out[r["instance_id"]] = (bool(r.get("resolved")), r.get("agent_patch_size") or 0, pf)
    return out


def classify(iid, o):
    res, size, pf = o
    if res:
        return "res"
    if not size:
        return "empty"
    return "wrongG" if pf & gold_files(iid) else "wrongL"


# ---------------------------------------------------------------- log parsing
TAG = re.compile(r"^(?:cs-[^_]+__)?(.+)_seed(\d+)_shard(\d+)$")


def load_runs(globs):
    runs = []
    for g in globs.split(","):
        for f in sorted(glob.glob(os.path.join(LLM, g))):
            tag = os.path.basename(f)[:-6]
            m = TAG.match(tag)
            if not m:
                continue
            run_name, seed = m.group(1), m.group(2)
            cur = None
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                req, resp = r.get("req"), r.get("resp")
                if not isinstance(req, dict) or not isinstance(resp, dict):
                    continue
                msgs = req.get("messages") or []
                if len(msgs) == 2 and [x.get("role") for x in msgs] == ["system", "user"]:
                    iid = iid_of(msgs[1].get("content"))
                    cur = {"tag": tag, "run": run_name, "seed": seed, "iid": iid, "calls": []} if iid else None
                    if cur:
                        runs.append(cur)
                if cur is None or len(msgs) == 1:  # skip compaction-summariser requests
                    continue
                msg = ((resp.get("choices") or [{}])[0].get("message") or {})
                cur["calls"].append({"req": req, "tools": [(t["function"]["name"], t["function"]["arguments"])
                                                            for t in (msg.get("tool_calls") or [])]})
    return runs


SEG = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_TRACKED = {}


def tracked_files(iid):
    """Paths in the base snapshot (relative, no .git) -- what a fork must leave untouched."""
    if iid not in _TRACKED:
        snap = snapshot_of(iid); names = set()
        if snap:
            with tarfile.open(snap) as tf:
                for m in tf.getmembers():
                    n = m.name[2:] if m.name.startswith("./") else m.name
                    if n and not n.startswith(".git/") and n != ".git":
                        names.add(n)
        _TRACKED[iid] = names
    return _TRACKED[iid]


def _rel(p):
    p = p.strip("'\"")
    if p.startswith("/workspace/"):
        p = p[len("/workspace/"):]
    return p[2:] if p.startswith("./") else p


def unsafe_command(cmd, tracked=None):
    """True if a shell command may change a *tracked* file of the base snapshot (i.e. acts as an edit).
    Scratch files, /tmp writes and git no-ops on an unmodified tree are allowed. With tracked=None: conservative."""
    hit = (lambda p: True) if tracked is None else (lambda p: _rel(p) in tracked or any(t.startswith(_rel(p).rstrip("/") + "/") for t in tracked))
    if re.search(r"sed\s+-i|perl\s+-\w*i|\bgit\s+(apply|am|rm|mv|switch|merge|rebase|cherry-pick)\b", cmd):
        return True
    for m in re.finditer(r"(?<![0-9&>])>>?\s*([^\s;|&]+)", cmd):
        t = m.group(1)
        if not (t.startswith("/tmp") or t.startswith("/dev/") or t.startswith("&")) and hit(t):
            return True
    if re.search(r"open\([^)]*['\"][wa]b?\+?['\"]", cmd):
        if tracked is None or any(f in cmd for f in tracked if "/" in f or f.endswith(".py")):
            return True
    for seg in SEG.split(cmd):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        if not toks:
            continue
        v = os.path.basename(toks[0])
        args = [t for t in toks[1:] if not t.startswith("-")]
        if v in ("mv", "rm", "truncate", "patch", "chmod", "ln") and any(not a.startswith("/tmp") and hit(a) for a in args):
            return True
        if v == "cp" and args and not args[-1].startswith("/tmp") and hit(args[-1]):
            return True
        if v == "tee" and any(not a.startswith("/tmp") and not a.startswith("/dev/") and hit(a) for a in args):
            return True
    return False


def fork_points(run, forks):
    calls = run["calls"]
    first_edit, unsafe_after = None, None
    for i, c in enumerate(calls):
        for name, a in c["tools"]:
            try:
                args = json.loads(a)
            except Exception:
                args = {}
            if name in EDIT and not str(args.get("filepath", "")).startswith("/tmp"):
                first_edit = i if first_edit is None else first_edit
            if name == "run_command" and unsafe_after is None and unsafe_command(str(args.get("command", "")), tracked_files(run["iid"])):
                unsafe_after = i
        if first_edit is not None:
            break
    # the request at call k reflects the actions of calls < k: forks up to the first edit / first unsafe command are clean
    limit = min(x for x in (first_edit, unsafe_after, len(calls) - 1) if x is not None)
    pts = [k for k in forks if 1 <= k <= limit]
    natural = first_edit if (first_edit is not None and first_edit <= limit) else None
    return pts, natural, first_edit, unsafe_after


# ---------------------------------------------------------------- workspace + edits (venv-harness)
def snapshot_of(iid):
    for d in SNAP_DIRS:
        p = os.path.join(d, f"{iid}.tgz")
        if os.path.exists(p):
            return p
    return None


def make_ws(iid):
    TMP.mkdir(parents=True, exist_ok=True)
    ws = Path(tempfile.mkdtemp(prefix=f"{iid[:40]}_", dir=TMP))
    with tarfile.open(snapshot_of(iid)) as tf:
        tf.extractall(ws, filter="fully_trusted")
    return ws


def ws_path(ws, fp):
    from swegemma.tools.workspace import _resolve_workspace_path
    return _resolve_workspace_path(fp, workspace_root=ws, allow_root=False)


def apply_tool(ws, name, args, created):
    from swegemma.edit import apply_replacement
    try:
        p = ws_path(ws, str(args.get("filepath", "")))
        if name == "write_file":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args.get("content", ""), encoding="utf-8")
            created.add(str(p.relative_to(ws)))
            return {"status": "ok", "filepath": args.get("filepath"), "size": len(args.get("content", ""))}, True
        content = p.read_text(encoding="utf-8", errors="replace")
        if not content:
            # same failure as the harness's edit_file; the published runs returned the harness's own wording for it
            raise ValueError(f"empty file {args.get('filepath')!r}: edit_file needs existing content")
        am = args.get("allow_multiple", False)
        am = am.strip().lower() in ("true", "1", "yes") if isinstance(am, str) else bool(am)
        r = apply_replacement(content, args.get("old_string", ""), args.get("new_string", ""), filepath=str(args.get("filepath")), allow_multiple=am)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(r.new_content)
        diff = r.diff[:MAX_STDOUT]
        return {"status": "ok", "filepath": args.get("filepath"), "occurrences": r.occurrences, "strategy": r.strategy,
                "diff": diff, "is_truncated": len(r.diff) > MAX_STDOUT}, True
    except Exception as e:
        return {"status": "error", "error_type": "FileEditError" if name == "edit_file" else "FileWriteError", "error_message": str(e)}, False


def ws_patch(ws, created):
    for c in created:
        subprocess.run(["git", "add", "-N", c], cwd=ws, capture_output=True)
    r = subprocess.run(["git", "diff"], cwd=ws, capture_output=True, text=True, errors="replace")
    return r.stdout


# ---------------------------------------------------------------- LLM
def chat(api, body, timeout=900):
    rq = urllib.request.Request(f"{api}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        return json.loads(r.read())


def inject(msgs, text):
    """Add `text` as budget_warning to the last tool message (JSON) -- the harness's own injection channel."""
    msgs = json.loads(json.dumps(msgs))
    for m in reversed(msgs):
        if m.get("role") == "tool":
            c = flat(m.get("content"))
            try:
                d = json.loads(c)
                if isinstance(d, dict):
                    d["budget_warning"] = (d.get("budget_warning", "") + " " + text).strip()
                    m["content"] = json.dumps(d, ensure_ascii=False)
                    return msgs, True
            except Exception:
                pass
            m["content"] = c + f"\n\nbudget_warning: {text}"
            return msgs, True
        if m.get("role") in ("user", "assistant"):
            break
    return msgs, False


def commit_episode(api, run, k, j, max_calls, natural, model=None):
    req = run["calls"][k]["req"]
    msgs, ok = inject(req["messages"], FORCE)
    rec = {"iid": run["iid"], "tag": run["tag"], "k": k, "natural": natural, "j": j, "injected": ok, "model": model or req.get("model")}
    if not ok:
        rec["skip"] = "last message not a tool response"
        return rec
    seed = int(hashlib.sha1(f"r3:{run['tag']}:{run['iid']}:{k}:{j}".encode()).hexdigest()[:8], 16)
    ws, created = None, set()
    calls, comp_tokens, edits_ok, edits_err = [], 0, 0, 0
    try:
        for step in range(max_calls):
            body = {x: v for x, v in req.items() if x not in ("messages", "seed")}
            body.update(messages=msgs, seed=seed + step)
            if model:
                body["model"] = model  # P3b: another size of the ladder commits from this run's context
            resp = chat(api, body)
            ch = resp["choices"][0]; msg = ch.get("message") or {}
            comp_tokens += (resp.get("usage") or {}).get("completion_tokens", 0)
            tcs = msg.get("tool_calls") or []
            names = [t["function"]["name"] for t in tcs]
            calls.append(names)
            if not any(n in EDIT for n in names):
                break
            if ws is None:
                ws = make_ws(run["iid"])
            am = {"role": "assistant", "tool_calls": tcs}
            if msg.get("reasoning") or msg.get("reasoning_content"):
                am["reasoning_content"] = msg.get("reasoning_content") or msg.get("reasoning")
                am["reasoning"] = msg.get("reasoning") or msg.get("reasoning_content")
            tool_msgs = []
            for t in tcs:
                n = t["function"]["name"]
                try:
                    args = json.loads(t["function"]["arguments"])
                except Exception:
                    args = {}
                if n in EDIT:
                    out, good = apply_tool(ws, n, args, created)
                    edits_ok += good; edits_err += (not good)
                else:
                    out = {"status": "error", "error_type": "NotExecuted", "error_message": "commit episode: only edits are executed"}
                tool_msgs.append({"role": "tool", "tool_call_id": t.get("id"), "content": json.dumps(out, ensure_ascii=False)})
            msgs = msgs + [am] + tool_msgs
            msgs, _ = inject(msgs, FOLLOW)
        patch = ws_patch(ws, created) if ws is not None else ""
    finally:
        if ws is not None:
            shutil.rmtree(ws, ignore_errors=True)
    pf = set(re.findall(r"^\+\+\+ b/(\S+)", patch, re.M))
    rec.update(seed=seed, calls=calls, compliant=bool(calls and any(n in EDIT for n in calls[0])), edits_ok=edits_ok,
               edits_err=edits_err, comp_tokens=comp_tokens, patch=patch, patch_files=sorted(pf),
               touches_gold=bool(pf & gold_files(run["iid"])))
    return rec


# ---------------------------------------------------------------- token-level forced commits (prefill)
CH_OPEN, CH_CLOSE, TC_OPEN, TC_CLOSE, TURN_END, TOOL_RESP = 100, 101, 48, 49, 106, 50
STR = '<|"|>'


def post(api, path, body, timeout=900):
    rq = urllib.request.Request(f"{api}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        return json.loads(r.read())


def render(api, req, msgs, model=None):
    body = {"model": model or req.get("model"), "messages": msgs, "tools": req.get("tools"), "add_generation_prompt": True,
            "chat_template_kwargs": req.get("chat_template_kwargs")}
    return post(api, "/tokenize", body)["tokens"]


_TXT = {}


def text_ids(api, model, text):
    if text not in _TXT:
        _TXT[text] = post(api, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]
    return _TXT[text]


def complete(api, model, ids, max_tokens, req, seed, stop_ids, stop=None):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens,
            "temperature": TEMP if TEMP is not None else req.get("temperature", 0.2),
            "top_p": req.get("top_p", 1.0), "seed": seed, "stop_token_ids": stop_ids, "skip_special_tokens": False,
            "return_token_ids": True}
    if stop:
        body.update(stop=stop, include_stop_str_in_output=True)
    ch = post(api, "/v1/completions", body)["choices"][0]
    return ch.get("token_ids") or [], ch.get("text") or "", ch.get("stop_reason"), ch.get("finish_reason")


def parse_call(text):
    args = {}
    for k, v in re.findall(r'(\w+):' + re.escape(STR) + r'(.*?)' + re.escape(STR), text, re.S):
        args[k] = v
    for k, v in re.findall(r'(\w+):(true|false|\d+)(?=[,}])', text):
        args.setdefault(k, v)
    return args


def forced_episode(api, run, k, j, max_calls, natural, model=None):
    """Thinking runs as usual; at its end the start of an edit_file call is prefilled, so the agent must commit."""
    req = run["calls"][k]["req"]
    msgs, ok = inject(req["messages"], FORCE) if INJECT else (json.loads(json.dumps(req["messages"])), False)
    mdl = model or req.get("model")
    rec = {"iid": run["iid"], "tag": run["tag"], "k": k, "natural": natural, "j": j, "injected": ok, "model": mdl, "mode": "prefill",
           "temp": TEMP}
    seed = int(hashlib.sha1(f"r3f:{run['tag']}:{run['iid']}:{k}:{j}".encode()).hexdigest()[:8], 16)
    ids = render(api, req, msgs, mdl)
    budget = req.get("thinking_token_budget") or 4096
    g1, t1, stop1, fin1 = complete(api, mdl, ids, budget, req, seed, [CH_CLOSE, TC_OPEN, TURN_END])
    if not g1 and t1:  # completions endpoint without token ids: fall back to re-tokenising the text (approximate)
        g1 = text_ids(api, mdl, t1)
    # the chat template usually opens the thought channel in the prompt (completion starts inside the thought)
    opened = (CH_OPEN in ids[-20:] or CH_OPEN in g1) and CH_CLOSE not in g1
    if stop1 == TC_OPEN or (g1 and g1[-1] == TC_OPEN):
        prefix = ([] if g1 and g1[-1] == TC_OPEN else [TC_OPEN])
        model_started_call = True
    else:
        prefix = ([CH_CLOSE] if opened else []) + [TC_OPEN]
        model_started_call = False
    prefix += text_ids(api, mdl, "call:edit_file{")
    g2, t2, stop2, fin2 = complete(api, mdl, ids + g1 + prefix, 4096, req, seed + 1, [TC_CLOSE, TOOL_RESP, TURN_END])
    args = parse_call("{" + t2)
    rec.update(seed=seed, think_tokens=len(g1), think_stop=stop1, model_started_call=model_started_call,
               call_tokens=len(g2), call_finish=fin2, parsed=sorted(args), call_text_head=t2[:300])
    return _finish_forced(api, run, req, msgs, mdl, seed, j, args, t1, max_calls, rec)


def _finish_forced(api, run, req, msgs, mdl, seed, j, args, t1, max_calls, rec):
    """Apply the forced edit, then let the agent continue with edit-only calls through the chat API."""
    ws, created = make_ws(run["iid"]), set()
    try:
        out, good = apply_tool(ws, "edit_file", args, created)
        edits_ok, edits_err = int(good), int(not good)
        thinking = re.sub(r"<[|][^>]*>|<[a-z_]+[|]>|</?think>", "", t1).strip()
        tc = {"id": f"call_forced_{j}", "type": "function", "function": {"name": "edit_file", "arguments": json.dumps(args)}}
        msgs2 = msgs + [{"role": "assistant", "tool_calls": [tc], "reasoning_content": thinking, "reasoning": thinking},
                        {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out, ensure_ascii=False)}]
        if INJECT:
            msgs2, _ = inject(msgs2, FOLLOW)
        calls = [["edit_file(forced)"]]
        for step in range(max_calls - 1):  # natural edit-only continuation
            body = {x: v for x, v in req.items() if x not in ("messages", "seed")}
            body.update(messages=msgs2, seed=seed + 2 + step, model=mdl)
            if TEMP is not None:
                body["temperature"] = TEMP
            resp = chat(api, body)
            msg = resp["choices"][0].get("message") or {}
            tcs = msg.get("tool_calls") or []
            names = [t["function"]["name"] for t in tcs]; calls.append(names)
            if not any(n in EDIT for n in names):
                break
            am = {"role": "assistant", "tool_calls": tcs}
            if msg.get("reasoning") or msg.get("reasoning_content"):
                am["reasoning_content"] = msg.get("reasoning_content") or msg.get("reasoning")
                am["reasoning"] = am["reasoning_content"]
            tms = []
            for t in tcs:
                try:
                    a2 = json.loads(t["function"]["arguments"])
                except Exception:
                    a2 = {}
                if t["function"]["name"] in EDIT:
                    o2, g = apply_tool(ws, t["function"]["name"], a2, created); edits_ok += g; edits_err += (not g)
                else:
                    o2 = {"status": "error", "error_type": "NotExecuted", "error_message": "commit episode: only edits are executed"}
                tms.append({"role": "tool", "tool_call_id": t.get("id"), "content": json.dumps(o2, ensure_ascii=False)})
            msgs2 = msgs2 + [am] + tms
            if INJECT:
                msgs2, _ = inject(msgs2, FOLLOW)
        patch = ws_patch(ws, created)
    finally:
        shutil.rmtree(ws, ignore_errors=True)
    pf = set(re.findall(r"^\+\+\+ b/(\S+)", patch, re.M))
    rec.update(calls=calls, compliant=True, edits_ok=edits_ok, edits_err=edits_err, patch=patch, patch_files=sorted(pf),
               touches_gold=bool(pf & gold_files(run["iid"])))
    return rec


def parse_xml_call(text):
    """qwen3_coder format: <parameter=NAME>\nVALUE\n</parameter> blocks."""
    args = {}
    for k, v in re.findall(r"<parameter=(\w+)>(.*?)</parameter>", text, re.S):
        if v.startswith("\n"):
            v = v[1:]
        if v.endswith("\n"):
            v = v[:-1]
        args[k] = v
    return args


def forced_episode_xml(api, run, k, j, max_calls, natural, model=None):
    """Qwen3.5/3.6 (qwen3_coder XML tool calls): thinking until </think>, then '<tool_call><function=edit_file>' is
    prefilled; the rest of the episode (edit-only continuation) is identical to forced_episode."""
    req = run["calls"][k]["req"]
    msgs, ok = inject(req["messages"], FORCE) if INJECT else (json.loads(json.dumps(req["messages"])), False)
    mdl = model or req.get("model")
    rec = {"iid": run["iid"], "tag": run["tag"], "k": k, "natural": natural, "j": j, "injected": ok, "model": mdl, "mode": "prefill-xml",
           "temp": TEMP}
    seed = int(hashlib.sha1(f"r3f:{run['tag']}:{run['iid']}:{k}:{j}".encode()).hexdigest()[:8], 16)
    ids = render(api, req, msgs, mdl)
    budget = req.get("thinking_token_budget") or 4096
    g1, t1, stop1, fin1 = complete(api, mdl, ids, budget, req, seed, [], stop=["</think>", "<tool_call>"])
    if t1.endswith("<tool_call>"):
        t1 = t1[: -len("<tool_call>")]
    if "</think>" not in t1:
        t1 += "\n</think>\n\n"
    else:
        t1 += "\n\n"
    prefix = "<tool_call>\n<function=edit_file>\n<parameter=filepath>\n"
    ids2 = ids + text_ids(api, mdl, t1 + prefix)
    g2, t2, stop2, fin2 = complete(api, mdl, ids2, 4096, req, seed + 1, [], stop=["</tool_call>", "</function>"])
    args = parse_xml_call("<parameter=filepath>\n" + t2)
    rec.update(seed=seed, think_tokens=len(g1), think_stop=stop1, model_started_call=False,
               call_tokens=len(g2), call_finish=fin2, parsed=sorted(args), call_text_head=t2[:300])
    return _finish_forced(api, run, req, msgs, mdl, seed, j, args, t1, max_calls, rec)


def cmd_check_render(a):
    """Exactness check: /tokenize(messages) must equal the logged prompt_token_ids (token-id logs only)."""
    n = same = 0
    for f in sorted(glob.glob(os.path.join(LLM, a.logs)))[:4]:
        for line in open(f):
            r = json.loads(line)
            if not isinstance(r.get("resp"), dict) or not r["resp"].get("prompt_token_ids"):
                continue
            ids = render(a.api, r["req"], r["req"]["messages"])
            n += 1; same += ids == r["resp"]["prompt_token_ids"]
            if n >= a.n:
                break
        if n >= a.n:
            break
    print(f"render check: {same}/{n} prompts token-identical to the logged prompt_token_ids")


def sel_file(samples_name):
    """Selection metadata per samples file (R3 and R3f share the original selection.json)."""
    if samples_name in ("samples.jsonl", "samples_prefill.jsonl"):
        return OUT / "selection.json"
    return OUT / f"selection_{samples_name.rsplit('.', 1)[0]}.json"


def cmd_sample(a):
    global INJECT, TEMP
    INJECT, TEMP = not a.no_instruction, a.temperature
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.logs)
    forks = [int(x) for x in a.forks.split(",")]
    tables = {}
    pool = defaultdict(list)
    for r in runs:
        key = (r["run"], r["seed"])
        if key not in tables:
            tables[key] = outcome_table(*key)
        o = tables[key].get(r["iid"])
        if not o or not snapshot_of(r["iid"]):
            continue
        r["cls"] = classify(r["iid"], o)
        r["pts"], r["natural"], r["first_edit"], r["unsafe_after"] = fork_points(r, forks)
        if r["pts"] or r["natural"] is not None:
            pool[r["cls"]].append(r)
    rnd = random.Random(a.select_seed)
    chosen = []
    if a.runs_file:  # explicit (tag, iid) list, e.g. E1's task-matched boundary pairs; class and n-per-class ignored
        want = {(x["tag"], x["iid"]) for x in json.load(open(a.runs_file))}
        chosen = sorted((r for rs in pool.values() for r in rs if (r["tag"], r["iid"]) in want), key=lambda r: (r["iid"], r["tag"]))
        print("runs-file:", len(want), "requested,", len(chosen), "with fork points")
    else:
        for cls in a.classes.split(","):
            rs = pool.get(cls, [])
            rnd.shuffle(rs)
            chosen += rs[: a.n_per_class]
    print("pool sizes:", {c: len(v) for c, v in pool.items()}, "| chosen:", Counter(r["cls"] for r in chosen))
    meta = [{"iid": r["iid"], "tag": r["tag"], "cls": r["cls"], "first_edit": r["first_edit"], "unsafe_after": r["unsafe_after"],
             "pts": r["pts"], "natural": r["natural"], "n_calls": len(r["calls"])} for r in chosen]
    json.dump(meta, open(sel_file(a.out), "w"), indent=1)
    done = set()
    sp = OUT / a.out
    if sp.exists():
        for l in open(sp):
            x = json.loads(l)
            if "iid" in x and not x.get("skip"):
                done.add((x["tag"], x["iid"], x["k"], x["j"]))
    jobs = []
    for r in chosen:
        ks = [(k, False) for k in r["pts"]] + ([(r["natural"], True)] if r["natural"] is not None and r["natural"] not in r["pts"] else [])
        if a.only_forks:  # selection used --forks; sample only these forks (plus each run's own commit point)
            keep = {int(x) for x in a.only_forks.split(",")}
            ks = [(k, nat) for k, nat in ks if nat or k in keep]
        for k, nat in ks:
            for j in range(a.k):
                if (r["tag"], r["iid"], k, j) not in done:
                    jobs.append((r, k, j, nat))
    print("jobs:", len(jobs))
    if a.force == "prefill" and jobs and not done:
        # smoke check before spending the GPU budget: the forced call must parse into a real edit
        probe = []
        for r, k, j, nat in jobs[:4]:
            try:
                probe.append((forced_episode_xml if a.format == "qwen3xml" else forced_episode)(a.api, r, k, j, a.max_commit_calls, nat, a.model))
            except Exception as e:
                probe.append({"error": repr(e)[:300]})
        good = [x for x in probe if {"old_string", "new_string"} <= set(x.get("parsed", []))]
        print("prefill smoke:", len(good), "/ 4 parsed edits;", [x.get("error") or x.get("call_text_head", "")[:120] for x in probe], flush=True)
        if not good:
            sys.exit("prefill smoke check failed: no forced call parsed into old_string/new_string")
    with ThreadPoolExecutor(a.par) as ex, open(sp, "a") as f:
        fn = (forced_episode_xml if a.format == "qwen3xml" else forced_episode) if a.force == "prefill" else commit_episode
        futs = [ex.submit(fn, a.api, r, k, j, a.max_commit_calls, nat, a.model) for r, k, j, nat in jobs]
        for i, fu in enumerate(as_completed(futs)):
            try:
                rec = fu.result()
            except Exception as e:
                rec = {"error": repr(e)[:300]}
            if "iid" in rec:
                rec["cls"] = next(m["cls"] for m in meta if m["iid"] == rec["iid"] and m["tag"] == rec["tag"])
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------- grading (official evaluator, CPU)
def free_gb(path=WORK):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def cmd_grade(a):
    comp = OUT / "comp"; (comp / "snapshots").mkdir(parents=True, exist_ok=True)
    patches, tasks = {}, {}
    grades = load_grades()
    for l in (l for fn in a.samples.split(",") for l in open(OUT / fn)):
        x = json.loads(l)
        if not x.get("patch"):
            continue
        h = hashlib.sha1((x["iid"] + "\n" + x["patch"]).encode()).hexdigest()[:12]
        if h in grades:
            continue
        vid = f"{x['iid']}__pl{h}"
        patches[vid] = x["patch"]
        t = dict(TASKS[x["iid"]][0]); t["instance_id"] = vid
        tasks[vid] = t
        link = comp / "snapshots" / f"{vid}.tgz"
        if not link.exists():
            link.symlink_to(snapshot_of(x["iid"]))
    if not tasks:
        print("nothing to grade"); return
    with open(comp / "tasks.jsonl", "w") as f:
        for t in tasks.values():
            f.write(json.dumps(t) + "\n")
    json.dump(patches, open(comp / "patches.json", "w"))
    # keep all patches of one task in one process: the first grade of a task pays the environment setup (~2 min),
    # later ones reuse it (~15 s). Tasks are graded in chunks; our own cache misses are deleted after each chunk.
    groups = defaultdict(list)
    for vid in sorted(tasks):
        groups[vid.rsplit("__pl", 1)[0]].append(vid)
    glist = sorted(groups.values(), key=len, reverse=True)
    env = dict(os.environ, SWEG_WHEELS=os.environ.get("SWEG_WHEELS", WORK + "/comp/wheels_plus"), TMPDIR="/tmp")
    n_built = 0
    for c0 in range(0, len(glist), a.chunk_tasks):
        if free_gb() < 30:
            print("disk guard: $PL_WORK free < 30 GB, stopping"); break
        # read-only view of the shared environment cache: hits are read through symlinks; a miss is built in our own
        # folder (never in $PL_WORK/envcache), and the cache's evictor cannot rmtree symlinked entries. (Not under /tmp:
        # the subprocess sandbox rewrites /tmp in commands to its private tmp dir.)
        farm = FARM; farm.mkdir(parents=True, exist_ok=True)
        for e in farm.iterdir():  # drop links to entries the shared cache has evicted (or from another box)
            if e.is_symlink() and not e.exists():
                e.unlink()
        for e in Path(WORK + "/envcache").iterdir():
            if e.is_dir() and not e.name.startswith(".") and (e / "ok").exists() and not (farm / e.name).exists():
                (farm / e.name).symlink_to(e)
        shards = [[] for _ in range(a.procs)]
        for g in glist[c0:c0 + a.chunk_tasks]:
            min(shards, key=len).extend(g)
        batch = time.strftime("%m%d%H%M%S")
        procs = []
        for s, sub in enumerate(shards):
            if not sub:
                continue
            idf = comp / f"ids_{s}.txt"; idf.write_text("\n".join(sub))
            procs.append(subprocess.Popen(["nice", "-n", "10", WORK + "/venv-harness/bin/python", "-I", os.path.abspath(__file__),
                                           "grade-shard", "--shard", str(s), "--ids", str(idf), "--batch", batch], env=env,
                                          stdout=open(OUT / f"grade_{s}.log", "a"), stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
        built = [e for e in farm.iterdir() if e.is_dir() and not e.is_symlink() and not e.name.startswith(".")]
        n_built += len(built)
        for e in built:
            shutil.rmtree(e, ignore_errors=True)
        print(f"chunk {c0 // a.chunk_tasks + 1}: graded so far {len(load_grades())}; own cache misses built+deleted {len(built)}", flush=True)
    print("graded:", len(load_grades()), "| envcache misses built under", FARM, ":", n_built)


def cmd_grade_shard(a):
    import asyncio
    sys.path.insert(0, WORK + "/src/eval")
    import sandbox_patch  # noqa: F401  (must precede Evaluator construction)
    import swegemma.sandbox.subprocess as _sp
    _orig_exec = _sp.SubprocessManager.exec

    def _exec(self, sandbox_id, command, **kw):  # the sandbox env is sanitised: pass the cache view on the command line
        if "sandbox_install.py" in command:
            command = f"SWEG_ENV_CACHE={FARM} " + command
        return _orig_exec(self, sandbox_id, command, **kw)
    _sp.SubprocessManager.exec = _exec
    from adk_submission.registry import ModelRegistry
    from swegemma.config import EvalConfig
    from swegemma.evaluate import Evaluator
    from adk_eval_core.tracing.trace import SessionTrace
    comp = OUT / "comp"
    patches = json.load(open(comp / "patches.json"))
    ids = [x for x in Path(a.ids).read_text().split("\n") if x]

    class PatchEvaluator(Evaluator):
        async def _run_agent_sandbox(self, task, snapshot_path, task_index, total_tasks, **kw):
            if free_gb() < 30:
                raise RuntimeError("disk guard: $PL_WORK free < 30 GB")
            return patches.get(task.instance_id, ""), None, SessionTrace()

    out = OUT / "grade" / f"{a.batch}_s{a.shard:02d}"; out.mkdir(parents=True, exist_ok=True)
    cfg = EvalConfig(tasks_path=comp / "tasks.jsonl", snapshots_dir=comp / "snapshots", results_dir=out,
                     submission_dir=Path(WORK + "/bundles/empty"), models=ModelRegistry(), sandbox="subprocess",
                     wheels_dir=Path(os.environ.get("SWEG_WHEELS", WORK + "/comp/wheels_plus")),
                     graph_dir=str(comp / "graphs"), embeddings_dir=str(comp / "embeddings"),
                     verification_timeout_seconds=300, task_ids=ids, display_mode="quiet")
    res = asyncio.run(PatchEvaluator(cfg).run())
    print(json.dumps({"shard": a.shard, "resolved": res.resolved, "total": res.total}))


def load_grades():
    g = {}
    for f in glob.glob(str(OUT / "grade" / "*" / "task_results.jsonl")):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            h = r["instance_id"].rsplit("__pl", 1)[-1]
            g[h] = {"resolved": bool(r.get("resolved")), "error": r.get("error")}
    return g


# ---------------------------------------------------------------- replay / ever-green audit
def pm_lines(patch):
    return sorted(l for l in patch.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))


def cmd_replay(a):
    """Re-apply each selected run's own logged edits, in order, to the base snapshot; one record per edit with the
    cumulative patch (grade with `grade --samples replay.jsonl`). Validates the apply path (final replay patch vs the
    run's saved patch) and gives the ever-green audit (did an intermediate tree pass while the final one failed?)."""
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.logs); tables = {}; pool = defaultdict(list)
    for r in runs:
        key = (r["run"], r["seed"])
        tables.setdefault(key, outcome_table(*key))
        o = tables[key].get(r["iid"])
        if o and snapshot_of(r["iid"]):
            r["cls"] = classify(r["iid"], o); pool[r["cls"]].append(r)
    rnd = random.Random(a.select_seed); chosen = []
    for cls in a.classes.split(","):
        rs = pool.get(cls, []); rnd.shuffle(rs); chosen += rs[: a.n_per_class]
    done = set()
    if (OUT / a.out).exists():
        for l in open(OUT / a.out):
            x = json.loads(l)
            if x.get("kind") == "replay_final":
                done.add((x["tag"], x["iid"]))
    chosen = [r for r in chosen if (r["tag"], r["iid"]) not in done]
    print("runs to replay:", len(chosen), flush=True)
    with open(OUT / a.out, "a") as f:
        for r in chosen:
            ws, created = make_ws(r["iid"]), set()
            e = 0; unsafe = None; last_patch = ""
            try:
                for i, c in enumerate(r["calls"]):
                    for name, arg in c["tools"]:
                        try:
                            args = json.loads(arg)
                        except Exception:
                            args = {}
                        if name == "run_command" and unsafe is None and unsafe_command(str(args.get("command", "")), tracked_files(r["iid"])):
                            unsafe = i
                        if name in EDIT and not str(args.get("filepath", "")).startswith("/tmp"):
                            out, good = apply_tool(ws, name, args, created)
                            last_patch = ws_patch(ws, created)
                            f.write(json.dumps({"kind": "replay", "iid": r["iid"], "tag": r["tag"], "cls": r["cls"], "k": i, "j": e,
                                                "edit_ok": good, "unsafe_before": unsafe is not None, "patch": last_patch,
                                                "touches_gold": bool(set(re.findall(r"^\+\+\+ b/(\S+)", last_patch, re.M)) & gold_files(r["iid"]))}) + "\n")
                            e += 1
            finally:
                shutil.rmtree(ws, ignore_errors=True)
            saved = ""
            for pp in glob.glob(f"{RES}/{r['run']}/seed{r['seed']}/shard*/patches/{r['iid']}.patch"):
                saved = open(pp, errors="replace").read()
            f.write(json.dumps({"kind": "replay_final", "iid": r["iid"], "tag": r["tag"], "cls": r["cls"], "n_edits": e,
                                "unsafe_cmd_at": unsafe, "matches_saved_patch": pm_lines(last_patch) == pm_lines(saved)}) + "\n")
            f.flush()
            print(r["iid"], r["cls"], "edits", e, "matches saved patch", pm_lines(last_patch) == pm_lines(saved), "unsafe", unsafe, flush=True)


def cmd_replay_report(a):
    """Ever-green audit over replayed runs whose replay reproduces the saved patch (edits all via edit/write_file)."""
    grades = load_grades()
    R = [json.loads(l) for l in open(OUT / a.out)]
    fin = {(x["tag"], x["iid"]): x for x in R if x["kind"] == "replay_final"}
    n_calls = {}
    seq = defaultdict(list)
    for x in R:
        if x["kind"] == "replay" and x.get("patch"):
            h = hashlib.sha1((x["iid"] + "\n" + x["patch"]).encode()).hexdigest()[:12]
            seq[(x["tag"], x["iid"])].append((x["j"], x["k"], (grades.get(h) or {}).get("resolved")))
    c = Counter(); first_green_call = defaultdict(list); lost = []
    for key, f in fin.items():
        c[f"runs_{f['cls']}"] += 1
        if not f.get("matches_saved_patch"):
            c[f"not_reproducible_{f['cls']}"] += 1; continue
        v = sorted(seq.get(key, []))
        if any(g is None for _, _, g in v):
            c[f"ungraded_{f['cls']}"] += 1; continue
        cls = f["cls"]; c[f"repro_{cls}"] += 1
        res = [g for _, _, g in v]
        if not res:
            continue
        c[f"first_edit_green_{cls}"] += bool(res[0])
        c[f"ever_green_{cls}"] += any(res)
        c[f"final_green_{cls}"] += bool(res[-1])
        if any(res):
            first_green_call[cls].append(next(k for _, k, g in v if g))
        if any(res) and not res[-1]:
            lost.append((key[1], cls, [int(g) for g in res]))
    lines = ["ever-green audit (runs whose edit_file replay reproduces the saved patch):", json.dumps(dict(sorted(c.items())))]
    for cls, ks in first_green_call.items():
        ks.sort(); lines.append(f"  {cls}: first green at LLM call median {ks[len(ks) // 2]} (n={len(ks)}, q25 {ks[len(ks) // 4]}, q75 {ks[3 * len(ks) // 4]})")
    lines.append(f"  lost green (an intermediate tree passed, the final one failed): {len(lost)} " + json.dumps(lost[:20]))
    open(OUT / a.out.replace(".jsonl", "_report.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


# ---------------------------------------------------------------- analysis
def cmd_analyze(a):
    grades = load_grades()
    S = [json.loads(l) for l in open(OUT / a.samples)]
    meta = {(m["tag"], m["iid"]): m for m in json.load(open(sel_file(a.samples)))}
    rows = []
    for x in S:
        if "iid" not in x or x.get("skip"):
            continue
        if x.get("patch"):
            h = hashlib.sha1((x["iid"] + "\n" + x["patch"]).encode()).hexdigest()[:12]
            g = grades.get(h)
            if g is None:
                continue
            res = g["resolved"]
        else:
            res = False
        rows.append({**{k: x.get(k) for k in ("iid", "tag", "k", "natural", "j", "compliant", "touches_gold", "cls")}, "res": res})
    lines = [f"R3 forced-commit values: {len(rows)} graded samples, {len({(r['tag'], r['iid']) for r in rows})} runs"]
    by = defaultdict(list)
    for r in rows:
        by[(r["cls"], "nat" if r["natural"] else r["k"])].append(r)
    for (cls, k), v in sorted(by.items(), key=lambda t: (t[0][0], str(t[0][1]))):
        lines.append(f"  {cls:7s} k={str(k):>4s}: n={len(v):3d}  compliant={sum(x['compliant'] for x in v) / len(v):.2f}  "
                     f"touches gold={sum(x['touches_gold'] for x in v) / len(v):.2f}  Q_c={sum(x['res'] for x in v) / len(v):.3f}")
    # per-run curves and stopping regret
    run = defaultdict(lambda: defaultdict(list))
    for r in rows:
        run[(r["tag"], r["iid"])]["nat" if r["natural"] else r["k"]].append(r["res"])
    reg = []
    for key, d in run.items():
        m = meta.get(key, {})
        fe = m.get("first_edit")
        ks = sorted(k for k in d if k != "nat")
        ready = next((k for k in ks if sum(d[k]) / len(d[k]) >= 0.5), None)
        if m.get("cls") == "res" and fe is not None and ready is not None:
            reg.append(fe - ready)
    if reg:
        reg.sort()
        lines.append(f"  resolved runs with a fork reaching Q_c >= 0.5: {len(reg)}; calls between readiness and actual first edit: "
                     f"median {reg[len(reg) // 2]}, q25 {reg[len(reg) // 4]}, q75 {reg[3 * len(reg) // 4]}")
    open(OUT / ("report.txt" if a.samples == "samples.jsonl" else f"report_{a.samples.rsplit('.', 1)[0]}.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--logs", default="traj_s1_seed0_shard*.jsonl")
    s.add_argument("--forks", default="6,10,14,18")
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--n-per-class", type=int, default=20)
    s.add_argument("--classes", default="res,wrongG,empty")
    s.add_argument("--select-seed", type=int, default=0)
    s.add_argument("--max-commit-calls", type=int, default=3)
    s.add_argument("--par", type=int, default=8)
    s.add_argument("--api", default="http://127.0.0.1:8000")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--model", default=None, help="served model name to commit with (P3b cross-model); default: as logged")
    s.add_argument("--out", default="samples.jsonl", help="samples file under $PL_WORK/paper_lab/r3/ (use one per model)")
    s.add_argument("--format", choices=["gemma4", "qwen3xml"], default="gemma4", help="tool-call format for --force prefill")
    s.add_argument("--runs-file", default=None, help="JSON list of {tag, iid}: sample exactly these runs (E1)")
    s.add_argument("--no-instruction", action="store_true", help="prefill without injecting COMMIT NOW / FOLLOW (M3 ablation)")
    s.add_argument("--temperature", type=float, default=None, help="override the logged temperature (E3)")
    s.add_argument("--only-forks", default=None, help="sample only these fork calls (+ own point); selection still uses --forks")
    s.add_argument("--force", choices=["instruct", "prefill"], default="instruct",
                   help="instruct: commit instruction only (agent may ignore it); prefill: edit_file call start prefilled after thinking")
    cr = sub.add_parser("check-render"); cr.add_argument("--logs", default="traj_v0_s2_seed0_shard*.jsonl")
    cr.add_argument("--api", default="http://127.0.0.1:8000"); cr.add_argument("--n", type=int, default=20)
    g = sub.add_parser("grade"); g.add_argument("--procs", type=int, default=4); g.add_argument("--samples", default="samples.jsonl")
    g.add_argument("--chunk-tasks", type=int, default=15)
    gs = sub.add_parser("grade-shard"); gs.add_argument("--shard", type=int); gs.add_argument("--ids"); gs.add_argument("--batch", default="b")
    rp = sub.add_parser("replay")
    rp.add_argument("--logs", default="traj_s1_seed0_shard*.jsonl")
    rp.add_argument("--n-per-class", type=int, default=3)
    rp.add_argument("--classes", default="res")
    rp.add_argument("--select-seed", type=int, default=0)
    rp.add_argument("--out", default="replay.jsonl")
    rr = sub.add_parser("replay-report"); rr.add_argument("--out", default="replay.jsonl")
    an = sub.add_parser("analyze"); an.add_argument("--samples", default="samples.jsonl")
    a = ap.parse_args()
    if a.cmd == "sample":
        if a.dry_run:
            runs = load_runs(a.logs); forks = [int(x) for x in a.forks.split(",")]
            c = Counter(); npts = Counter(); tables = {}
            for r in runs:
                key = (r["run"], r["seed"])
                tables.setdefault(key, outcome_table(*key))
                o = tables[key].get(r["iid"])
                if not o:
                    c["no_outcome"] += 1; continue
                cls = classify(r["iid"], o)
                pts, nat, fe, ua = fork_points(r, forks)
                c[cls] += 1; npts[(cls, len(pts) + (nat is not None))] += 1
                c["unsafe_before_first_edit"] += ua is not None and (fe is None or ua < fe)
            print("runs", len(runs), dict(c)); print("fork points per run:", sorted(npts.items()))
            return
        cmd_sample(a)
    elif a.cmd == "grade":
        cmd_grade(a)
    elif a.cmd == "grade-shard":
        cmd_grade_shard(a)
    elif a.cmd == "check-render":
        cmd_check_render(a)
    elif a.cmd == "replay":
        cmd_replay(a)
    elif a.cmd == "replay-report":
        cmd_replay_report(a)
    else:
        cmd_analyze(a)


if __name__ == "__main__":
    main()

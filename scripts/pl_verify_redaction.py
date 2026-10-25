# SPDX-License-Identifier: Apache-2.0
"""paper-lab: verify that the public (redacted) slice carries no source text from the licence-flagged repositories.
Compares the public copy with the internal, unredacted slice. Reports, per file and field:
  - for the flagged tasks: value types and whether any string could hold code (patch fields must be "" or sha256:<hex>);
  - for every row of every file: hits of source fragments from the flagged repositories, i.e. diff lines (context,
    removed, added; >= 20 characters, not occurring in any unflagged task's patches) and file paths from diff headers.
Stdlib only. Usage: python3 pl_verify_redaction.py <internal slice> <public slice>   (exit 1 on any problem)"""
import collections, csv, gzip, json, os, re, sys

INT, PUB = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(PUB, "repo_licences.tsv")), delimiter="\t"))
flag = {r["repo"] for r in rows if r["not_plain_permissive"] == "yes"}
T = {t for r in rows if r["repo"] in flag for t in r["task_ids"].split(",")}
SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
PATCH_FIELDS = {"patch", "cumulative_patch"}


def records(d, name):
    p = os.path.join(d, name)
    if name.endswith(".jsonl.gz"):
        return [json.loads(l) for l in gzip.open(p, "rt")]
    if name.endswith(".jsonl"):
        return [json.loads(l) for l in open(p)]
    if name.endswith(".json"):
        x = json.load(open(p))
        return x if isinstance(x, list) else [x]
    return [{"_text": open(p, encoding="utf-8").read()}]


# source fragments of the flagged repositories, from the unredacted patches
def diff_parts(text):
    lines, paths = set(), set()
    for l in (text or "").split("\n"):
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", l)
        if m:
            paths.update(m.groups()); continue
        if l.startswith(("+++", "---", "@@", "index ", "new file", "deleted file")):
            continue
        body = l[1:].strip() if l[:1] in " +-" else l.strip()
        if len(body) >= 20:
            lines.add(body)
    return lines, paths


flag_lines, flag_paths, other_lines = set(), set(), set()
for name, key in (("forced_commits.jsonl.gz", "patch"), ("evergreen_edits.jsonl.gz", "cumulative_patch")):
    for x in records(INT, name):
        ls, ps = diff_parts(x.get(key))
        if x["task"] in T:
            flag_lines |= ls; flag_paths |= ps
        else:
            other_lines |= ls
frag = flag_lines - other_lines  # lines attributable to the flagged repositories only
paths = {p for p in flag_paths if "/" in p or "." in p}
print(f"flagged repositories: {len(flag)}; tasks: {len(T)}; distinctive source lines: {len(frag)} "
      f"(of {len(flag_lines)}; {len(flag_lines - frag)} also occur in other repositories' patches); file paths: {len(paths)}")


def strings(v, path=""):
    if isinstance(v, str):
        yield path, v
    elif isinstance(v, dict):
        for k, w in v.items():
            yield from strings(w, f"{path}.{k}" if path else k)
    elif isinstance(v, list):
        for w in v:
            yield from strings(w, path + "[]")


problems = 0
for name in sorted(os.listdir(PUB)):
    recs = records(PUB, name)
    per_field = collections.defaultdict(lambda: collections.Counter())
    for x in recs:
        flagged = x.get("task") in T if isinstance(x, dict) else False
        for field, s in strings(x):
            fld = field.split(".")[0].replace("[]", "")
            c = per_field[field]
            c["values"] += 1
            if flagged:
                c["flagged-task values"] += 1
                if fld in PATCH_FIELDS:
                    c["flagged patch: empty"] += s == ""
                    c["flagged patch: sha256"] += bool(SHA.match(s))
                    c["flagged patch: OTHER"] += not (s == "" or SHA.match(s))
                elif "\n" in s or len(s) > 200:
                    c["flagged: long or multi-line string"] += 1
            if fld in PATCH_FIELDS or field == "_text" or "\n" in s or len(s) > 40:
                body = {l[1:].strip() if l[:1] in " +-" else l.strip() for l in s.split("\n")}
                c["source-line hits"] += len(body & frag)
                c["file-path hits"] += sum(1 for p in paths if p in s and not (field == "_text" and p in {r["repo"] for r in rows}))
    print(f"== {name}: {len(recs)} records")
    for field, c in sorted(per_field.items()):
        bad = c["flagged patch: OTHER"] + c["flagged: long or multi-line string"] + c["source-line hits"] + c["file-path hits"]
        problems += bad
        keys = [k for k in ("flagged-task values", "flagged patch: empty", "flagged patch: sha256", "flagged patch: OTHER",
                            "flagged: long or multi-line string", "source-line hits", "file-path hits") if c[k] or k in
                ("source-line hits", "file-path hits")]
        print(f"   {field:28s} values {c['values']:6d}  " + "  ".join(f"{k} {c[k]}" for k in keys) + ("  <-- CHECK" if bad else ""))
print("RESULT:", "no source text from the flagged repositories found" if not problems else f"{problems} problems")
sys.exit(1 if problems else 0)

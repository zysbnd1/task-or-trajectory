# SPDX-License-Identifier: Apache-2.0
"""paper-lab: optional release variant — replace the patch text of tasks from repositories flagged as not plainly
permissive (repo_licences.tsv, column not_plain_permissive = yes) by "sha256:<hex of the text>". Empty patches stay
empty, distinct patches stay distinct, so pl_slice_check.py prints identical output (verified 2026-10-25).
Stdlib only. Usage: python3 pl_redact_slice.py <slice dir> <output dir>  (the output dir must not exist)."""
import csv, gzip, hashlib, json, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
shutil.copytree(src, dst)
rows = list(csv.DictReader(open(os.path.join(dst, "repo_licences.tsv")), delimiter="\t"))
tasks = {t for r in rows if r["not_plain_permissive"] == "yes" for t in r["task_ids"].split(",")}
red = lambda t: ("sha256:" + hashlib.sha256(t.encode()).hexdigest()) if t else t
for name, key in (("forced_commits.jsonl.gz", "patch"), ("evergreen_edits.jsonl.gz", "cumulative_patch")):
    p = os.path.join(dst, name)
    recs = [json.loads(l) for l in gzip.open(p, "rt")]
    n = 0
    for x in recs:
        if x["task"] in tasks and x.get(key):
            x[key] = red(x[key]); n += 1
    with gzip.open(p, "wt") as g:
        for x in recs:
            g.write(json.dumps(x) + "\n")
    print(f"{name}: {n} patches redacted ({len(tasks)} tasks)")
man = os.path.join(dst, "MANIFEST.json")
m = json.load(open(man)); files = m.get("files", m)
for k, v in files.items():
    d = open(os.path.join(dst, k), "rb").read()
    v.update({"sha256": hashlib.sha256(d).hexdigest(), "bytes": len(d)})
json.dump(m, open(man, "w"), indent=1)

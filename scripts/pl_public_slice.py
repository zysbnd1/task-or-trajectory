# SPDX-License-Identifier: Apache-2.0
"""paper-lab: build the public copy of the release slice (nothing is published by this script).
- Leaves out fork_contexts.jsonl.gz (logged prompts with harness text, kept offline) and the internal MANIFEST
  (scan log, scrubbing map).
- Writes the three .jsonl.gz files decompressed (.jsonl), the form Kaggle serves (it decompresses .gz uploads), so the
  uploaded and served files, and their hashes, are the same; pass --keep-gzip to keep them compressed.
- Writes a public MANIFEST.json with sha256 and size per file.
- With --redact, replaces the patch text of tasks from repositories flagged as not plainly permissive
  (repo_licences.tsv) by "sha256:<hex>" (as pl_redact_slice.py; pl_slice_check.py output is unchanged).
Stdlib only. Usage: python3 pl_public_slice.py <slice dir> <out dir> [--redact] [--keep-gzip]   (out dir must not exist)"""
import csv, gzip, hashlib, json, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
redact = "--redact" in sys.argv[3:]
keep_gzip = "--keep-gzip" in sys.argv[3:]
INTERNAL = {"fork_contexts.jsonl.gz", "MANIFEST.json"}
os.makedirs(dst)
for name in sorted(os.listdir(src)):
    if name not in INTERNAL and not name.startswith("."):
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
if redact:
    rows = list(csv.DictReader(open(os.path.join(dst, "repo_licences.tsv")), delimiter="\t"))
    tasks = {t for r in rows if r["not_plain_permissive"] == "yes" for t in r["task_ids"].split(",")}
    red = lambda t: ("sha256:" + hashlib.sha256(t.encode()).hexdigest()) if t else t
    for name, key in (("forced_commits.jsonl.gz", "patch"), ("evergreen_edits.jsonl.gz", "cumulative_patch")):
        p = os.path.join(dst, name)
        recs = [json.loads(l) for l in gzip.open(p, "rt")]
        for x in recs:
            if x["task"] in tasks and x.get(key):
                x[key] = red(x[key])
        with gzip.open(p, "wt") as g:
            for x in recs:
                g.write(json.dumps(x) + "\n")
if not keep_gzip:
    for name in sorted(os.listdir(dst)):
        if name.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(dst, name), "rb") as g, open(os.path.join(dst, name[:-3]), "wb") as o:
                shutil.copyfileobj(g, o)
            os.remove(os.path.join(dst, name))
files = {}
for name in sorted(os.listdir(dst)):
    d = open(os.path.join(dst, name), "rb").read()
    files[name] = {"sha256": hashlib.sha256(d).hexdigest(), "bytes": len(d)}
json.dump({"files": files,
           "note": "Public release slice. Logged prompts are not included. Hashes are of the files as listed (uncompressed "
                   "JSONL); pl_slice_check.py also reads gzip-compressed copies (X.jsonl.gz)." +
                   (" Patch text of tasks from repositories flagged in repo_licences.tsv is replaced by its sha256." if redact else "")},
          open(os.path.join(dst, "MANIFEST.json"), "w"), indent=1)
print(f"public slice: {len(files)} files in {dst}" + (" (redacted)" if redact else ""))

# Task or Trajectory? — reproduction code

Code for the Kaggle writeup *Task or Trajectory? Execution-graded counterfactuals bound where a budgeted Gemma 4 coding
agent's fate is decided* (Gemma 4 Developer Agent, Paper Track).

**What can be reproduced:**
- `scripts/pl_slice_check.py` recomputes every table and the main estimates of the writeup. It reads the public
  per-episode records (Kaggle Dataset `zysbnd1/task-or-trajectory-release-slice`) and needs only the Python standard
  library (Python ≥ 3.9).
- `scripts/make_figures.py` redraws Figures 1–3.

**What cannot be reproduced from public material:** rerunning the interventions (forced commits, replays, probes)
needs:
- the logged agent prompts, which contain competition harness text and are not public;
- the competition's evaluation harness;
- GPUs.

The instrument code is included for reference.

## Reproduce the tables (about 2 s)
```bash
kaggle datasets download zysbnd1/task-or-trajectory-release-slice -p slice --unzip
python3 scripts/pl_slice_check.py slice > out.txt
diff out.txt expected/slice_check_output.txt && echo "every number reproduced"
```
- Without the Kaggle CLI, download the dataset from its Kaggle page and unzip it into `slice/`.
- The public Kaggle Notebook runs the same check on Kaggle.

File integrity (sha256 against the dataset's `MANIFEST.json`):
```bash
python3 -c "import json,hashlib; m=json.load(open('slice/MANIFEST.json'))['files']; print(all(hashlib.sha256(open('slice/'+k,'rb').read()).hexdigest()==v['sha256'] for k,v in m.items()))"
```

## Redraw the figures
```bash
python3 scripts/make_figures.py figures figures_out
```
This writes `figure1_evergreen.svg`, `figure2_boundary.svg` and `figure3_swaps.svg`, identical to `figures/`.
Figures 1 and 3 hold their values in the script; Figure 2 reads `figures/fig5_data.json`.

## Scripts
| Script | What it does | Inputs |
|---|---|---|
| `pl_slice_check.py` | Recomputes all tables and estimates | public slice |
| `make_figures.py` | Figures 1–3 | values in the script, `figures/fig5_data.json` |
| `pl_verify_redaction.py`, `pl_redact_slice.py`, `pl_public_slice.py` | Build the public slice; replace licence-sensitive patch text by sha256; verify that no such text remains | internal slice |
| `pl_r3.py` | The instruments: forced commits (prefill and instruct modes, cross-model committers), replay of logged edits, grading driver | logged prompts, harness, vLLM |
| `pl_r3_paired.py`, `pl_r8.py`, `pl_selcheck.py`, `pl_e1_list.py` | Paired analyses, committer swaps, selection check, task-matched pair list | instrument outputs |
| `pl_e1_meta.py`, `pl_m8.py`, `pl_release_slice.py`, `pl_fig5data.py` | Build the slice files and the figure data from instrument outputs | instrument outputs |
| `pl_probe_perm.py` | Cross-fitted probes, permutation tests and out-of-fold scores | probe features (not public) |
| `pl_pubtext.py` | Word count of a writeup draft | — |

Scripts that read instrument outputs take the data root from the environment variable `PL_WORK`.

## Licences
- **Code:** Apache-2.0 (`LICENSE`, `NOTICE`).
- **Figures and expected output:** CC-BY-4.0.
- **The public slice** (separate Kaggle Dataset): CC-BY-4.0 for our annotations. It contains AI-generated patches,
  labelled with the model that wrote them. Patch excerpts of the task repositories remain under their licences;
  licence-sensitive excerpts are redacted (see the dataset README).

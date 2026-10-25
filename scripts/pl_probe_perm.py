# SPDX-License-Identifier: Apache-2.0
"""paper-lab P1 permutation test (CPU, numpy/scipy; runs locally on extracted features). Read-only.

Question: is there ANY within-task signal in the hidden-state probes at calls 6/10/14 once multiplicity over the 16 cells
(4 layers x 2 poolings x 2 probe types) is accounted for? Null: outcomes exchangeable among the runs of a task at a fork
(task difficulty and per-task counts preserved; only within-task information destroyed). For each permutation the
probes are refitted (PCA is label-free and fitted once per training fold); statistics = max and mean over cells of the
within-task AUC on held-out boundary tasks. Same probe recipe as pl_probe_analyze.py (standardize -> PCA -> L2 logistic,
5 task-grouped folds; all-run probe: PCA 128, lambda 1e-2; task-demeaned probe on boundary tasks: PCA 32, lambda 1e-1).

  python -I pl_probe_perm.py <dir with f2_31b_l*_part*.npz> <boundary.json> [n_perm]
"""
import collections, glob, json, os, random, statistics as st, sys
import numpy as np
from scipy.optimize import minimize

D, BJ = sys.argv[1], sys.argv[2]
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 100
B = json.load(open(BJ))
PS = {**B["train"]["ps"], **B["dev"]["ps"]}


def load():
    meta, last, mean = [], [], []
    for f in sorted(glob.glob(f"{D}/f2_31b_l*_part*.npz")):
        z = np.load(f); last.append(z["last"]); mean.append(z["mean"]); layers = list(z["layers"])
        meta += [json.loads(l) for l in open(f[:-4] + ".jsonl")]
    return meta, np.concatenate(last), np.concatenate(mean), layers


def folds(tasks, k=5, seed=0):
    ts = sorted(set(tasks)); rnd = random.Random(seed); rnd.shuffle(ts)
    return {t: i % k for i, t in enumerate(ts)}


def project(Xtr, Xte, dim):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    d = min(dim, Xtr.shape[0] - 1, Xtr.shape[1])
    _, _, Vt = np.linalg.svd(Xtr, full_matrices=False)
    V = Vt[:d].T
    Ztr, Zte = Xtr @ V, Xte @ V
    zs = Ztr.std(0) + 1e-4
    return Ztr / zs, Zte / zs


def logit_fit(Z, y, lam):
    def f(wb):
        w, b = wb[:-1], wb[-1]; s = Z @ w + b
        p = 1 / (1 + np.exp(-s))
        loss = np.mean(np.logaddexp(0, s) - y * s) + lam * (w @ w)
        g = Z.T @ (p - y) / len(y) + 2 * lam * w
        return loss, np.append(g, np.mean(p - y))
    r = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return r.x[:-1], r.x[-1]


def within_auc(scores, ys, tasks):
    per = collections.defaultdict(list)
    for s, y, t in zip(scores, ys, tasks):
        per[t].append((s, y))
    vals = []
    for t, v in per.items():
        if not (0 < PS.get(t, 0) < 1):
            continue
        pos = [s for s, y in v if y]; neg = [s for s, y in v if not y]
        if pos and neg:
            vals.append(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)))
    return st.mean(vals) if vals else float("nan")


SC = {"rows": {}}


def main():
    meta, last, mean, layers = load()
    y_all = np.array([m["class"] == "res" for m in meta], dtype=float)
    task_all = [m["task"] for m in meta]; fork_all = [m["fork"] for m in meta]
    F = folds(task_all)
    print(f"loaded {len(meta)} contexts; layers {layers}; permutations {NPERM}")
    for fk in ("6", "10", "14", "own", "final"):
        idx = [i for i, f in enumerate(fork_all) if f == fk]
        tt = [task_all[i] for i in idx]; y0 = y_all[idx]
        bmask = np.array([0 < PS.get(t, 0) < 1 for t in tt])
        groups = collections.defaultdict(list)
        for j, t in enumerate(tt):
            groups[t].append(j)
        # precompute label-free projections per cell and fold
        cells = []
        for li in range(len(layers)):
            for pool, X in (("last", last), ("mean64", mean)):
                Xi = X[idx, li].astype(np.float32)
                Xdm = Xi.copy()
                for t, js in groups.items():
                    Xdm[js] -= Xi[js].mean(0)
                proj_all, proj_dm = [], []
                for f in range(5):
                    tr = np.array([j for j, t in enumerate(tt) if F[t] != f]); te = np.array([j for j, t in enumerate(tt) if F[t] == f])
                    if len(te) == 0:
                        continue
                    proj_all.append((tr, te, *project(Xi[tr], Xi[te], 128)))
                    trb = tr[bmask[tr]]; teb = te[bmask[te]]
                    if len(teb) and len(trb) > 10:
                        proj_dm.append((trb, teb, *project(Xdm[trb], Xdm[teb], 32)))
                cells.append((f"L{layers[li]}-{pool}-all", proj_all, 1e-2))
                cells.append((f"L{layers[li]}-{pool}-demeaned", proj_dm, 1e-1))

        def stats(y):
            out = []
            for name, projs, lam in cells:
                sc = np.full(len(tt), np.nan)
                for tr, te, Ztr, Zte in projs:
                    if len(set(y[tr])) < 2:
                        continue
                    w, b = logit_fit(Ztr, y[tr], lam); sc[te] = Zte @ w + b
                ok = ~np.isnan(sc)
                out.append(within_auc(sc[ok], y[ok], [tt[j] for j in np.where(ok)[0]]))
            return np.array(out)
        obs = stats(y0)
        if os.environ.get("PL_PROBE_SCORES"):  # v0.8: export out-of-fold scores of every cell (observed labels)
            SC.setdefault("cells", [n for n, _, _ in cells])
            for name, projs, lam in cells:
                for tr, te, Ztr, Zte in projs:
                    if len(set(y0[tr])) < 2:
                        continue
                    w, b = logit_fit(Ztr, y0[tr], lam)
                    for j, sv in zip(te, Zte @ w + b):
                        SC["rows"].setdefault(idx[j], {})[name] = round(float(sv), 5)
        rng = np.random.default_rng(0); pmax, pmean = [], []
        for _ in range(NPERM):
            yp = y0.copy()
            for t, js in groups.items():
                yp[js] = rng.permutation(y0[js])
            s = stats(yp); pmax.append(np.nanmax(s)); pmean.append(np.nanmean(s))
        pm = (1 + sum(v >= np.nanmax(obs) for v in pmax)) / (1 + NPERM)
        pa = (1 + sum(v >= np.nanmean(obs) for v in pmean)) / (1 + NPERM)
        print(f"== fork {fk}: {len(idx)} contexts; cells: " + ", ".join(f"{n} {v:.3f}" for (n, _, _), v in zip(cells, obs)))
        print(f"   max over 16 cells {np.nanmax(obs):.3f} (perm p {pm:.3f}; null 95th pct {np.percentile(pmax, 95):.3f}); "
              f"mean over cells {np.nanmean(obs):.3f} (perm p {pa:.3f}; null mean {np.mean(pmean):.3f}, 95th pct {np.percentile(pmean, 95):.3f})", flush=True)


if __name__ == "__main__":
    main()
    out = os.environ.get("PL_PROBE_SCORES")
    if out:  # one row per context: identifiers, label and the 16 out-of-fold scores (no features, no text)
        import gzip
        meta = load()[0]
        with gzip.open(out, "wt") as f:
            for i in sorted(SC["rows"]):
                m = meta[i]
                f.write(json.dumps({"ctx_index": i, "run": m["run"], "tag": m["tag"], "task": m["task"], "fork": m["fork"],
                                    "call": m["call"], "class": m["class"], "scores": SC["rows"][i]}) + "\n")
        print("wrote", len(SC["rows"]), "context rows to", out)

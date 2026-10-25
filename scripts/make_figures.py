# SPDX-License-Identifier: Apache-2.0
"""paper-lab: static SVG figures for the decision memo / writeup (stdlib only; aggregates, no task content).
Usage: python make_figures.py <fig2.json> <out_dir>
Palette: dataviz reference categorical slots 1-3 (validated light-mode; contrast WARN -> every mark is direct-labelled).
"""
import json, sys, os

SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e6e5e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
FONT = "font-family='Inter, Helvetica, Arial, sans-serif'"


def svg(w, h, body, title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT}>"
            f"<title>{title}</title><rect width='{w}' height='{h}' fill='{SURF}'/>{body}</svg>")


def text(x, y, s, size=12, color=INK, anchor="start", weight="normal"):
    return f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' text-anchor='{anchor}' font-weight='{weight}'>{s}</text>"


def fig1():
    """Ever-green audit, four model/pool rows, three panels (single series each)."""
    rows = ["Gemma 4 31B (train)", "Gemma 4 31B (dev)", "DeepSeek V4 Pro (dev)", "Qwen3.6-27B (dev)"]
    panels = [(("Successful runs already passing", "at their first applied edit"), [67, 60, 69, 88],
               ["76/113", "26/43", "24/35", "15/17"]),
              (("Wrong-patch runs that ever passed", "at any point mid-run"), [0.6, 2.6, 0, 0], ["1/179", "2/76", "0/33", "0/16"]),
              (("Share of a successful run's calls", "after its patch already passes"), [40, 48, 42, 45],
               ["", "", "", ""])]
    W, left, pw, gap, top, rh = 1200, 170, 240, 90, 92, 34
    H = top + rh * len(rows) + 60
    b = [text(16, 26, "Wrong-patch runs almost never pass mid-run; most successes pass at their first edit (three model families)", 15, INK, weight="bold"),
         text(16, 46, "Ever-green audit: every intermediate working tree re-applied and graded against the hidden tests", 12, INK2)]
    for i, r in enumerate(rows):
        b.append(text(left - 10, top + rh * i + rh / 2 + 4, r, 12, INK, "end"))
    for p, (ttl, vals, notes) in enumerate(panels):
        x0 = left + p * (pw + gap)
        b.append(text(x0, top - 26, ttl[0], 11, INK2)); b.append(text(x0, top - 12, ttl[1], 11, INK2))
        for t in (0, 50, 100):
            gx = x0 + pw * t / 100
            b.append(f"<line x1='{gx:.1f}' y1='{top}' x2='{gx:.1f}' y2='{top + rh * len(rows)}' stroke='{GRID}' stroke-width='1'/>")
            b.append(text(gx, top + rh * len(rows) + 16, f"{t}%", 10, INK2, "middle"))
        for i, v in enumerate(vals):
            y = top + rh * i + 9
            w = max(pw * v / 100, 1.5) if v > 0 else 0.0  # a zero share draws no bar
            b.append(f"<rect x='{x0}' y='{y}' width='{w:.1f}' height='{rh - 18}' rx='2' fill='{S1}'><title>{rows[i]}: {v}% ({notes[i]})</title></rect>")
            b.append(text(x0 + w + 6, y + rh - 22, f"{v:g}%", 11, INK, weight="bold"))
            b.append(text(x0 + w + 6 + 34, y + rh - 22, notes[i], 10, INK2))
    b.append(text(16, H - 14, "Runs whose logged patch is reproduced by replay (551 of 704; in 82 of the others, edits came after the 28-call budget, which the harness refuses). Third panel: median per successful run.", 10, INK2))
    return svg(W, H, "".join(b), "Ever-green audit across model families")




def fig4():
    """R8 explorer x committer swap at the explorer's own commit point. Colour = committer model (fixed across panels)."""
    groups = [("Contexts explored by Gemma 4 31B", [("resolved", 0.81, 0.89, 25), ("hesitating (no patch)", 0.14, 0.14, 26), ("wrong patch", 0.00, 0.00, 29)]),
              ("Contexts explored by Qwen3.6-27B", [("resolved", 0.72, 0.92, 12), ("hesitating (no patch)", 0.67, 0.44, 3), ("wrong patch", 0.00, 0.00, 12)])]
    W, H, top, ph, left, pw, gap = 1200, 450, 104, 250, 70, 470, 120
    b = [text(16, 26, "Swapping the committer: no commit from a sampled wrong-patch context passed; on successful ones only Qwen on Gemma's (+0.08) is significant", 15, INK, weight="bold"),
         text(16, 46, "Forced commit at the explorer's own commit point; same logged context (incl. the explorer's reasoning), committed by each model; runs in brackets", 12, INK2),
         f"<rect x='16' y='60' width='10' height='10' rx='2' fill='{S1}'/>" + text(30, 69, "Gemma 4 31B commits", 11, INK2),
         f"<rect x='170' y='60' width='10' height='10' rx='2' fill='{S2}'/>" + text(184, 69, "Qwen3.6-27B commits", 11, INK2)]
    for g, (title, rows) in enumerate(groups):
        x0 = left + g * (pw + gap)
        b.append(text(x0, top - 10, title, 12, INK, weight="bold"))
        for t in (0, 0.5, 1.0):
            gy = top + ph * (1 - t)
            b.append(f"<line x1='{x0}' y1='{gy:.1f}' x2='{x0 + pw}' y2='{gy:.1f}' stroke='{GRID}' stroke-width='1'/>")
            b.append(text(x0 - 8, gy + 4, f"{t:.1f}", 10, INK2, "end"))
        bw, cw = 46, pw / len(rows)
        for i, (lab, vg, vq, n) in enumerate(rows):
            cx = x0 + cw * i + cw / 2
            for j, (v, col, who) in enumerate(((vg, S1, "Gemma commits"), (vq, S2, "Qwen commits"))):
                bx = cx - bw - 1 + j * (bw + 2); h = max(ph * v, 1.5)
                b.append(f"<rect x='{bx:.1f}' y='{top + ph - h:.1f}' width='{bw}' height='{h:.1f}' rx='2' fill='{col}'><title>{lab}, {who}: {v:.2f} ({n} runs)</title></rect>")
                b.append(text(bx + bw / 2, top + ph - h - 6, f"{v:.2f}", 10, INK, "middle"))
            b.append(text(cx, top + ph + 18, f"{lab} [{n}]", 11, INK2, "middle"))
    b.append(text(16, H - 30, "Successful contexts: +0.08 [+0.01, +0.17] (Qwen on Gemma's); −0.19 [−0.39, +0.03] (Gemma on Qwen's, not significant; 19 % of Gemma's edits there do not apply, Qwen's 0 %).", 10, INK2))
    b.append(text(16, H - 14, "Contexts are selected on the explorer's success, which favours its own committer. Task-clustered bootstrap; the Qwen-explored hesitating cell rests on 3 runs.", 10, INK2))
    return svg(W, H, "".join(b), "Explorer x committer swap")



def fig5(d5):
    """E1: same-task resolved vs failed run, Q_c by fork with 95 % task-bootstrap CIs (each fork its own task subset, so no
    connecting lines); right: the variance split as an interval (the context share is bounded below, not identified)."""
    xs = ["6", "10", "14", "own"]; xl = ["call 6", "call 10", "call 14", "own commit point"]
    W, H, top, ph, left, pw = 1180, 440, 104, 240, 70, 560
    b = [text(16, 26, "Within a task, resolved and failed runs are equivalent at call 6 and differ clearly at the own commit point", 15, INK, weight="bold"),
         text(16, 46, "Boundary tasks, one resolved and one failed Gemma 4 31B run each; Q_c = forced-commit pass rate; bars: 95 % bootstrap over tasks; tasks per fork below", 12, INK2),
         f"<circle cx='21' cy='65' r='4.5' fill='{S1}'/>" + text(30, 69, "resolved run", 11, INK2),
         f"<circle cx='131' cy='65' r='4.5' fill='{S2}'/>" + text(140, 69, "failed run, same task", 11, INK2)]
    X = lambda i: left + 50 + i * (pw - 100) / 3 + (24 if i == 3 else 0)
    Y = lambda v: top + ph * (1 - v)
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(f"<line x1='{left}' y1='{Y(t):.1f}' x2='{left + pw}' y2='{Y(t):.1f}' stroke='{GRID}' stroke-width='1'/>")
        b.append(text(left - 8, Y(t) + 4, f"{t:.2f}", 10, INK2, "end"))
    sep = (X(2) + X(3)) / 2
    b.append(f"<line x1='{sep:.1f}' y1='{top - 6}' x2='{sep:.1f}' y2='{top + ph}' stroke='{INK2}' stroke-width='1' stroke-dasharray='3,4'/>")
    for i, l in enumerate(xl):
        b.append(text(X(i), top + ph + 18, l, 11, INK2, "middle"))
        b.append(text(X(i), top + ph + 32, f"{d5[xs[i]]['n']} tasks" + (", calls 1–30" if xs[i] == "own" else ""), 10, INK2, "middle"))
    for key, lab, col, dx in (("resolved", "resolved run", S1, -9), ("failed", "failed run, same task", S2, 9)):
        for i, x in enumerate(xs):
            v = d5[x][key]; cx = X(i) + dx
            b.append(f"<line x1='{cx:.1f}' y1='{Y(v[2]):.1f}' x2='{cx:.1f}' y2='{Y(v[1]):.1f}' stroke='{col}' stroke-width='2'/>")
            b.append(f"<circle cx='{cx:.1f}' cy='{Y(v[0]):.1f}' r='4.5' fill='{col}' stroke='{SURF}' stroke-width='2'><title>{lab}, {xl[i]}: {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]</title></circle>")
        v = d5["own"][key]
        b.append(text(X(3) + 22, Y(v[0]) + 4, f"{v[0]:.2f}", 11, INK, weight="bold"))
    # right panel: where outcome variance lives, drawn as shares with intervals (not a partition)
    px, bw = left + pw + 110, 300
    b.append(text(px, top - 22, "Where outcome variance lives", 12, INK, weight="bold"))
    rows = [("Between tasks (share of all outcome variance)", 0.79, 0.83, 0.81, "79–83 % (train, dev)"),
            ("Predictable from the context at the first edit (within-task share)", 0.09, 1.0, 0.22, "lower bound: estimate 0.22; one-sided 95 % limit 0.09")]
    for r, (lab, lo, hi, pt, note) in enumerate(rows):
        y0 = top + 20 + r * 110
        b.append(text(px, y0, lab, 11, INK2))
        ay = y0 + 30
        b.append(f"<line x1='{px}' y1='{ay}' x2='{px + bw}' y2='{ay}' stroke='{GRID}' stroke-width='6' stroke-linecap='round'/>")
        for t in (0, 0.5, 1.0):
            b.append(text(px + bw * t, ay + 20, f"{t:g}", 10, INK2, "middle"))
        b.append(f"<line x1='{px + bw * lo:.1f}' y1='{ay}' x2='{px + bw * hi:.1f}' y2='{ay}' stroke='{INK2}' stroke-width='6' stroke-linecap='round'"
                 + (" stroke-opacity='0.35'" if r == 1 else "") + "/>")
        b.append(f"<circle cx='{px + bw * pt:.1f}' cy='{ay}' r='5' fill='{INK}' stroke='{SURF}' stroke-width='2'><title>{lab}: {note}</title></circle>")
        b.append(text(px, ay + 38, note, 11, INK))
    b.append(text(px, top + 20 + 110 + 82, "A lower bound on predictability, not a causal share;", 10, INK2))
    b.append(text(px, top + 20 + 110 + 96, "the upper end is not identified.", 10, INK2))
    b.append(text(16, H - 30, "Equivalent at call 6 (±0.15); no difference detected at calls 10–14 (intervals reach +0.3); +0.47 [+0.28, +0.64] at the own commit point, which lies anywhere", 10, INK2))
    b.append(text(16, H - 14, "between calls 1 and 30 and is plotted apart. Predictable share: Cauchy–Schwarz lower bound, estimate 0.22 (0.21 bias-corrected), one-sided 95 % limit 0.09.", 10, INK2))
    return svg(W, H, "".join(b), "Within-task divergence")


def main():
    # argv[1]: the figure-data folder (fig5_data.json; a path to a file inside it also works); argv[2]: output folder
    ddir = sys.argv[1] if os.path.isdir(sys.argv[1]) else os.path.dirname(sys.argv[1]); out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    # paper figures: figure1/2/3 (v0.8 numbering, review m6); INTERNAL_* are not in the paper
    figs = [("figure1_evergreen.svg", fig1()), ("figure3_swaps.svg", fig4()),
            ("figure2_boundary.svg", fig5(json.load(open(os.path.join(ddir, "fig5_data.json")))))]
    for name, content in figs:
        open(os.path.join(out, name), "w").write(content)
        print("wrote", name)


if __name__ == "__main__":
    main()

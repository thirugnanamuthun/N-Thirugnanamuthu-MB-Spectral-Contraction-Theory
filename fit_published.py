#!/usr/bin/env python3
"""
fit_published.py -- test of the MB-Spectral Contraction Theory on PUBLISHED accuracy-vs-compute data.

Data (copied verbatim from the cited tables; no value was read off a plot):
  NIE : W. Nie et al., "The Coupling Tax ...", arXiv:2605.07686 (8 May 2026), Tabs. 4, 10, 11, 18, 21
  WANG: X. Wang et al., "Self-consistency improves chain of thought ...", ICLR 2023, Tab. 6

Pre-declared protocol (fixed before any fit was run):
  P1  accuracy is non-decreasing in compute                       (Theorem 1: error strictly decreasing)
  P2  diminishing returns: accuracy concave in compute            (Theorem 4b: error strictly convex)
      violation z-score of consecutive slope differences, flagged if z > 2 (independent-SE, conservative)
  P3  model comparison on curves with >= 4 points, 3 parameters each, leave-one-out RMSE:
        SPEC  a = a_inf - A exp(-g t)          (K = 1 spectral contraction; error -> 0 geometrically)
        POW   a = a_inf - c t^(-b)             (power-law scaling form)
        LOG   a = a_inf / (1 + (t50/t)^s)      (logistic in ln t; allows a threshold)
Fits use an exhaustive grid over the nonlinear parameters and bounded linear least squares for the rest.
"""
import numpy as np
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

CURVES = [
    dict(id="C1", name="Qwen3-8B no-think MATH-500", src="NIE", n=500, unit="tok",
         t=[128, 256, 512, 1024, 2048], a=[4.6, 16.6, 40.6, 59.8, 64.4]),
    dict(id="C2", name="Qwen3-8B think MATH-500", src="NIE", n=500, unit="tok",
         t=[256, 512, 1024, 2048], a=[4.2, 6.2, 18.0, 44.0]),
    dict(id="C3", name="Qwen3-8B no-think GSM8K", src="NIE", n=200, unit="tok",
         t=[32, 64, 128, 256, 512], a=[3.0, 12.0, 54.5, 89.0, 94.0]),
    dict(id="C4", name="Qwen3-8B think GSM8K", src="NIE", n=200, unit="tok",
         t=[128, 256, 512], a=[2.0, 22.0, 66.5]),
    dict(id="C5", name="R1-Distill-Llama-8B MATH-500", src="NIE", n=500, unit="tok",
         t=[1024, 2048, 4096], a=[28.4, 40.2, 49.0]),
    dict(id="C6", name="R1-Distill-Llama-8B GSM8K", src="NIE", n=1319, unit="tok",
         t=[256, 512, 1024], a=[19.9, 59.6, 63.8]),
    dict(id="C7", name="UL2-20B self-consist. AQuA", src="WANG", unit="path",
         t=[1, 5, 10, 20, 40], a=[19.7, 24.9, 25.3, 26.7, 26.9], sd=[2.5, 2.6, 1.8, 1.0, 0.5]),
    dict(id="C8", name="UL2-20B self-consist. MultiArith", src="WANG", unit="path",
         t=[1, 5, 10, 20, 40], a=[9.5, 11.3, 12.3, 13.7, 14.7], sd=[1.2, 1.2, 0.8, 0.9, 0.3]),
]
for c in CURVES:
    c["t"], c["a"] = np.array(c["t"], float), np.array(c["a"], float)
    if "sd" in c:
        c["se"] = np.array(c["sd"], float)                       # reported std over 10 runs (conservative)
    else:
        p = c["a"] / 100
        c["se"] = 100 * np.sqrt(p * (1 - p) / c["n"])            # binomial standard error, in points
    c["scale"] = 1000.0 if c["unit"] == "tok" else 1.0           # report slopes per 1k tokens / per path


# ---------------------------------------------------------------- model fitting
def _bounded(f, y, cap=100.0):
    """y ~ a_inf - A f, 0<=a_inf<=cap, A>=0"""
    X = np.column_stack([np.ones_like(f), -f])
    r = lsq_linear(X, y, bounds=([0.0, 0.0], [cap, np.inf]))
    return r.x, float(np.sum((X @ r.x - y) ** 2))


def fit_spec(t, y):
    best = None
    for g in np.logspace(np.log10(0.01 / t.max()), np.log10(100 / t.min()), 600):
        (ai, A), sse = _bounded(np.exp(-g * t), y)
        if best is None or sse < best[0]:
            best = (sse, ai, A, g)
    sse, ai, A, g = best
    return dict(sse=sse, par=(ai, g), pred=lambda x: ai - A * np.exp(-g * np.asarray(x, float)))


def fit_pow(t, y):
    best = None
    for b in np.linspace(0.02, 4.0, 400):
        (ai, c), sse = _bounded(t ** (-b), y)
        if best is None or sse < best[0]:
            best = (sse, ai, c, b)
    sse, ai, c, b = best
    return dict(sse=sse, par=(ai, b), pred=lambda x: ai - c * np.asarray(x, float) ** (-b))


def fit_log(t, y):
    best = None
    for t50 in np.logspace(np.log10(t.min() / 4), np.log10(t.max() * 4), 120):
        for s in np.logspace(np.log10(0.3), np.log10(12), 80):
            h = 1 / (1 + (t50 / t) ** s)
            ai = float(np.clip(h @ y / (h @ h), 0, 100))
            sse = float(np.sum((ai * h - y) ** 2))
            if best is None or sse < best[0]:
                best = (sse, ai, t50, s)
    sse, ai, t50, s = best
    return dict(sse=sse, par=(ai, t50, s), pred=lambda x: ai / (1 + (t50 / np.asarray(x, float)) ** s))


FITTERS = dict(SPEC=fit_spec, POW=fit_pow, LOG=fit_log)


def rmse_in(c, fit):
    return float(np.sqrt(np.mean((fit["pred"](c["t"]) - c["a"]) ** 2)))


def rmse_loo(c, fitter):
    e = []
    for i in range(len(c["t"])):
        m = np.arange(len(c["t"])) != i
        e.append(fitter(c["t"][m], c["a"][m])["pred"](c["t"][i:i + 1])[0] - c["a"][i])
    return float(np.sqrt(np.mean(np.square(e))))


def err_last(c, fitter):
    return float(abs(fitter(c["t"][:-1], c["a"][:-1])["pred"](c["t"][-1:])[0] - c["a"][-1]))


# ---------------------------------------------------------------- P1, P2
print("MB-Spectral Contraction Theory: test on published accuracy-vs-compute data")
print("sources: NIE = arXiv:2605.07686 (Tabs 4,10,11,18,21); WANG = ICLR 2023 (Tab 6)")
print("curves:", len(CURVES), "   points:", sum(len(c["t"]) for c in CURVES))
print()
print("P1  monotone non-decreasing accuracy (Theorem 1)")
p1 = 0
for c in CURVES:
    ok = bool(np.all(np.diff(c["a"]) >= 0)); p1 += ok
    print(f"  {c['id']} {c['name']:<34s} {'PASS' if ok else 'FAIL'}")
print(f"  -> {p1}/{len(CURVES)} curves monotone")
print()
print("P2  diminishing returns: accuracy concave (Theorem 4b); FAIL if max z > 2")
print("    slopes in points per 1k tokens (tok) or per path; z = slope rise / s.e.")
p2 = 0
for c in CURVES:
    s = np.diff(c["a"]) / np.diff(c["t"]) * c["scale"]
    v = (c["se"][:-1] ** 2 + c["se"][1:] ** 2) / np.diff(c["t"]) ** 2 * c["scale"] ** 2
    z = (s[1:] - s[:-1]) / np.sqrt(v[1:] + v[:-1])
    ok = bool(z.max() <= 2.0); p2 += ok
    print(f"  {c['id']} slopes " + " ".join(f"{x:7.2f}" for x in s) + f"   max z = {z.max():6.2f}  {'PASS' if ok else 'FAIL'}")
print(f"  -> {p2}/{len(CURVES)} curves concave within noise")
print()

# ---------------------------------------------------------------- P3
print("P3  model comparison, curves with >= 4 points (RMSE in accuracy points; mean s.e. for scale)")
print("    I = in-sample, L = leave-one-out, X = error predicting the last point from the others")
hdr = "  id  n  mean-se |  I:SPEC   POW   LOG |  L:SPEC   POW   LOG |  X:SPEC   POW   LOG"
print(hdr)
tally = dict(SPEC=0, POW=0, LOG=0)
loo_all = {k: [] for k in FITTERS}
fits_store = {}
for c in CURVES:
    if len(c["t"]) < 4:
        continue
    fits = {k: f(c["t"], c["a"]) for k, f in FITTERS.items()}
    fits_store[c["id"]] = fits
    I = [rmse_in(c, fits[k]) for k in FITTERS]
    L = [rmse_loo(c, FITTERS[k]) for k in FITTERS]
    X = [err_last(c, FITTERS[k]) for k in FITTERS]
    for k, v in zip(FITTERS, L):
        loo_all[k].append(v)
    tally[list(FITTERS)[int(np.argmin(L))]] += 1
    print(f"  {c['id']} {len(c['t'])}  {c['se'].mean():6.2f}  | " + " ".join(f"{x:6.2f}" for x in I)
          + "  | " + " ".join(f"{x:6.2f}" for x in L) + "  | " + " ".join(f"{x:6.2f}" for x in X))
print()
print("  mean leave-one-out RMSE:  " + "   ".join(f"{k} {np.mean(v):.2f}" for k, v in loo_all.items()))
print("  curves where model has the lowest leave-one-out RMSE:  " + "   ".join(f"{k} {v}" for k, v in tally.items()))
print()
print("SPEC parameters (a_inf in %, g in 1/ktok or 1/path; 1/g = e-folding compute)")
for c in CURVES:
    if c["id"] in fits_store:
        ai, g = fits_store[c["id"]]["SPEC"]["par"]
        g_rep = g * c["scale"]
        print(f"  {c['id']}  a_inf = {ai:6.1f}   g = {g_rep:7.3f}   1/g = {1 / g_rep:8.3f} {'ktok' if c['unit'] == 'tok' else 'path'}")

# ---------------------------------------------------------------- figure + csv
COL = dict(SPEC="#2a78d6", POW="#eb6834", LOG="#1baf7a")
SHORT = dict(C1="Qwen3 no-think MATH", C2="Qwen3 think MATH", C3="Qwen3 no-think GSM8K", C4="Qwen3 think GSM8K",
             C5="R1-Distill MATH", C6="R1-Distill GSM8K", C7="UL2 self-cons. AQuA", C8="UL2 self-cons. MultiArith")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 6.8,
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                     "axes.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#898781", "grid.color": "#e1e0d9", "grid.linewidth": .5, "savefig.facecolor": "white"})
fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.25))
for ax, c in zip(axs.ravel(), CURVES):
    ax.errorbar(c["t"], c["a"], yerr=c["se"], fmt="o", color="#0b0b0b", ms=3, lw=.8, capsize=1.5, zorder=5)
    if c["id"] in fits_store:
        tt = np.logspace(np.log10(c["t"].min() / 1.3), np.log10(c["t"].max() * 1.3), 200)
        for k in FITTERS:
            ax.plot(tt, fits_store[c["id"]][k]["pred"](tt), color=COL[k], lw=1.1, label=k)
    else:
        ax.plot(c["t"], c["a"], color="#898781", lw=.8)
    ax.set_xscale("log"); ax.grid(alpha=.7)
    ax.set_title(f"{c['id']}  {SHORT[c['id']]}", fontsize=6.2)
    tk = c["t"][::2] if len(c["t"]) > 3 else c["t"][[0, -1]]
    ax.set_xticks(tk); ax.set_xticklabels([f"{x:g}" for x in tk]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(bottom=max(-4, min(ax.get_ylim()[0], c["a"].min() - 3)), top=min(104, ax.get_ylim()[1]))
    ax.set_xlabel("tokens" if c["unit"] == "tok" else "paths"); ax.set_ylabel("accuracy (%)")
    ax.tick_params(length=2, width=.5)
axs[0, 0].legend(frameon=False, loc="lower right")
fig.tight_layout(w_pad=.8, h_pad=1.0)
fig.savefig("fig_published.pdf")
with open("published_data.csv", "w") as f:
    f.write("curve,name,source,compute,unit,accuracy_pct,se_pts\n")
    for c in CURVES:
        for t, a, s in zip(c["t"], c["a"], c["se"]):
            f.write(f"{c['id']},{c['name']},{c['src']},{t:g},{c['unit']},{a},{s:.3f}\n")
print()
print("saved: fig_published.pdf, published_data.csv")

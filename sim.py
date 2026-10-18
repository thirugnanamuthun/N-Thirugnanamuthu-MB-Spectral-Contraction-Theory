"""
MB-Spectral Contraction Theory of AI Computation -- numerical verification + figures.
Everything printed / plotted in the paper comes from this script (numbers.tex is auto-generated).
Model:  E(tau) = exp(-alpha G tau) E(0),  G = I - M symmetric positive definite,  tau = d*T
        Euler (first-order) version:  E(t+1) = (I - alpha T G) E(t)
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

os.chdir("/home/claude/spectral")
rng = np.random.default_rng(2026)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SEC, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
OKGREEN, CRIT = "#006300", "#d03b3b"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 5.8,
    "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": SEC, "xtick.color": SEC,
    "ytick.color": SEC, "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.2, "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white"})
W1, W2 = 3.4, 7.0

NUM = {}


def put(name, text):
    NUM[name] = text


def sci(x):
    """LaTeX scientific notation"""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.1f}\\times10^{{{e}}}"


# ------------------------------------------------------------------ core functions
def err_sg(tau, g, c, alpha):
    tau = np.atleast_1d(np.asarray(tau, float))[:, None]
    return np.sqrt(((c ** 2)[None, :] * np.exp(-2 * alpha * g[None, :] * tau)).sum(1))


def d_err(tau, g, c, alpha):
    tau = np.atleast_1d(np.asarray(tau, float))[:, None]
    e = np.sqrt(((c ** 2)[None, :] * np.exp(-2 * alpha * g[None, :] * tau)).sum(1))
    num = alpha * ((g * c ** 2)[None, :] * np.exp(-2 * alpha * g[None, :] * tau)).sum(1)
    return -num / e


def err_euler(T, t, g, c, alpha):
    T = np.atleast_1d(np.asarray(T, float))[:, None]
    return np.sqrt(((c ** 2)[None, :] * np.abs(1 - alpha * T * g[None, :]) ** (2 * t)).sum(1))


def tau_star(beta, g, c, alpha):
    if -d_err(0.0, g, c, alpha)[0] <= beta:
        return 0.0
    lo, hi = 0.0, 1.0
    while -d_err(hi, g, c, alpha)[0] > beta:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if -d_err(mid, g, c, alpha)[0] > beta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rho_euler(T, gmin, gmax, alpha):
    return np.maximum(np.abs(1 - alpha * T * gmin), np.abs(1 - alpha * T * gmax))


# ------------------------------------------------------------------ main instance
alpha = 1.0
mu = np.array([0.90, 0.75, 0.55, 0.35, 0.15, -0.05, -0.25, -0.40])
g = 1 - mu
n = len(mu)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
M = Q @ np.diag(mu) @ Q.T
G = np.eye(n) - M
assert np.allclose(np.linalg.eigvalsh(G), np.sort(g))
g = np.sort(g)                                   # ascending: g[0]=gmin
gmin, gmax = g[0], g[-1]
kappa = gmax / gmin
Tstab, Tmono, Tstar = 2 / (alpha * gmax), 1 / (alpha * gmax), 2 / (alpha * (gmin + gmax))
rho_star = (kappa - 1) / (kappa + 1)
c0 = np.ones(n) / np.sqrt(n)                     # modal coordinates of E0, ||E0|| = 1
# direct (matrix) check of the modal formula
w, V = np.linalg.eigh(G)
E0 = V @ c0
for tau in (0.3, 2.0, 7.0):
    S = V @ np.diag(np.exp(-alpha * w * tau)) @ V.T
    assert abs(np.linalg.norm(S @ E0) - err_sg(tau, g, c0, alpha)[0]) < 1e-12
Ae = np.eye(n) - alpha * 1.0 * G
for t in (1, 5, 12):
    assert abs(np.linalg.norm(np.linalg.matrix_power(Ae, t) @ E0) - err_euler(1.0, t, g, c0, alpha)[0]) < 1e-12
print(f"main instance: gmin={gmin:.3f} gmax={gmax:.3f} kappa={kappa:.2f} Tmono={Tmono:.4f} "
      f"Tstab={Tstab:.4f} T*={Tstar:.4f} rho*={rho_star:.4f}")
put("Gmin", f"{gmin:.2f}"); put("Gmax", f"{gmax:.2f}"); put("Kappa", f"{kappa:.0f}")
put("Tstab", f"{Tstab:.3f}"); put("Tmono", f"{Tmono:.3f}"); put("Tstar", f"{Tstar:.3f}")
put("RhoStar", f"{rho_star:.4f}")

# ------------------------------------------------------------------ Monte-Carlo verification
NMC = 5000
F = dict(sand=0, mono=0, conv=0, thr=0, grow=0, approx=0, opt=0)
opt_err_max = 0.0
kap_pts, rho_pts = [], []
for i in range(NMC):
    nn = int(rng.integers(3, 13))
    mu_i = np.sort(rng.uniform(-0.9, 0.95, nn))[::-1]
    gi = np.sort(1 - mu_i)
    al = rng.uniform(0.2, 2.0)
    ci = rng.standard_normal(nn)
    gl, gh = gi[0], gi[-1]
    nrm = np.linalg.norm(ci)
    tg = np.linspace(0, 20 / (al * gl), 400)
    e = err_sg(tg, gi, ci, al)
    up, lo = nrm * np.exp(-al * gl * tg), nrm * np.exp(-al * gh * tg)
    if not (np.all(e <= up * (1 + 1e-9)) and np.all(e >= lo * (1 - 1e-9))):
        F["sand"] += 1
    if not np.all(np.diff(e) < 0):
        F["mono"] += 1
    if np.any(np.diff(e, 2) < -1e-12):
        F["conv"] += 1
    Ts = 2 / (al * gh)
    if not (rho_euler(0.99 * Ts, gl, gh, al) < 1 and rho_euler(1.01 * Ts, gl, gh, al) > 1):
        F["thr"] += 1
    if not (err_euler(1.01 * Ts, 2000, gi, ci, al)[0] > nrm):
        F["grow"] += 1
    x = al * rng.uniform(0, 3 / (al * gh), 10) * gi[rng.integers(0, nn, 10)]
    if np.any(np.abs((1 - x) - np.exp(-x)) > x ** 2 / 2 + 1e-14):
        F["approx"] += 1
    ts = 2 / (al * (gl + gh)); rs = (gh / gl - 1) / (gh / gl + 1)
    ok = abs(rho_euler(ts, gl, gh, al) - rs) < 1e-12
    for dlt in (1e-3, 1e-2, 1e-1):
        for sgn in (-1, 1):
            ok &= rho_euler(ts * (1 + sgn * dlt), gl, gh, al) > rs - 1e-15
    if not ok:
        F["opt"] += 1
    if i < 400:
        Tg = np.linspace(0, Ts, 2001)
        kap_pts.append(gh / gl); rho_pts.append(rho_euler(Tg, gl, gh, al).min())
print("MC failures:", F)

# Law II : chains of arbitrary (non-normal) contractions
NCH, chain_fail, mono_fail = 1000, 0, 0
for i in range(NCH):
    nn = int(rng.integers(3, 11)); rho = 0.9
    x = rng.standard_normal(nn); x0n = np.linalg.norm(x)
    prev = x0n
    for k in range(40):
        B = rng.standard_normal((nn, nn))
        B *= rng.uniform(0.5, rho) / np.linalg.norm(B, 2)
        x = B @ x
        nx = np.linalg.norm(x)
        if nx >= prev:
            mono_fail += 1
        prev = nx
        if nx > rho ** (k + 1) * x0n * (1 + 1e-12):
            chain_fail += 1
print("chain bound failures", chain_fail, "monotone failures", mono_fail)
Bnn = np.array([[0.5, 5.0], [0.0, 0.5]])
nn_norm = np.linalg.norm(Bnn, 2)
xx = np.array([0.0, 1.0]); peak = 1.0
for _ in range(60):
    xx = Bnn @ xx; peak = max(peak, np.linalg.norm(xx))
print(f"non-normal example: rho={max(abs(np.linalg.eigvals(Bnn))):.2f} ||B||2={nn_norm:.3f} peak={peak:.3f}")

# Law III : equilibrium tau*
NEQ, eq_fail = 500, 0
for i in range(NEQ):
    nn = int(rng.integers(3, 13))
    gi = np.sort(1 - np.sort(rng.uniform(-0.9, 0.95, nn))[::-1]); al = rng.uniform(0.2, 2.0)
    ci = rng.standard_normal(nn)
    neg0 = -d_err(0.0, gi, ci, al)[0]
    interior = i < 400
    beta = rng.uniform(0.001, 0.999) * neg0 if interior else neg0 * rng.uniform(1.0, 3.0)
    ts = tau_star(beta, gi, ci, al)
    top = 3 * ts + 1.0 if interior else 10 / (al * gi[0])
    gridt = np.linspace(0, top, 20001)
    J = err_sg(gridt, gi, ci, al) + beta * gridt
    Js = err_sg(ts, gi, ci, al)[0] + beta * ts
    if Js > J.min() + 1e-9 or (interior and ts <= 0) or ((not interior) and ts != 0):
        eq_fail += 1
# closed form for a single mode
e0, gg, bb = 0.8, 0.37, 0.05
ts_num = tau_star(bb, np.array([gg]), np.array([e0]), 1.0)
ts_cf = np.log(gg * e0 / bb) / gg
print("single-mode equilibrium: numeric", ts_num, "closed form", ts_cf)

# load linearity (wall clock)
nl = 512
Ql, _ = np.linalg.qr(rng.standard_normal((nl, nl)))
Bl = 0.999 * Ql; x0 = rng.standard_normal(nl)
ds = np.arange(100, 2001, 100); tm = []
for _ in range(300):          # warm-up (BLAS threads, caches)
    x0 @ Bl
for d in ds:
    best = 1e9
    for _ in range(15):
        x = x0.copy(); t0 = time.perf_counter()
        for _ in range(d):
            x = Bl @ x
        best = min(best, time.perf_counter() - t0)
    tm.append(best)
tm = np.array(tm)
sl, ic = np.polyfit(ds, tm, 1)
r2 = 1 - ((tm - (sl * ds + ic)) ** 2).sum() / ((tm - tm.mean()) ** 2).sum()
print(f"load: slope={sl*1e6:.2f} us/step intercept={ic*1e3:.3f} ms  R2={r2:.5f}")

# ------------------------------------------------------------------ worked 2x2 example
M2 = np.array([[0.5, 0.2], [0.2, 0.3]]); G2 = np.eye(2) - M2
assert abs(np.trace(G2) - 1.2) < 1e-12 and abs(np.linalg.det(G2) - 0.31) < 1e-12
w2, V2 = np.linalg.eigh(G2)
ga, gb = w2
c2 = V2.T @ np.array([1.0, 0.0])
Ts2, Tst2 = 2 / gb, 2 / (ga + gb); rho2 = (gb - ga) / (gb + ga)
sg1, sg3 = err_sg(1.0, w2, c2, 1.0)[0], err_sg(3.0, w2, c2, 1.0)[0]
eu1, eu3 = err_euler(1.0, 1, w2, c2, 1.0)[0], err_euler(1.0, 3, w2, c2, 1.0)[0]
euB = err_euler(2.6, 10, w2, c2, 1.0)[0]; rhoB = abs(1 - 2.6 * gb)
print(f"2x2: g=({ga:.4f},{gb:.4f}) c^2=({c2[0]**2:.4f},{c2[1]**2:.4f}) Tstab={Ts2:.4f} T*={Tst2:.4f} rho*={rho2:.4f}")
print(f"2x2: sg(1)={sg1:.4f} eu(1,1)={eu1:.4f} sg(3)={sg3:.4f} eu(1,3)={eu3:.4f} eu(2.6,10)={euB:.4f} rhoB={rhoB:.4f}")
for k, v in dict(WgA=f"{ga:.4f}", WgB=f"{gb:.4f}", WcA=f"{c2[0]**2:.4f}", WcB=f"{c2[1]**2:.4f}",
                 WTstab=f"{Ts2:.3f}", WTstar=f"{Tst2:.3f}", WRho=f"{rho2:.4f}", WsgOne=f"{sg1:.4f}",
                 WeuOne=f"{eu1:.4f}", WsgThree=f"{sg3:.4f}", WeuThree=f"{eu3:.4f}", WeuBad=f"{euB:.2f}",
                 WrhoBad=f"{rhoB:.4f}", WexpA=f"{np.exp(-ga):.4f}", WexpB=f"{np.exp(-gb):.4f}",
                 WeulA=f"{1-ga:.4f}", WeulB=f"{1-gb:.4f}").items():
    put(k, v)

# macros
put("NumMC", f"{NMC}"); put("FailSand", str(F["sand"])); put("FailMono", str(F["mono"])); put("FailConv", str(F["conv"]))
put("FailThr", str(F["thr"])); put("FailGrow", str(F["grow"])); put("FailApprox", str(F["approx"])); put("FailOpt", str(F["opt"]))
put("NumChain", str(NCH)); put("FailChain", str(chain_fail)); put("FailChainMono", str(mono_fail))
put("NumEq", str(NEQ)); put("FailEq", str(eq_fail)); put("EqErr", sci(abs(ts_num - ts_cf)) if abs(ts_num - ts_cf) > 0 else "0")
put("LoadRsq", f"{r2:.4f}"); put("LoadSlope", f"{sl*1e6:.1f}"); put("LoadN", str(nl))
put("NNnorm", f"{nn_norm:.2f}"); put("NNpeak", f"{peak:.2f}")
put("EqNum", f"{ts_num:.3f}"); put("EqCf", f"{ts_cf:.3f}")
with open("numbers.tex", "w") as f:
    for k, v in NUM.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))

# ================================================================== FIGURES
def style(ax):
    ax.grid(alpha=.6)
    ax.tick_params(length=2.5, width=.5)


# ---- concept diagram
fig, ax = plt.subplots(figsize=(W1, 1.95)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.9)


def box(x, y, w, h, txt, fc="#cde2fb", ec=BLUE, fs=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)


box(0.05, 3.9, 2.0, 1.7, "Input\n$X(0)$")
box(2.95, 3.9, 4.1, 1.7, "Reasoning chain\n$B_{d-1}\\cdots B_0$\n(depth $d$, time $T$/step)")
box(7.95, 3.9, 2.0, 1.7, "Output\n$X(d)$")
for x0_, x1_ in ((2.1, 2.9), (7.1, 7.9)):
    ax.annotate("", xy=(x1_, 4.75), xytext=(x0_, 4.75), arrowprops=dict(arrowstyle="->", color=SEC, lw=.9))
ax.text(5, 3.2, "error  $E=Y-X$,   $E(\\tau)=e^{-\\alpha G\\tau}E(0)$,   $\\tau=dT$", ha="center", fontsize=6.3, color=SEC)
for i, (ttl, body) in enumerate([("Law I (latency)", "$T\\uparrow\\ \\Rightarrow\\ \\varepsilon\\downarrow$"),
                                  ("Law II (load)", "$d\\uparrow\\ \\Rightarrow\\ C\\uparrow,\\ \\varepsilon\\downarrow$"),
                                  ("Law III (speed)", "$v\\uparrow\\ \\Rightarrow\\ \\varepsilon\\uparrow$")]):
    box(0.05 + i * 3.35, 0.15, 3.2, 2.3, ttl + "\n" + body, fc="#fcfcfb", ec=MUTED, fs=6.2)
fig.savefig("fig_concept.pdf", bbox_inches="tight", pad_inches=.02); plt.close(fig)

# ---- KPI tiles
tot_a = F["sand"] + F["mono"] + F["conv"]
tot_b = F["thr"] + F["grow"] + F["approx"] + F["opt"]
tiles = [(f"{NMC}", "random spectra tested", None),
         (f"{tot_a}", "Thm 1 failures\n(3 properties each)", tot_a == 0),
         (f"{tot_b}", "Euler-model failures\n(4 properties each)", tot_b == 0),
         (f"{chain_fail + mono_fail}/{NCH}", "Law II chain-bound\nfailures (40 steps)", chain_fail + mono_fail == 0),
         (f"{eq_fail}/{NEQ}", "Law III equilibrium\nfailures", eq_fail == 0),
         (f"{r2:.3f}", "$R^2$ wall-clock load\nvs. depth, $n=512$", None)]
fig, ax = plt.subplots(figsize=(W2, 0.95)); ax.axis("off"); ax.set_xlim(0, len(tiles)); ax.set_ylim(0, 1)
for i, (val, lab, ok) in enumerate(tiles):
    ax.add_patch(FancyBboxPatch((i + 0.03, 0.04), 0.94, 0.92, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#fcfcfb", ec=GRID, lw=.9))
    ax.text(i + 0.5, 0.64, val, ha="center", va="center", fontsize=15, color=INK)
    ax.text(i + 0.5, 0.25, lab, ha="center", va="center", fontsize=5.8, color=SEC, linespacing=1.15)
    if ok is not None:
        ax.text(i + 0.95, 0.88, "pass" if ok else "FAIL", ha="right", va="center", fontsize=5.5,
                color=OKGREEN if ok else CRIT)
fig.savefig("fig_tiles.pdf", bbox_inches="tight", pad_inches=.02); plt.close(fig)

# ---- dashboard
fig, axs = plt.subplots(2, 3, figsize=(W2, 4.5))
# (a) spectrum
ax = axs[0, 0]; Tg = np.linspace(0, 2.2, 300)
ax.axhspan(-1, 1, color=GRID, alpha=.55, lw=0)
for gg, col, nm in ((gmin, BLUE, "$g_{\\min}$"), (gmax, ORANGE, "$g_{\\max}$")):
    ax.plot(Tg, 1 - alpha * Tg * gg, color=col, label=f"Euler, {nm}")
    ax.plot(Tg, np.exp(-alpha * Tg * gg), color=col, ls="--", label=f"exact, {nm}")
ax.axvline(Tstab, color=MUTED, lw=.7, ls=":")
ax.text(Tstab + .03, -1.95, "$T=2/(\\alpha g_{\\max})$", fontsize=5.5, color=SEC)
ax.set_xlabel("step time $T$"); ax.set_ylabel("eigenvalue $\\lambda(T)$"); ax.set_ylim(-2.2, 1.2)
ax.set_title("(a) Spectrum: Euler vs. exact"); ax.legend(frameon=False, loc="lower left", ncol=1, bbox_to_anchor=(0.0, 0.08)); style(ax)
# (b) Law I
ax = axs[0, 1]; Tg2 = np.linspace(0.01, 2.2, 400); t = 10
ax.semilogy(Tg2, err_euler(Tg2, t, g, c0, alpha), color=ORANGE, label="Euler model")
ax.semilogy(Tg2, err_sg(Tg2 * t, g, c0, alpha), color=BLUE, label="exact (semigroup)")
for xv, lb in ((Tmono, "$T_{mono}$"), (Tstab, "$T_{stab}$")):
    ax.axvline(xv, color=MUTED, lw=.7, ls=":"); ax.text(xv + .03, 2e2, lb, fontsize=5.5, color=SEC)
ax.set_ylim(1e-3, 1e3); ax.set_xlabel("step time $T$"); ax.set_ylabel("$\\|E\\|$ after $t=10$ steps")
ax.set_title("(b) Law I: error vs. time"); ax.legend(frameon=False, loc="lower left"); style(ax)
# (c) Law II
ax = axs[0, 2]; d = np.arange(0, 61); Tc = 1.0
act = err_sg(d * Tc, g, c0, alpha)
ax.semilogy(d, act, color=BLUE, label="$\\|E_d\\|$ (exact)")
ax.semilogy(d, np.exp(-alpha * gmin * Tc * d), color=INK, ls="--", lw=.9, label="upper $e^{-\\alpha g_{\\min}Td}$")
ax.semilogy(d, np.exp(-alpha * gmax * Tc * d), color=INK, ls=":", lw=.9, label="lower $e^{-\\alpha g_{\\max}Td}$")
ax.semilogy(d, [err_euler(Tc, dd, g, c0, alpha)[0] for dd in d], color=ORANGE, lw=1.0, label="Euler, $T=1$")
ax.set_ylim(1e-4, 2); ax.set_xlabel("depth $d$ (load $C=c_0d$)"); ax.set_ylabel("$\\|E_d\\|$")
ax.set_title("(c) Law II: error vs. depth"); ax.legend(frameon=False, loc="upper right"); style(ax)
# (d) Law III
ax = axs[1, 0]; tg = np.linspace(0, 45, 600)
ax.semilogy(tg, err_sg(tg, g, c0, alpha), color=BLUE)
for beta, off in ((0.2, (3, 1.3)), (0.05, (3, 1.3)), (0.01, (3, 1.3)), (0.002, (3, 1.3))):
    ts = tau_star(beta, g, c0, alpha); es_ = err_sg(ts, g, c0, alpha)[0]
    ax.plot([ts], [es_], "o", color=ORANGE, ms=3.8, mec="white", mew=.6, zorder=5)
    ax.annotate(f"$\\beta$={beta}", (ts, es_), xytext=(ts + 2.2, es_ * 1.5), fontsize=5.5, color=SEC)
ax.set_xlabel("latency $\\tau=1/v$"); ax.set_ylabel("error $\\varepsilon(\\tau)$"); ax.set_ylim(1e-2, 1.5)
ax.set_title("(d) Law III: speed-error equilibrium"); style(ax)
# (e) joint map
ax = axs[1, 1]
Tm = np.linspace(0.05, 2.0, 200); dm = np.linspace(1, 40, 200)
TT, DD = np.meshgrid(Tm, dm)
Z = np.log10(err_sg((TT * DD).ravel(), g, c0, alpha)).reshape(TT.shape)
im = ax.imshow(Z, origin="lower", aspect="auto", extent=[Tm[0], Tm[-1], dm[0], dm[-1]],
               cmap=CMAP.reversed(), vmin=-3.5, vmax=0, interpolation="bilinear")
cs = ax.contour(Tm, dm, Z, levels=[-3, -2, -1], colors="white", linewidths=.8, linestyles="--")
ax.clabel(cs, fmt=lambda v: f"{v:.0f}", fontsize=5.5, inline_spacing=6)
ax.set_xlabel("step time $T$"); ax.set_ylabel("depth $d$")
ax.set_title("(e) $\\log_{10}\\varepsilon(T,d)$ depends on $dT$ only")
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02); cb.ax.tick_params(labelsize=5.5, length=2)
cb.set_label("$\\log_{10}\\varepsilon$", fontsize=6)
# (f) load
ax = axs[1, 2]
ax.plot(ds, tm * 1e3, "o", color=BLUE, ms=3, mec="white", mew=.5, label="measured")
ax.plot(ds, (sl * ds + ic) * 1e3, color=ORANGE, label=f"linear fit, $R^2$={r2:.4f}")
ax.set_xlabel("depth $d$ (matvec steps)"); ax.set_ylabel("wall-clock (ms)")
ax.set_title("(f) Load $C(d)\\propto d$, $n=512$"); ax.legend(frameon=False, loc="upper left"); style(ax)
fig.tight_layout(w_pad=1.1, h_pad=1.2); fig.savefig("fig_dashboard.pdf"); plt.close(fig)

# ---- optimal Euler contraction vs kappa
fig, ax = plt.subplots(figsize=(W1, 2.2))
kk = np.linspace(1, 45, 300)
ax.plot(kk, (kk - 1) / (kk + 1), color=BLUE, label="theory $(\\kappa-1)/(\\kappa+1)$")
ax.plot(kap_pts, rho_pts, "o", color=ORANGE, ms=2.2, alpha=.7, mew=0, label="numerical minimum (400 random spectra)")
ax.set_xlabel("condition number $\\kappa=g_{\\max}/g_{\\min}$"); ax.set_ylabel("optimal contraction $\\rho^\\ast$")
ax.set_ylim(0, 1.02); ax.legend(frameon=False, loc="lower right"); style(ax)
fig.tight_layout(); fig.savefig("fig_opt.pdf"); plt.close(fig)

print("worst-case numbers in numbers.tex:", len(NUM), "macros")

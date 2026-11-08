"""Figures 8-16 for the 230B paper."""
import json, math, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as S
import bkick as B
import estimators as E
import models as M
from mplhelp import hist2d_contour

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


ANS = _load("answer.json")
SENS = _load("sens.json")
DES = _load("design.json")
BUD = _load("budget.json")
PRI = _load("sens_prior.json")
TRK = _load("tracking_noise.json")


# ---------------------------------------------------------------- f08
def f08():
    """Tornado: effect of each prior choice on the entry point."""
    base = 3.029
    rows = [  # parkey, value label, mean y, sd
        ("omega_sigma", "17", 3.164, 0.157),
        ("omega_sigma", "70", 2.838, 0.394),
        ("cl", "0.7", 3.121, 0.205),
        ("cl", "1.3", 2.976, 0.326),
        ("cd_median", "0.17", 3.020, 0.275),
        ("cd_median", "0.22", 3.031, 0.282),
        ("cd_log_sigma", "0.30", 3.030, 0.280),
        ("cd_log_sigma", "0.50", 3.017, 0.278),
        ("tau", "2 s", 3.040, 0.265),
        ("tau", "12 s", 3.017, 0.284),
    ]
    labels = [r"spin prior $\sigma_\omega$ = 17 rad/s",
              r"spin prior $\sigma_\omega$ = 70 rad/s",
              r"lift slope $C_{L0}$ = 0.7",
              r"lift slope $C_{L0}$ = 1.3",
              r"$C_D$ median 0.17",
              r"$C_D$ median 0.22",
              r"$C_D$ prior sd 0.30",
              r"$C_D$ prior sd 0.50",
              r"spin decay $\tau$ = 2 s",
              r"spin decay $\tau$ = 12 s"]
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.9, 2.9), layout="constrained")
    ys = np.arange(len(rows))[::-1]
    for (k, v, m, sd), lab, yy in zip(rows, labels, ys):
        d = m - base
        ax.plot([base, m], [yy, yy], color=PAL(k), lw=2.6, solid_capstyle="butt")
        ax.plot([m], [yy], marker="|", ms=9, color=PAL(k))
        if abs(d) > 0.02:
            ax.text(m + (0.012 if d > 0 else -0.012), yy,
                    f"{d:+.2f} m", fontsize=6.8, va="center",
                    ha="left" if d > 0 else "right", color=PAL(k))
    ax.axvline(base, color=S.PAL["ink"], lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlim(2.72, 3.28)
    ax.set_xlabel("posterior mean entry ordinate $\\bar y$  (m)")
    S.save(fig, "f08-sensitivity")


def PAL(k):
    if "omega" in k or k == "cl":
        return S.PAL["blue"]
    if "cd" in k:
        return S.PAL["rust"]
    return S.PAL["green"]


# ---------------------------------------------------------------- f09
def f09():
    """Entry uncertainty and scoring probability vs position noise."""
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTW * 0.92, 2.5),
                             layout="constrained")
    e1 = SENS["e1_sweep"]
    e2 = _load("e2_sweep.json")
    s1 = np.array([r["sigma"] for r in e1]) * 1e3
    y1 = np.array([r["std"]["y"] for r in e1])
    p1 = np.array([r["p_inside"] for r in e1])
    s2 = np.array([r["sigma"] for r in e2]) * 1e3
    y2 = np.array([r["std"] for r in e2])
    p2 = np.array([r["p"] for r in e2])
    ax = axes[0]
    ax.plot(s1, y1, "o-", color=S.PAL["rust"], ms=3.6, lw=1.2,
            label="constant-$a$ (E1)")
    ax.plot(s2, y2, "s-", color=S.PAL["blue"], ms=3.4, lw=1.2,
            label="physical (E2)")
    for x, lab in ((10, None), (15, "GLT"), (56, "open\ndata")):
        ax.axvline(x, color=S.PAL["grey"], lw=0.6, ls=":")
    ax.text(16.5, 0.12, "GLT\n1.5 cm", fontsize=6.6, color=S.PAL["grey"])
    ax.text(47, 0.12, "open tracking", fontsize=6.6, color=S.PAL["grey"])
    ax.set_xlabel("position noise $\\sigma_{\\rm pos}$  (mm)")
    ax.set_ylabel("$\\sigma_y$ of the entry point  (m)")
    ax.set_xlim(0, 62)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_yscale("log")
    ax.set_ylim(0.1, 3.0)
    ax.legend(fontsize=7, loc="upper left")
    S.panel_label(ax, "a")
    ax = axes[1]
    ax.plot(s1, p1, "o-", color=S.PAL["rust"], ms=3.6, lw=1.2)
    ax.plot(s2, p2, "s-", color=S.PAL["blue"], ms=3.4, lw=1.2)
    ax.axhline(0.5, color=S.PAL["grey"], lw=0.6, ls=":")
    for x in (10, 15, 56):
        ax.axvline(x, color=S.PAL["grey"], lw=0.6, ls=":")
    ax.set_xlabel("position noise $\\sigma_{\\rm pos}$  (mm)")
    ax.set_ylabel("$P(\\text{clean goal})$")
    ax.set_xlim(0, 62)
    ax.set_ylim(0.22, 1.04)
    S.panel_label(ax, "b")
    S.save(fig, "f09-noise")


# ---------------------------------------------------------------- f10
def f10():
    """Design map: sigma_y against number of fixes and their precision."""
    g = DES["grid"]
    ns = np.array(g["n"])
    sm = np.array(g["sigma_mm"])
    sy = np.array(g["sigma_y"]).T  # (len(sigma), len(n))? check
    if sy.shape[0] != len(sm):
        sy = np.array(g["sigma_y"])
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.62, 2.5), layout="constrained")
    cs = ax.contourf(ns, sm, sy * 1e3, levels=[10, 25, 50, 75, 100, 150, 200,
                                              300, 400],
                     cmap="Blues_r", extend="both")
    cl = ax.contour(ns, sm, sy * 1e3, levels=[50, 100, 200, 300],
                    colors=[S.PAL["ink"]], linewidths=0.7)
    ax.clabel(cl, fmt=lambda v: f"{v:.0f} mm", fontsize=6.5)
    ax.plot([2], [10], marker="o", ms=5, mfc="none", mew=1.4,
            color=S.PAL["rust"])
    ax.annotate("this problem", (2, 10), xytext=(2.6, 22), fontsize=7,
                color=S.PAL["rust"],
                arrowprops=dict(arrowstyle="-", lw=0.6,
                                color=S.PAL["rust"]))
    ax.set_xlabel("number of fixes $n$")
    ax.set_ylabel("position noise $\\sigma_{\\rm pos}$  (mm)")
    ax.set_xticks(ns)
    cb = fig.colorbar(cs, ax=ax, pad=0.02, fraction=0.06)
    cb.set_label("$\\sigma_y$  (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    S.save(fig, "f10-design")


# ---------------------------------------------------------------- f11
def f11():
    """Two-fix timing map at the adopted 5 mm noise."""
    d6 = json.load(open("../results/design6_timing.json"))
    grid = d6["grid"]
    ts = sorted({round(r[0], 4) for r in grid} | {round(r[1], 4) for r in grid})
    T = np.array(ts)
    sy = np.full((len(T), len(T)), np.nan)
    idx = {v: i for i, v in enumerate(ts)}
    for r in grid:
        sy[idx[round(r[0], 4)], idx[round(r[1], 4)]] = r[2]*1e3
    t = T
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.58, 2.55),
                           layout="constrained")
    m = np.ma.masked_invalid(sy)
    cs = ax.contourf(t, t, m, levels=[10, 25, 50, 75, 100, 150, 200, 250, 300],
                     cmap="Blues_r", extend="both")
    cl = ax.contour(t, t, m, levels=[25, 50, 100, 200], colors=[S.PAL["ink"]],
                    linewidths=0.7)
    ax.clabel(cl, fmt=lambda v: f"{v:.0f}", fontsize=6.5)
    ax.plot([0.05], [0.10], marker="o", ms=5.5, mfc="none", mew=1.4,
            color=S.PAL["rust"])
    ax.plot([0.12], [0.31], marker="*", ms=8, color=S.PAL["green"])
    ax.annotate("this problem\n211 mm", (0.05, 0.10), xytext=(0.115, 0.135),
                fontsize=6.8, color=S.PAL["rust"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.PAL["rust"]))
    ax.annotate("best\n12 mm", (0.11, 0.31), xytext=(0.175, 0.245),
                fontsize=6.8, color=S.PAL["green"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.PAL["green"]))
    ax.set_xlabel("time of first fix $t_1$  (s)")
    ax.set_ylabel("time of second fix $t_2$  (s)")
    ax.set_xlim(0.025, 0.325)
    ax.set_ylim(0.025, 0.325)
    ax.set_aspect("equal")
    cb = fig.colorbar(cs, ax=ax, pad=0.02, fraction=0.06)
    cb.set_label("$\\sigma_y$  (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    S.save(fig, "f11-timing")


# ---------------------------------------------------------------- f12
def f12():
    """Synthetic validation: accuracy and interval coverage."""
    v = _load("valid.json")
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTW * 0.95, 2.6),
                             layout="constrained")
    ax = axes[0]
    cols = {"E1": S.PAL["rust"], "E2": S.PAL["blue"], "AVG": S.PAL["green"]}
    for i, (truth, res) in enumerate(v.items()):
        for est in ("E1", "E2", "AVG"):
            yt = np.array([r["y_true"] for r in res])
            ye = np.array([r[est]["y"] for r in res])
            ax.plot(yt, ye - yt, marker=".", ms=3.0, ls="none", alpha=0.75,
                    color=cols[est], label=est if i == 0 else None)
    for y in (-0.5, 0.5):
        ax.axhline(y, color=S.PAL["grey"], lw=0.6, ls=":")
    ax.axhline(0, color=S.PAL["ink"], lw=0.8)
    ax.set_xlabel("true entry ordinate $y_{\\rm true}$  (m)")
    ax.set_ylabel("error $\\hat y-y_{\\rm true}$  (m)")
    ax.set_ylim(-1.6, 1.6)
    ax.legend(fontsize=7, loc="lower left", ncol=3, columnspacing=0.9,
              handletextpad=0.3)
    S.panel_label(ax, "a")
    ax = axes[1]
    names, lo, hi = [], [], []
    for i, (truth, res) in enumerate(v.items()):
        for est in ("E1", "E2"):
            cov68 = np.mean([r[est]["cov68"] for r in res]) * 100
            names.append(est + "\n" + ("phys." if i == 0 else "kin."))
            lo.append(cov68)
    x = np.arange(len(names))
    ax.bar(x - 0.19, lo, 0.34, color=[S.PAL["rust"] if "E1" in n else
                                      S.PAL["blue"] for n in names],
           label="68 % interval")
    h = [np.mean([r["E1"]["cov68"] for r in res]) for res in v.values()]
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.8)
    ax.axhline(68, color=S.PAL["ink"], lw=0.9, ls="--")
    ax.text(2.4, 69.5, "nominal 68 %", fontsize=7, color=S.PAL["ink"])
    ax.set_ylabel("coverage of the 68 % interval  (\\%)")
    ax.set_ylim(0, 100)
    S.panel_label(ax, "b")
    S.save(fig, "f12-valid")


# ---------------------------------------------------------------- f13
def f13():
    """Five defensible models, five answers."""
    mods = [("gravity only", 3.217, "M0"),
            ("drag $+$ gravity", 3.217, "M1"),
            ("full physics", 3.012, "M2"),
            ("constant $a$", 2.495, "M3"),
            ("ballistic from $t=0.10$ s", 3.118, "")]
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.72, 2.35),
                           layout="constrained")
    for i, (lab, y, m) in enumerate(mods):
        ax.plot([y], [i], marker="o", ms=6, color=S.PAL["blue"] if m else
                S.PAL["grey"])
        ax.text(y, i + 0.26, f"$y={y:.3f}$ m", fontsize=7, ha="center",
                color=S.PAL["ink"])
    ax.axvspan(2.495, 3.217, color=S.PAL["light"], alpha=0.35, lw=0)
    ax.axvline(2.764, color=S.PAL["green"], lw=1.1, ls="--")
    ax.text(2.780, 4.12, "model average 2.764 m", fontsize=6.8,
            color=S.PAL["green"], rotation=90, va="top")
    ax.annotate("", xy=(2.495, -0.36), xytext=(3.217, -0.36),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=S.PAL["ink"]))
    ax.text(2.856, -0.62, "model spread 0.72 m", fontsize=6.8, ha="center",
            va="top")
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels([m[0] for m in mods], fontsize=7.6)
    ax.set_ylim(-0.95, len(mods) - 0.05)
    ax.set_xlabel("predicted entry ordinate $y$  (m)")
    ax.set_xlim(2.30, 3.45)
    S.save(fig, "f13-models")


# ---------------------------------------------------------------- f14
def f14():
    """Aerodynamic inputs: drag crisis and lift vs spin parameter."""
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTW * 0.92, 2.45),
                             layout="constrained")
    ax = axes[0]
    Re = np.logspace(4.4, 6.2, 300)
    cd = 0.50 - 0.32 / (1 + np.exp(-(np.log10(Re) - 5.02) * 9.0))
    ax.semilogx(Re, cd, color=S.PAL["blue"], lw=1.4)
    ax.axvspan(9e4, 1.3e5, color=S.PAL["light"], alpha=0.4, lw=0)
    ax.plot([4.4e5], [0.19], marker="o", ms=6, mfc="none", mew=1.4,
            color=S.PAL["rust"])
    ax.annotate("penalty kick\n$Re=4.4\\times10^5$", (4.4e5, 0.19),
                xytext=(1.1e5, 0.36), fontsize=7, color=S.PAL["rust"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.PAL["rust"]))
    ax.text(1.05e5, 0.06, "drag crisis", fontsize=7, color=S.PAL["grey"])
    ax.set_xlabel("Reynolds number $Re$")
    ax.set_ylabel("drag coefficient $C_D$")
    ax.set_xlim(2.5e4, 1.5e6)
    ax.set_ylim(0, 0.55)
    S.panel_label(ax, "a")
    ax = axes[1]
    sp = np.array([0.18, 0.22, 0.34])
    cl = np.array([0.21, 0.24, 0.29])
    ax.plot(sp, 1.0 * sp, color=S.PAL["blue"], lw=1.0, ls="--",
            label="$C_L=C_{L0}S_p$, $C_{L0}=1$")
    ax.fill_between(sp, 0.7 * sp, 1.3 * sp, color=S.PAL["blue"], alpha=0.18,
                    lw=0, label="$C_{L0}\\in[0.7,1.3]$")
    ax.plot(sp, cl, "o", ms=6, color=S.PAL["rust"], label="Asai et al.")
    ax.plot([0.187], [0.21], marker="s", ms=6, mfc="none", mew=1.3,
            color=S.PAL["green"])
    ax.annotate("six-parameter\nfit", (0.187, 0.21), xytext=(0.235, 0.108),
                fontsize=7, color=S.PAL["green"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.PAL["green"]))
    ax.set_xlabel("spin parameter $S_p=\\omega R/v$")
    ax.set_ylabel("lift coefficient $C_L$")
    ax.set_xlim(0.14, 0.38)
    ax.set_ylim(0.05, 0.42)
    ax.legend(fontsize=6.8, loc="upper left")
    S.panel_label(ax, "b")
    S.save(fig, "f14-aero")


# ---------------------------------------------------------------- f15
def f15():
    """Timeline: observation window, prediction latency, keeper deadlines."""
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.95, 2.15), layout="constrained")
    ax.set_xlim(-0.175, 0.475); ax.set_ylim(-1.35, 1.15); ax.axis("off")
    ax.annotate("", xy=(0.470, 0), xytext=(-0.170, 0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=S.PAL["ink"]))
    ax.plot([0], [0], marker="|", ms=13, mew=1.8, color=S.PAL["ink"])
    ax.text(0, -0.34, "kick", fontsize=7.5, ha="center")
    ax.add_patch(Rectangle((0.10, -0.09), 0.299, 0.18, facecolor=S.PAL["blue"],
                           alpha=0.13, lw=0))
    ax.text(0.25, 0.13, "extrapolation, 0.30 s", fontsize=6.8, ha="center",
            color=S.PAL["blue"])
    for xx, lab in ((0.05, "$t_1$"), (0.10, "$t_2$")):
        ax.plot([xx], [0], marker="|", ms=10, color=S.PAL["blue"], mew=1.6)
        ax.text(xx, 0.34, lab, fontsize=8, ha="center", color=S.PAL["blue"])
    ax.plot([0.11], [0], marker="o", ms=5.2, color=S.PAL["green"], zorder=5)
    ax.annotate("prediction ready" + chr(10) + "0.11 s", (0.108, 0.02),
                xytext=(0.075, 0.80), fontsize=6.8, ha="center",
                color=S.PAL["green"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.PAL["green"]))
    ax.plot([-0.115], [0], marker="o", ms=5.2, mfc="white", mew=1.4,
            color=S.PAL["rust"], zorder=5)
    ax.annotate("side of the dive fixed" + chr(10) + "115 ms before the kick",
                (-0.112, -0.02), xytext=(-0.155, -0.78), fontsize=6.8,
                ha="left", color=S.PAL["rust"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.PAL["rust"]))
    ax.plot([0.245], [0], marker="s", ms=5.0, mfc="white", mew=1.3,
            color=S.PAL["purple"], zorder=5)
    ax.annotate("height of the shot" + chr(10) + "resolved, 0.245 s",
                (0.247, -0.02), xytext=(0.255, 0.80), fontsize=6.8, ha="left",
                color=S.PAL["purple"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.PAL["purple"]))
    ax.plot([0.399], [0], marker="v", ms=6.5, color=S.PAL["ink"], zorder=5)
    ax.annotate("ball arrives" + chr(10) + "0.399 s", (0.399, 0.03),
                xytext=(0.352, -0.72), fontsize=6.8, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.PAL["ink"]))
    ax.text(0.02, -1.18, "observation window 0.05--0.10 s", fontsize=6.8,
            color=S.PAL["blue"])
    S.save(fig, "f15-timeline")


# ---------------------------------------------------------------- f16
def f16():
    """Prior predictive of the aerodynamic acceleration."""
    rng = np.random.default_rng(7)
    v = np.array([29.6, 8.6, 6.3])
    speed = np.linalg.norm(v)
    n = 200000
    cd = 0.19 * np.exp(0.15 * rng.standard_normal(n))
    a_d = (0.5 * B.RHO * B.BALL_AREA / B.BALL_MASS) * cd[:, None] * speed * v
    om = rng.standard_normal((n, 3)) * 35.0
    km = 0.5 * B.RHO * B.BALL_AREA * B.BALL_RADIUS / B.BALL_MASS
    a_m = km * np.cross(om, v)
    a = a_d + a_m
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.62, 2.4),
                           layout="constrained")
    an = np.linalg.norm(a, axis=1)
    ax.hist(an, bins=70, color=S.PAL["blue"], alpha=0.45, lw=0, density=True,
            label="prior predictive")
    ax.axvline(np.median(an), color=S.PAL["ink"], lw=1.0, ls=":",
               label="median, %.1f" % np.median(an))
    ax.axvline(22.63, color=S.PAL["rust"], lw=1.4, ls="--",
               label="implied by the data")
    ax.text(23.1, 0.118, "99.2nd percentile\n(one kick in 130)", fontsize=6.6,
            color=S.PAL["rust"], va="top", ha="left")
    ax.set_xlabel("aerodynamic acceleration $|a_{\\rm aero}|$  (m/s$^2$)")
    ax.set_ylabel("prior predictive density")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, 34)
    ax.set_ylim(0, 0.185)
    S.save(fig, "f16-priorpred")


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in dir() if n.startswith("f") and
                             len(n) == 3 or n.startswith("f1")]
    for n in names:
        if callable(globals().get(n)):
            globals()[n]()
            print("ok", n)

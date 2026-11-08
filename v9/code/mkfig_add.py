"""Extra figures for the merged main line: rms ladder, sigma sweep, cross-check."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle as S
import paths

S.setup()
GPT = paths.GPT

# ---------------------------------------------------------------- f19 residuals
LAD = [("M0 constant $a$", 0.000, 6, "$y=2.495$"),
       ("M1 gravity", 10.376, 3, "$y=3.217$"),
       ("M2 gravity+drag", 5.029, 4, "$y=3.217$"),
       ("M3 drag+lateral", 0.542, 5, "$y=2.549$"),
       ("M5 MAP", 0.869, 7, "$y=2.707$"),
       ("M6 answer", 1.448, 6, "$y=2.785$")]


def f19():
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.62, 2.55), layout="constrained")
    names = [r[0] for r in LAD]
    rms = [r[1] for r in LAD]
    x = np.arange(len(LAD))
    ax.axhline(5.0, color=S.PAL["grey"], lw=0.7, ls=(0, (4, 3)))
    ax.text(len(LAD) - 0.55, 5.35, r"$\sigma_{\rm pos}=5$ mm", fontsize=7,
            color=S.PAL["grey"], ha="right")
    bars = ax.bar(x, rms, width=0.62,
                  color=[S.PAL["rust"] if i < 4 else S.PAL["blue"] for i in range(len(LAD))],
                  alpha=0.9, lw=0)
    for xi, r in zip(x, rms):
        ax.text(xi, r + 0.35, ("%.2f" % r) if r > 0 else "0.00", fontsize=6.6,
                ha="center", color=S.PAL["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=32, ha="right", fontsize=6.8)
    ax.set_ylabel("rms residual at the two fixes  (mm)")
    ax.set_ylim(0, 12.6)
    ax.set_yticks([0, 2.5, 5, 7.5, 10])
    return S.save(fig, "f19-residuals")


# ---------------------------------------------------------------- f20 sigma sweep
def f20():
    scen = json.load(open(GPT + "/scenario_predictions.json"))
    by = {}
    for s in scen:
        by[s["label"]] = s["summary"]
    keys = [("sigma_0.0025", 2.5), ("nominal", 5.0),
            ("sigma_0.01", 10.0), ("sigma_0.02", 20.0)]
    sig = np.array([k[1] for k in keys])
    yv = np.array([by[k[0]]["y"]["mean"] for k in keys])
    ylo = np.array([by[k[0]]["y"]["q025_q50_q975"][0] for k in keys])
    yhi = np.array([by[k[0]]["y"]["q025_q50_q975"][2] for k in keys])
    p = np.array([by[k[0]]["full_sphere_clearance_probability"] for k in keys])
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTW * 0.86, 2.35),
                             layout="constrained")
    ax = axes[0]
    ax.fill_between(sig, ylo, yhi, color=S.PAL["blue"], alpha=0.16, lw=0)
    ax.plot(sig, yv, "o-", color=S.PAL["blue"], ms=4.5, lw=1.2)
    ax.axhline(3.66, color=S.PAL["ink"], lw=0.8, ls=(0, (3, 2)))
    ax.text(2.6, 3.70, "post", fontsize=6.8, color=S.PAL["ink"])
    ax.set_xlabel(r"assumed fix noise $\sigma_{\mathrm{pos}}$  (mm)")
    ax.set_ylabel(r"mean entry ordinate $\bar y$  (m)")
    ax.set_xlim(0, 22)
    ax.set_ylim(2.3, 3.98)
    ax = axes[1]
    ax.plot(sig, p, "s-", color=S.PAL["rust"], ms=4.5, lw=1.2)
    for x_, p_ in zip(sig, p):
        ax.text(x_ - 0.75 if x_ > 15 else x_ + 0.55, p_ + 0.010, "%.3f" % p_,
                fontsize=6.4, color=S.PAL["rust"],
                ha="right" if x_ > 15 else "left")
    ax.set_xlabel(r"assumed fix noise $\sigma_{\mathrm{pos}}$  (mm)")
    ax.set_ylabel(r"$P(\mathrm{clean\ goal})$")
    ax.set_xlim(0, 22)
    ax.set_ylim(0.66, 1.04)
    return S.save(fig, "f20-sigma")


# ---------------------------------------------------------------- f21 cross-check
def f21():
    rows = [("independent 7-parameter\nBayesian fit (5 mm)", 2.7844, 0.203),
            ("our re-implementation\nof that fit (5 mm)", 2.7849, 0.203),
            ("adopted headline", 2.70, 0.0),
            ("M3 drag + lateral", 2.549, 0.0),
            ("constant-$a$ control", 2.500, 0.598),
            ("literature-prior control", 3.029, 0.277)]
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.72, 2.5), layout="constrained")
    for i, (lab, m, sd) in enumerate(rows):
        yy = len(rows) - 1 - i
        if sd > 0:
            ax.errorbar([m], [yy], xerr=[1.645 * sd], fmt="o", ms=4.6,
                        color=S.PAL["blue"], ecolor=S.PAL["light"], elinewidth=1.0,
                        capsize=2.4, zorder=5)
        else:
            ax.plot([m], [yy], "D", ms=4.4, color=S.PAL["rust"], zorder=5)
        ax.text(m, yy + 0.26, "%.3f" % m, fontsize=6.4, ha="center",
                color=S.PAL["ink"])
    ax.axvspan(2.125, 3.121, color=S.PAL["gold"], alpha=0.13, lw=0)
    ax.axvline(3.66, color=S.PAL["ink"], lw=0.8, ls=(0, (3, 2)))
    ax.text(3.60, len(rows) - 0.45, "post", fontsize=6.8, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.6)
    ax.set_xlabel(r"entry ordinate $y$  (m);  bars show 90 % intervals")
    ax.set_xlim(1.9, 3.8)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    return S.save(fig, "f21-crosscheck")


if __name__ == "__main__":
    import sys
    for n in (sys.argv[1:] or ["f19", "f20", "f21"]):
        {"f19": f19, "f20": f20, "f21": f21}[n]()
        print("ok", n, flush=True)

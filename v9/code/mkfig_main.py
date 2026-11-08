"""Figures for the main (six-parameter) line, built from the GPT v5 result files."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as S, mplhelp as MH
import paths

S.setup()
GPT = paths.GPT
NOM = np.load(GPT + "/geometry_nominal.npz")
YZ = NOM["center_yz"]; W = NOM["w"] / NOM["w"].sum(); FC = NOM["full_clear"]
SCEN = json.load(open(GPT + "/scenario_predictions.json"))
ADOPT = (2.6723, 1.9373)
M6 = (2.7847, 1.9452)
M3 = (2.5486, 1.8931)
ENV_Y = (2.125, 3.121)
B, H, R = 3.66, 2.44, 0.110


def f03():
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.80, 2.85), layout="constrained")
    ax.add_patch(Rectangle((-B, 0), 2 * B, H, facecolor="#f2f5fa",
                           edgecolor=S.PAL["ink"], lw=1.6, zorder=1))
    ax.add_patch(Rectangle((-(B - R), R), 2 * (B - R), H - 2 * R,
                           facecolor="none", edgecolor=S.PAL["grey"],
                           lw=0.8, ls=(0, (4, 3)), zorder=2))
    idx = np.random.default_rng(7).choice(len(YZ), 12000, replace=False,
                                         p=W / W.sum())
    ax.scatter(YZ[idx, 0], YZ[idx, 1], s=1.1, color=S.PAL["blue"],
               alpha=0.10, lw=0, rasterized=True, zorder=3)
    MH.hist2d_contour(ax, YZ[:, 0], YZ[:, 1], levels=(0.28, 0.60, 0.90),
                      color=S.PAL["blue"], nbin=90, smooth=1.3, w=W,
                      ylim=(-0.3, 3.25), zlim=(-0.3, 3.25), label="predictive")
    ax.plot([ADOPT[0]], [ADOPT[1]], marker="s", ms=7, mfc="none", mew=1.5,
            color=S.PAL["green"], zorder=8, label=f"mean of the four: ({ADOPT[0]:.2f}, {ADOPT[1]:.2f}) m")
    ax.plot([M6[0]], [M6[1]], marker="D", ms=6, mfc="none", mew=1.4,
            color=S.PAL["blue"], zorder=8, label=f"six-parameter answer: ({M6[0]:.3f}, {M6[1]:.3f}) m")
    ax.plot([M3[0]], [M3[1]], marker="o", ms=6, mfc="none", mew=1.4,
            color=S.PAL["rust"], zorder=8, label=f"M3, no spin: ({M3[0]:.3f}, {M3[1]:.3f}) m")
    ax.legend(fontsize=6.4, loc="upper left", bbox_to_anchor=(0.01, 0.99),
              handlelength=1.2, borderpad=0.4, labelspacing=0.3,
              frameon=True, framealpha=0.95, facecolor="white",
              edgecolor=S.PAL["light"])
    ax.set_xlim(-3.95, 3.95); ax.set_ylim(-0.30, 3.25)
    ax.set_aspect("equal")
    ax.set_xlabel("$y$ at the goal plane  (m)")
    ax.set_ylabel("$z$  (m)")
    return S.save(fig, "f03-goalplane")


def f04():
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTW * 0.95, 2.45),
                             layout="constrained")
    for ax, i, lab, lim, ref, refl in (
            (axes[0], 0, "$y$ at the goal plane  (m)", (0.4, 5.0), B, "post"),
            (axes[1], 1, "$z$ at the goal plane  (m)", (0.4, 3.2), H, "crossbar")):
        v = YZ[:, i]
        ax.hist(v, bins=80, weights=W, density=True, histtype="step",
                color=S.PAL["blue"], lw=1.1)
        ax.axvline(ref, color=S.PAL["grey"], lw=0.9, ls=":")
        ax.annotate(refl, (ref, 0.85 * ax.get_ylim()[1]), xytext=(ref + 0.18, 0.72 * ax.get_ylim()[1]),
                    fontsize=6.8, color=S.PAL["ink"])
        ax.set_xlim(*lim)
        ax.set_xlabel(lab)
        ax.set_ylabel("weighted density")
    return S.save(fig, "f04-histograms")


def f08():
    rows = [s for s in SCEN if s["group"] != "nominal"]
    rows = sorted(rows, key=lambda s: s["summary"]["y"]["mean"])
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.95, 3.05), layout="constrained")
    ys = np.array([s["summary"]["y"]["mean"] for s in rows])
    ps = np.array([s["summary"]["full_sphere_clearance_probability"] for s in rows])
    labs = [s["label"] for s in rows]
    cmap = plt.get_cmap("coolwarm")
    for i, (y, p) in enumerate(zip(ys, ps)):
        ax.plot([ADOPT[0], y], [i, i], color=S.PAL["light"], lw=0.7, zorder=1)
        ax.plot([y], [i], marker="o", ms=4.6, color=cmap((p - ps.min()) / (ps.max() - ps.min())),
                zorder=3)
        ax.text(3.56, i, f"{p:.2f}", fontsize=5.9, va="center", ha="right",
                color=S.PAL["grey"])
    ax.axvline(ADOPT[0], color=S.PAL["green"], lw=1.0, ls="--")
    ax.axvspan(*ENV_Y, color=S.PAL["light"], alpha=0.30, lw=0)
    ax.text(ADOPT[0] - 0.012, len(rows) - 0.7, "mean of the four 2.67 m", fontsize=6.4,
            color=S.PAL["green"], rotation=90, va="top", ha="right")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labs, fontsize=5.8)
    ax.set_ylim(-0.95, len(rows) - 0.3)
    ax.set_xlim(2.45, 3.60)
    ax.set_xlabel("posterior mean entry ordinate $\\bar y$  (m);  right column: $P(\\mathrm{clearance})$;\n"
                  "grey band: model envelope $y\\in[2.125,\\,3.121]$ m")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(ps.min(), ps.max()))
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.032)
    cb.set_label("$P(\\mathrm{clearance})$", fontsize=7)
    return S.save(fig, "f08-scenarios")


def f13():
    mods = [("M0 constant $a$", 2.4948), ("M1 gravity", 3.2170),
            ("M2 gravity + drag", 3.2172), ("M3 drag + lateral", 2.5486),
            ("M4 force decomposition", 2.6481), ("M5 MAP", 2.7073),
            ("M6 six-parameter (answer)", 2.7847)]
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.82, 2.35), layout="constrained")
    for i, (n, y) in enumerate(mods):
        c = S.PAL["blue"] if "answer" in n else S.PAL["grey"]
        ax.plot([y], [i], marker="o", ms=5.5, color=c, zorder=4)
        ax.text(y, i + 0.26, f"{y:.3f}", fontsize=6.4, ha="center", color=c)
    ax.axvline(ADOPT[0], color=S.PAL["green"], lw=1.0, ls="--")
    ax.text(ADOPT[0] - 0.018, 6.35, "mean of the four 2.67 m", fontsize=6.4,
            color=S.PAL["green"], rotation=90, va="top", ha="right")
    ax.axvspan(2.4948, 3.2172, color=S.PAL["light"], alpha=0.35, lw=0)
    ax.annotate("", xy=(2.4948, -0.85), xytext=(3.2172, -0.85),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=S.PAL["ink"]))
    ax.text(2.856, -1.12, "model-form spread 0.72 m", fontsize=6.4, ha="center",
            va="top")
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels([m[0] for m in mods], fontsize=7)
    ax.set_ylim(-1.30, len(mods) - 0.05)
    ax.set_xlim(2.42, 3.30)
    ax.set_xlabel("predicted entry ordinate $y$  (m)")
    return S.save(fig, "f13-ladder")


if __name__ == "__main__":
    import sys
    for n in (sys.argv[1:] or ["f03", "f04", "f08", "f13"]):
        globals()[n]()
        print("ok", n, flush=True)

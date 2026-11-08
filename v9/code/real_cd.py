"""What real 2-D trajectories can and cannot say about C_D.

For every one-second window of a real 25 Hz ball track whose mean speed exceeds
12 m/s, the horizontal positions are fitted with the drag-only flight model
r(t) = r0 + v0 ln(1 + k|v0|t)/(k|v0|) and C_D = 2 m k /(rho A).  The same
pipeline is then applied to synthetic flights with a known C_D = 0.20, sampled at
25 Hz with 5 cm noise, which measures the noise floor of the estimator: the
spread it produces is the uncertainty of any single real window.
"""
import csv, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from drag_real2 import load, fit, CD
import figstyle as S

G = 9.81
RHO, A, M = 1.20, math.pi * 0.110 ** 2, 0.430


def windows_fit(dat, vmin=12.0, win=25, name=""):
    t, x, y = dat[:, 0], dat[:, 1], dat[:, 2]
    ok = np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    out, i = [], 0
    while i + win < len(t):
        tt = t[i:i + win]
        if tt[-1] - tt[0] > 1.4:
            i += 1; continue
        xx, yy = x[i:i + win], y[i:i + win]
        v = np.hypot(np.gradient(xx, tt), np.gradient(yy, tt))
        if v.mean() < vmin:
            i += 5; continue
        r = fit(tt - tt[0], xx, yy,
                [xx[0], yy[0], np.gradient(xx, tt)[0], np.gradient(yy, tt)[0],
                 math.log(0.0104)])
        if r is not None:
            p, rms = r
            k = math.exp(p[4])
            if rms < 0.15 and 0.002 < k < 0.06:
                out.append(dict(dataset=name, cd=CD * k, rms=rms,
                                speed=float(np.hypot(p[2], p[3]))))
        i += win
    return out


def synth(truth=0.20, n=260, seed=0):
    rng = np.random.default_rng(seed)
    k = 0.5 * RHO * A * truth / M
    out = []
    for _ in range(n):
        v0 = rng.uniform(12, 26); elev = math.radians(rng.uniform(0, 30))
        v = np.array([v0 * math.cos(elev), 0.0, v0 * math.sin(elev)])
        p = np.zeros(3); tt = 0.0; hist = []
        while tt < 2.2:
            s = np.linalg.norm(v)
            v = v + (-k * s * v - np.array([0, 0, G])) * 0.001
            p = p + v * 0.001; tt += 0.001
            if p[2] < 0 and tt > 0.05:
                break
            if abs(tt / 0.04 - round(tt / 0.04)) < 5e-4:
                hist.append([tt, p[0] + rng.normal(0, 0.05),
                             p[1] + rng.normal(0, 0.05)])
        if len(hist) > 30:
            out += windows_fit(np.asarray(hist), name="synth")
    return out


if __name__ == "__main__":
    base = ("/Users/peiziye/Desktop/国际物理竞赛/2026/"
            "UPC 2026 Problem B 资料库/08_数据集/")
    met = load(base + "Metrica-第1场-球轨迹.csv",
               ["time_s", "ball_x", "ball_y"], scale=(105.0, 68.0))
    ids = load(base + "IDSSE-J03WPY-ball.csv", ["frame", "x", "y"])
    ids[:, 0] = (ids[:, 0] - ids[:, 0][0]) * 0.04
    real = windows_fit(met, name="metrica") + windows_fit(ids, name="idsse")
    syn = synth()
    json.dump({"real": real, "synth": syn},
              open("../results/real_cd.json", "w"), indent=1)

    ri = np.array([r["cd"] for r in real if r["dataset"] == "idsse"])
    rm = np.array([r["cd"] for r in real if r["dataset"] == "metrica"])
    sc = np.array([r["cd"] for r in syn])
    print(f"metrica n={len(rm)}  IDSSE n={len(ri)}  synth n={len(sc)}")
    if len(ri):
        print("IDSSE  C_D median %.3f  IQR %.3f-%.3f"
              % (np.median(ri), *np.percentile(ri, [25, 75])))
    if len(sc):
        print("synth(0.20) C_D median %.3f  IQR %.3f-%.3f  (noise floor)"
              % (np.median(sc), *np.percentile(sc, [25, 75])))

    S.setup()
    fig, ax = plt.subplots(figsize=(S.TEXTW * 0.78, 2.5), layout="constrained")
    bins = np.linspace(0, 1.2, 61)
    if len(sc):
        ax.hist(sc, bins=bins, density=True, color=S.PAL["blue"], alpha=0.30,
                label="synthetic flights, $C_D=0.20$")
    if len(ri):
        ax.hist(ri, bins=bins, density=True, histtype="step", lw=1.1,
                color=S.PAL["rust"], label="real World Cup tracks (2-D, 25 Hz)")
    ax.axvspan(0.19, 0.22, color=S.PAL["green"], alpha=0.18, lw=0)
    ax.axvline(0.347, color=S.PAL["ink"], ls="--", lw=0.9)
    ax.text(0.347, ax.get_ylim()[1] * 0.92, " fitted from the two fixes: 0.35",
            fontsize=6.4, va="top")
    ax.text(0.205, 0.02, " wind tunnel\n 0.19-0.22", fontsize=6.4,
            color=S.PAL["green"], va="bottom", rotation=90)
    ax.set_xlabel("effective drag coefficient $C_D$ recovered from a 1 s window")
    ax.set_ylabel("probability density")
    ax.set_xlim(0, 1.1)
    ax.legend(fontsize=6.6, loc="upper right")
    S.save(fig, "f23-realdrag")

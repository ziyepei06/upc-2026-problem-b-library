"""Calibrate the 2-D drag estimator on synthetic flights with a known C_D.

The estimator of drag_real2.py works on horizontal coordinates only, so a lofted
flight biases the fitted coefficient upward by 1/cos(elevation) and by whatever
the window selection does.  This script generates flights with a known C_D,
samples them at 25 Hz with 5 cm noise (the real tracking accuracy measured in
S6.10), applies exactly the same window selection and fit, and reports the
distribution of the recovered coefficient.  Comparing that distribution with the
real-data one says whether the real passes are consistent with the literature
value.
"""
import math, json
import numpy as np
from drag_real2 import fit, RHO, A, M, CD

G = 9.81
K = lambda cd: 0.5 * RHO * A * cd / M


def flight(v0, elev_deg, cd, tmax=3.0, dt=0.001):
    th = math.radians(elev_deg)
    v = np.array([v0 * math.cos(th), 0.0, v0 * math.sin(th)])
    p = np.array([0.0, 0.0, 0.0])
    k = K(cd)
    out_t, out_xy, out_z = [], [], []
    t = 0.0
    while t < tmax:
        s = np.linalg.norm(v)
        a = -k * s * v - np.array([0, 0, G])
        v = v + a * dt
        p = p + v * dt
        t += dt
        if p[2] < 0 and t > 0.05:
            break
        out_t.append(t); out_xy.append(p[:2].copy()); out_z.append(p[2])
    return np.array(out_t), np.array(out_xy), np.array(out_z)


def sample(t, xy, fps=25.0, sigma=0.05, rng=None):
    tt = np.arange(0, t[-1], 1.0 / fps)
    x = np.interp(tt[:-1], t, xy[:, 0])
    y = np.interp(tt[:-1], t, xy[:, 1])
    x = x + rng.normal(0, sigma, len(x))
    y = y + rng.normal(0, sigma, len(y))
    return np.column_stack([tt[:-1], x, y])


def run(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        v0 = rng.uniform(12, 26)
        elev = rng.uniform(0, 45)
        cd = 0.20
        t, xy, z = flight(v0, elev, cd)
        if len(t) < 40:
            continue
        dat = sample(t, xy, rng=rng)
        # same window selector as the real-data run
        win = 25
        i = 0
        while i + win < len(dat):
            tt = dat[i:i + win, 0]
            xx = dat[i:i + win, 1]; yy = dat[i:i + win, 2]
            if tt[-1] - tt[0] > 1.4:
                i += 1; continue
            sp = np.hypot(np.gradient(xx, tt), np.gradient(yy, tt))
            if sp.mean() < 12:
                i += 5; continue
            ds = np.diff(sp)
            if ds.max() > 1.2 or np.median(ds) > -0.02:
                i += 1; continue
            r = fit(tt - tt[0], xx, yy,
                    [xx[0], yy[0], np.gradient(xx, tt)[0],
                     np.gradient(yy, tt)[0], math.log(0.0104)])
            if r is not None:
                p, rms = r
                k = math.exp(p[4])
                if rms <= 0.10 and 0.002 < k < 0.06:
                    rows.append(dict(cd=CD * k, speed=math.hypot(p[2], p[3]),
                                     elev=elev, v0=v0, rms=rms))
            i += win
    return rows


if __name__ == "__main__":
    rows = run(400)
    c = np.array([r["cd"] for r in rows]); s = np.array([r["speed"] for r in rows])
    print(f"n={len(c)}  recovered C_D (truth 0.200): median {np.median(c):.3f}"
          f"  IQR {np.percentile(c,25):.3f}-{np.percentile(c,75):.3f}")
    for lo, hi in ((12, 16), (16, 20), (20, 26)):
        m = (s >= lo) & (s < hi)
        if m.sum() > 4:
            print(f"  v {lo}-{hi}: median {np.median(c[m]):.3f} (n={m.sum()})")
    json.dump(rows, open("../results/drag_calib.json", "w"), indent=1)

"""Effective drag coefficient from real 2-D ball trajectories (robust version).

A free-flying ball has a smooth, monotonically decreasing speed; a bounce, a
touch or a rolling ball breaks that.  We therefore slide a 1 s window over each
tracking series and keep windows in which (i) the mean speed exceeds 12 m/s,
(ii) the speed decreases monotonically within the noise, and (iii) the drag-model
fit residual is below 0.10 m.  Each window gives one estimate of

    k = rho A C_D / (2 m),   dv_xy/dt = -k |v| v_xy,

fitted by Gauss-Newton on the closed-form solution.  Because |v| >= |v_xy|, the
fitted k is an upper bound on the true one, inflated by 1/cos(elevation) of the
flight; for the low driven passes that dominate the sample this is a few per
cent.
"""
import csv, json, math
import numpy as np

RHO, A, M = 1.20, math.pi * 0.110 ** 2, 0.430
CD = 2.0 * M / (RHO * A)          # C_D = CD * k
DT = 0.04


def load(path, cols, scale=None):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                row = [float(r[c]) for c in cols]
            except (ValueError, KeyError):
                continue
            rows.append(row)
    a = np.asarray(rows, float)
    if scale is not None:
        a[:, 1] *= scale[0]
        a[:, 2] *= scale[1]
    return a


def model(p, tt):
    if not (-6.5 < p[4] < 0.0):
        return None
    k = math.exp(p[4])
    s = math.hypot(p[2], p[3])
    if not np.isfinite(s) or s < 0.5:
        return None
    f = np.log1p(k * s * tt) / (k * s)
    return np.column_stack([p[0] + p[2] * f, p[1] + p[3] * f])


def fit(tt, xx, yy, p0, iters=60):
    p = np.array(p0, float)
    best = None
    for _ in range(iters):
        m = model(p, tt)
        if m is None:
            return None
        res = np.concatenate([xx - m[:, 0], yy - m[:, 1]])
        chi = float(res @ res)
        if best is None or chi < best[0]:
            best = (chi, p.copy())
        J = np.empty((2 * len(tt), 5))
        for j in range(5):
            h = 1e-6 * max(1.0, abs(p[j]))
            dp = np.zeros(5); dp[j] = h
            a, b = model(p + dp, tt), model(p - dp, tt)
            if a is None or b is None:
                return best[1], math.sqrt(best[0] / (2 * len(tt)))
            J[:, j] = -np.concatenate([a[:, 0] - b[:, 0],
                                       a[:, 1] - b[:, 1]]) / (2 * h)
        if not np.all(np.isfinite(J)):
            break
        try:
            step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        step = np.clip(step, -1.5, 1.5)
        lam = 1.0
        for _ in range(14):
            pn = p + lam * step
            mn = model(pn, tt)
            if mn is not None:
                rn = np.concatenate([xx - mn[:, 0], yy - mn[:, 1]])
                if float(rn @ rn) < chi:
                    p = pn
                    break
            lam *= 0.5
        else:
            break
    if best is None:
        return None
    chi, p = best
    return p, math.sqrt(chi / (2 * len(tt)))


def windows(dat, win=25, vmin=12.0):
    t, x, y = dat[:, 0], dat[:, 1], dat[:, 2]
    ok = np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    vx = np.gradient(x, t); vy = np.gradient(y, t)
    sp = np.hypot(vx, vy)
    i = 0
    while i + win < len(t):
        tt = t[i:i + win]
        if tt[-1] - tt[0] > 1.4 or np.median(np.diff(tt)) > 0.06:
            i += 1
            continue
        s = sp[i:i + win]
        if s.mean() < vmin:
            i += 5
            continue
        ds = np.diff(s)
        if ds.max() > 1.2 or np.median(ds) > -0.02:      # bounce or not decaying
            i += 1
            continue
        yield tt - tt[0], x[i:i + win], y[i:i + win], s[0]
        i += win


def analyse(dat, name):
    out = []
    for t0, xx, yy, s0 in windows(dat):
        vx = np.gradient(xx, t0); vy = np.gradient(yy, t0)
        r = fit(t0, xx, yy, [xx[0], yy[0], vx[0], vy[0], math.log(0.0104)])
        if r is None:
            continue
        p, rms = r
        k = math.exp(p[4])
        if rms > 0.10 or not (0.002 < k < 0.06):
            continue
        out.append(dict(dataset=name, k=k, cd=CD * k, rms=rms,
                        speed=math.hypot(p[2], p[3]), dur=float(t0[-1])))
    return out


if __name__ == "__main__":
    base = ("/Users/peiziye/Desktop/国际物理竞赛/2026/"
            "UPC 2026 Problem B 资料库/08_数据集/")
    met = load(base + "Metrica-第1场-球轨迹.csv",
               ["time_s", "ball_x", "ball_y"], scale=(105.0, 68.0))
    ids = load(base + "IDSSE-J03WPY-ball.csv", ["frame", "x", "y"])
    ids[:, 0] = (ids[:, 0] - ids[:, 0][0]) * DT
    res = analyse(met, "metrica") + analyse(ids, "idsse")
    json.dump(res, open("../results/drag_real.json", "w"), indent=1)
    for name in ("metrica", "idsse"):
        c = np.array([r["cd"] for r in res if r["dataset"] == name])
        if len(c) < 3:
            print(name, "too few:", len(c)); continue
        print(f"{name:8s} n={len(c):4d}  C_D median {np.median(c):.3f}"
              f"  IQR {np.percentile(c,25):.3f}-{np.percentile(c,75):.3f}"
              f"  16-84 % [{np.percentile(c,16):.3f},{np.percentile(c,84):.3f}]")
    c = np.array([r["cd"] for r in res])
    s = np.array([r["speed"] for r in res])
    print(f"ALL      n={len(c):4d}  C_D median {np.median(c):.3f}"
          f"  IQR {np.percentile(c,25):.3f}-{np.percentile(c,75):.3f}")
    for lo, hi in ((12, 16), (16, 20), (20, 26)):
        m = (s >= lo) & (s < hi)
        if m.sum() > 4:
            print(f"  v {lo}-{hi}: median {np.median(c[m]):.3f} (n={m.sum()})")

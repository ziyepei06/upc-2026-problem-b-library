"""Estimate the effective drag coefficient from real 2-D ball trajectories.

Two public datasets are used: the Metrica sample game (25 Hz, pitch-normalised
coordinates) and the IDSSE 2022 World Cup game (25 Hz, metres).  In the
horizontal plane a ball in flight obeys

    d v_xy / dt = -k |v| v_xy,      k = rho A C_D / (2 m),

whose solution is a straight line with logarithmically decaying speed,

    r_xy(t) = r0 + v0 / (k |v0|) * ln(1 + k |v0| t).

Each fast segment is fitted with that model (5 parameters) by Gauss-Newton and
the drag coefficient follows from k.  Gravity does not act in this plane, so no
vertical information is needed; a lofted ball biases C_D upward only by
1/cos(elevation).
"""
import csv, json, math
import numpy as np

RHO, A, M = 1.20, math.pi * 0.110 ** 2, 0.430
CD_PER_K = 2.0 * M / (RHO * A)          # C_D = CD_PER_K * k
DT = 0.04


def load(path, cols, scale=None):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                row = [float(r[c]) for c in cols]
            except ValueError:
                continue
            rows.append(row)
    a = np.asarray(rows)
    if scale is not None:
        a[:, 1] *= scale[0]
        a[:, 2] *= scale[1]
    return a                                  # columns: time, x, y


def segments(dat, vmin=8.0, minlen=30, gap=0.12):
    t, x, y = dat[:, 0], dat[:, 1], dat[:, 2]
    ok = np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    sp = np.hypot(vx, vy)
    fast = sp > vmin
    out, i, n = [], 0, len(t)
    while i < n:
        if not fast[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fast[j + 1] and t[j + 1] - t[j] < gap:
            j += 1
        if j - i + 1 >= minlen:
            out.append((t[i:j + 1], x[i:j + 1], y[i:j + 1]))
        i = j + 1
    return out


def model(p, tt):
    r0 = p[0:2]; v0 = p[2:4]
    if not (-6.0 < p[4] < 0.0):
        return None
    k = math.exp(p[4])
    s = np.linalg.norm(v0)
    if not np.isfinite(s) or s < 1e-6:
        return None
    out = np.empty((len(tt), 2))
    if k * s * tt[-1] <= -0.999:
        return None
    f = np.log1p(k * s * tt) / (k * s)
    out[:, 0] = r0[0] + v0[0] * f
    out[:, 1] = r0[1] + v0[1] * f
    return out


def fit(tt, xx, yy, p0, iters=40):
    p = np.array(p0, float)
    for _ in range(iters):
        m = model(p, tt)
        if m is None:
            return None
        res = np.concatenate([xx - m[:, 0], yy - m[:, 1]])
        J = np.empty((2 * len(tt), 5))
        for j in range(5):
            dp = np.zeros(5); h = 1e-6 * max(1.0, abs(p[j]))
            dp[j] = h
            a = model(p + dp, tt); b = model(p - dp, tt)
            if a is None or b is None:
                return None
            J[:, j] = -np.concatenate([(a[:, 0] - b[:, 0]), (a[:, 1] - b[:, 1])]) / (2 * h)
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(res)):
            return None
        try:
            step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        step = np.clip(step, -1.0, 1.0)
        lam = 1.0
        for _ in range(12):
            pn = p + lam * step
            mn = model(pn, tt)
            if mn is not None and np.sum((np.concatenate([xx - mn[:, 0], yy - mn[:, 1]])) ** 2) < np.sum(res ** 2):
                p = pn
                break
            lam *= 0.5
        else:
            break
    m = model(p, tt)
    if m is None:
        return None
    rms = math.sqrt(np.mean((xx - m[:, 0]) ** 2 + (yy - m[:, 1]) ** 2))
    return p, rms


def analyse(dat, name):
    out = []
    for tt, xx, yy in segments(dat):
        t0 = tt - tt[0]
        vx = np.gradient(xx, t0); vy = np.gradient(yy, t0)
        v0 = [vx[0], vy[0]]
        p0 = [xx[0], yy[0], v0[0], v0[1], math.log(0.0104)]
        r = fit(t0, xx, yy, p0)
        if r is None:
            continue
        p, rms = r
        k = math.exp(p[4])
        cd = CD_PER_K * k
        sp0 = math.hypot(p[2], p[3])
        if not (0.03 < k < 0.06) or rms > 0.45 or sp0 < 8:
            continue
        out.append(dict(dataset=name, n=len(t0), dur=float(t0[-1]),
                        speed=float(sp0), k=float(k), cd=float(cd),
                        rms=float(rms)))
    return out


if __name__ == "__main__":
    base = ("/Users/peiziye/Desktop/国际物理竞赛/2026/"
            "UPC 2026 Problem B 资料库/08_数据集/")
    met = load(base + "Metrica-第1场-球轨迹.csv",
               ["time_s", "ball_x", "ball_y"], scale=(105.0, 68.0))
    ids = load(base + "IDSSE-J03WPY-ball.csv", ["frame", "x", "y"])
    ids[:, 0] *= DT
    res = analyse(met, "metrica") + analyse(ids, "idsse")
    cd = np.array([r["cd"] for r in res])
    sp = np.array([r["speed"] for r in res])
    print(f"segments: {len(res)}  (metrica {sum(r['dataset']=='metrica' for r in res)},"
          f" idsse {sum(r['dataset']=='idsse' for r in res)})")
    if len(cd):
        print("C_D  median %.3f  mean %.3f  IQR %.3f-%.3f  n=%d"
              % (np.median(cd), cd.mean(),
                 *np.percentile(cd, [25, 75]), len(cd)))
        for lo, hi in ((8, 12), (12, 18), (18, 25)):
            s = cd[(sp >= lo) & (sp < hi)]
            if len(s) > 4:
                print(f"  speed {lo}-{hi} m/s: median {np.median(s):.3f}"
                      f"  n={len(s)}")
    json.dump(res, open("../results/drag_real.json", "w"), indent=1)

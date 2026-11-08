"""Linearised design study on the six-parameter main line."""
import json, numpy as np, fit6 as F

MAP = np.array([30.3726, 9.2809, 6.9108, 0.3466, 0.002103, -0.000840])
PRIOR_SD = np.array([15., 15., 10., 0.15, 0.008, 0.008])


def pred(th, dt=0.0025):
    return np.array(F.cross_plane(th, dt=dt))


def jac(th, h=1e-4):
    J = np.zeros((6, 6)); G = np.zeros((3, 6))
    for i in range(6):
        step = h * max(1.0, abs(th[i]))
        tp = th.copy(); tp[i] += step
        tm = th.copy(); tm[i] -= step
        G[:, i] = (pred(tp) - pred(tm)) / (2 * step)
        J[:, i] = (np.array(F.positions(tp)) - np.array(F.positions(tm))).ravel() / (2 * step)
    return J, G


def sigma(sig_pos, times=(0.05, 0.10), th=MAP):
    J, G = jac(th)
    A = J.T @ J / sig_pos**2
    P = np.diag(1.0 / PRIOR_SD**2)
    C = np.linalg.inv(A + P)
    cov = G @ C @ G.T
    return np.sqrt(np.diag(cov))


out = {}
for s in (0.0029, 0.005, 0.010, 0.020):
    st, sy, sz = sigma(s)
    out['%.4f' % s] = dict(t=float(st), y=float(sy), z=float(sz))
    print('sigma_pos %.4f m -> sigma_t %.4f s, sigma_y %.3f m, sigma_z %.3f m' % (s, st, sy, sz))
json.dump(out, open('../results/design6.json', 'w'), indent=1)

# ---- timing grid: which pair of fix epochs is best at sigma_pos = 5 mm ----
def sigma_at_times(times, sig_pos, th=MAP):
    J = np.zeros((6, 6)); G = np.zeros((3, 6)); h = 1e-4
    for i in range(6):
        step = h * max(1.0, abs(th[i]))
        tp = th.copy(); tp[i] += step
        tm = th.copy(); tm[i] -= step
        G[:, i] = (pred(tp) - pred(tm)) / (2 * step)
        J[:, i] = (np.array(F.positions(tp, times=times)) -
                   np.array(F.positions(tm, times=times))).ravel() / (2 * step)
    C = np.linalg.inv(J.T @ J / sig_pos**2 + np.diag(1.0 / PRIOR_SD**2))
    return np.sqrt(np.diag(G @ C @ G.T))

grid = np.round(np.arange(0.03, 0.315, 0.02), 3)
best = None
rows = []
for t1 in grid:
    for t2 in grid:
        if t2 <= t1:
            continue
        st, sy, sz = sigma_at_times((t1, t2), 0.005)
        rows.append((round(t1, 3), round(t2, 3), float(sy), float(sz), float(st)))
        if best is None or sy < best[2]:
            best = (round(t1, 3), round(t2, 3), float(sy), float(sz), float(st))
print('best pair at 5 mm:', best)
here = sigma_at_times((0.05, 0.10), 0.005)
print('problem pair at 5 mm: sigma_y %.3f m' % here[1])
print('factor', round(best[2] / here[1], 3))
json.dump(dict(best=best, here=list(map(float, here)), grid=rows),
          open('../results/design6_timing.json', 'w'))

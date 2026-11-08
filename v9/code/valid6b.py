"""Fast coverage validation of the six-parameter estimator (referee issue 2).

Same model as fit6.py; pure-Python scalar RK4 so a full trial (draw truth ->
synthesise two fixes at 5 mm -> Gauss-Newton MAP -> random-walk Metropolis ->
propagate to x = 11 -> coverage) costs about two seconds.
Two arms: truths from the same law, and truths with an extra vertical force of
2 m/s^2 that the fitted model does not contain.
"""
import json, sys, math
import numpy as np
import fit6 as F

G = 9.81; RAD = 0.110; XG = 11.0
FACT = F.FACTOR
MU = np.array([30., 0., 7., 0.25, 0., 0.])
SD = np.array([15., 15., 10., 0.15, 0.008, 0.008])
LO = np.array([5., -30., 0., 0., -0.08, -0.08])
HI = np.array([60., 30., 30., 1., 0.08, 0.08])


def beta_of(th):
    vx, vy, vz = th[0], th[1], th[2]
    n = math.sqrt(vx*vx + vy*vy + vz*vz)
    e0x, e0y, e0z = vx/n, vy/n, vz/n
    # e1 = z x e0
    e1x, e1y, e1z = -e0y, e0x, 0.0
    m = math.sqrt(e1x*e1x + e1y*e1y)
    e1x, e1y = e1x/m, e1y/m
    # e2 = e0 x e1
    e2x, e2y, e2z = e0y*e1z - e0z*e1y, e0z*e1x - e0x*e1z, e0x*e1y - e0y*e1x
    return (th[4]*e1x + th[5]*e2x, th[4]*e1y + th[5]*e2y, th[4]*e1z + th[5]*e2z)


def advance(th, targets, extra_z=0.0, dt=0.002):
    """Return positions at the requested target times (list) and the crossing."""
    vx, vy, vz = th[0], th[1], th[2]
    bx, by, bz = beta_of(th)
    kd = FACT*th[3]
    px, py, pz = 0.0, 0.0, 0.11
    out = []; t = 0.0
    def acc(x, y, z):
        s = math.sqrt(x*x + y*y + z*z)
        return (-kd*s*x + s*(by*z - bz*y),
                -kd*s*y + s*(bz*x - bx*z),
                -kd*s*z + s*(bx*y - by*x) - G + extra_z)
    for tgt in targets:
        while t < tgt - 1e-12:
            h = min(dt, tgt - t)
            a1 = acc(vx, vy, vz)
            a2 = acc(vx + .5*h*a1[0], vy + .5*h*a1[1], vz + .5*h*a1[2])
            a3 = acc(vx + .5*h*a2[0], vy + .5*h*a2[1], vz + .5*h*a2[2])
            a4 = acc(vx + h*a3[0], vy + h*a3[1], vz + h*a3[2])
            px += h/6*(vx + 2*(vx + .5*h*a1[0]) + 2*(vx + .5*h*a2[0]) + vx + h*a3[0])
            py += h/6*(vy + 2*(vy + .5*h*a1[1]) + 2*(vy + .5*h*a2[1]) + vy + h*a3[1])
            pz += h/6*(vz + 2*(vz + .5*h*a1[2]) + 2*(vz + .5*h*a2[2]) + vz + h*a3[2])
            vx += h/6*(a1[0] + 2*a2[0] + 2*a3[0] + a4[0])
            vy += h/6*(a1[1] + 2*a2[1] + 2*a3[1] + a4[1])
            vz += h/6*(a1[2] + 2*a2[2] + 2*a3[2] + a4[2])
            t += h
        out.append((px, py, pz))
    # crossing
    while px < XG and t < 1.0:
        h = dt
        a1 = acc(vx, vy, vz)
        a2 = acc(vx + .5*h*a1[0], vy + .5*h*a1[1], vz + .5*h*a1[2])
        a3 = acc(vx + .5*h*a2[0], vy + .5*h*a2[1], vz + .5*h*a2[2])
        a4 = acc(vx + h*a3[0], vy + h*a3[1], vz + h*a3[2])
        npx = px + h/6*(vx + 2*(vx + .5*h*a1[0]) + 2*(vx + .5*h*a2[0]) + vx + h*a3[0])
        npy = py + h/6*(vy + 2*(vy + .5*h*a1[1]) + 2*(vy + .5*h*a2[1]) + vy + h*a3[1])
        npz = pz + h/6*(vz + 2*(vz + .5*h*a1[2]) + 2*(vz + .5*h*a2[2]) + vz + h*a3[2])
        if npx >= XG:
            fr = (XG - px)/(npx - px)
            return out, (t + fr*h, py + fr*(npy - py), pz + fr*(npz - pz))
        px, py, pz = npx, npy, npz
        vx += h/6*(a1[0] + 2*a2[0] + 2*a3[0] + a4[0])
        vy += h/6*(a1[1] + 2*a2[1] + 2*a3[1] + a4[1])
        vz += h/6*(a1[2] + 2*a2[2] + 2*a3[2] + a4[2])
        t += h
    return out, (float('nan'),)*3


TIMES = (0.05, 0.10)
SIG = 0.005


def resid(th, obs):
    pos, _ = advance(th, TIMES)
    d = (obs - np.array(pos)).ravel()/SIG
    pr = (th - MU)/SD
    return np.concatenate([d, pr])


def map_fit(obs, iters=40):
    th = np.clip(np.array([30., 5., 7., 0.25, 0.002, 0.0]), LO, HI)
    r = resid(th, obs); cost = r @ r; lam = 1e-2
    for _ in range(iters):
        J = np.zeros((12, 6))
        for i in range(6):
            h = 1e-5*max(1., abs(th[i]))
            tp = th.copy(); tp[i] += h; tm = th.copy(); tm[i] -= h
            J[:, i] = (resid(tp, obs) - resid(tm, obs))/(2*h)
        A = J.T @ J; g = J.T @ r; done = False
        for _ in range(8):
            try:
                step = np.linalg.solve(A + lam*np.diag(np.diag(A) + 1e-9), -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            tn = np.clip(th + step, LO, HI); rn = resid(tn, obs); cn = rn @ rn
            if cn < cost:
                th, r, cost = tn, rn, cn; lam = max(lam*0.3, 1e-9); done = True; break
            lam *= 10
        if not done:
            break
    return th, cost


def laplace_cov(th0, obs):
    J = np.zeros((12, 6))
    for i in range(6):
        h = 1e-5*max(1., abs(th0[i]))
        tp = th0.copy(); tp[i] += h; tm = th0.copy(); tm[i] -= h
        J[:, i] = (resid(tp, obs) - resid(tm, obs))/(2*h)
    return np.linalg.inv(J.T @ J)*1.15*2.4**2/6.0


def mcmc(th0, obs, cov, n=1600, burn=400, thin=1):
    rng = np.random.default_rng(int(np.abs(th0[0])*1e6) % 10**6)
    L = np.linalg.cholesky(cov)
    def lnp(th):
        if np.any(th < LO) or np.any(th > HI):
            return -np.inf
        r = resid(th, obs)
        return -0.5*(r @ r)
    th = th0.copy(); lp = lnp(th)
    psd = np.array([0.30, 0.40, 0.30, 0.025, 0.0006, 0.0006])
    keep = []; acc = 0
    for i in range(n):
        cand = th + L @ rng.normal(0, 1, 6)
        lq = lnp(cand)
        if math.log(rng.random() + 1e-300) < lq - lp:
            th, lp = cand, lq; acc += 1
        if i >= burn and i % thin == 0:
            keep.append(th.copy())
    return np.array(keep), acc/n


def trial(rng, extra_z=0.0, mismatch=False):
    truth = np.array([30.4, 8.6, 6.9, 0.0, 0.0, 0.0])
    truth[:3] += rng.normal(0, [1.2, 1.2, 0.6])
    truth[3] = float(np.exp(rng.normal(np.log(0.35), 0.06)))
    truth[4:6] = rng.normal(0, 0.002, 2)
    truth = np.clip(truth, LO, HI)
    pos, tc = advance(truth, TIMES, extra_z=extra_z)
    if not np.isfinite(tc[1]):
        return None
    obs = np.array(pos) + rng.normal(0, SIG, (2, 3))
    th, cost = map_fit(obs)
    cov = laplace_cov(th, obs)
    chain, acc = mcmc(th, obs, cov)
    y = []; z = []
    for t in chain[::3]:
        _, c = advance(t, TIMES, dt=0.005)
        if np.isfinite(c[1]):
            y.append(c[1]); z.append(c[2])
    if len(y) < 200:
        return None
    y = np.array(y); z = np.array(z)
    yq = np.quantile(y, [.05, .95]); zq = np.quantile(z, [.05, .95])
    cy = bool(yq[0] <= tc[1] <= yq[1]); cz = bool(zq[0] <= tc[2] <= zq[1])
    inb = (np.abs(y) <= 3.55) & (z >= 0.11) & (z <= 2.33)
    return dict(cov_y=cy, cov_z=cz, cov_2d=cy and cz,
                wy=float(yq[1]-yq[0]), wz=float(zq[1]-zq[0]),
                by=float(np.median(y)-tc[1]), bz=float(np.median(z)-tc[2]),
                p=float(inb.mean()), acc=acc)


def arm(tag, n, extra_z, seed):
    rng = np.random.default_rng(seed); rows = []
    while len(rows) < n:
        r = trial(rng, extra_z)
        if r is not None:
            rows.append(r)
            if len(rows) % 10 == 0:
                print(tag, len(rows), flush=True)
    return rows


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = dict(same=arm('same', n, 0.0, 11),
               mismatch=arm('mm', max(n//2, 20), 2.0, 22))
    for k, rows in out.items():
        print(k, 'n=%d' % len(rows),
              'cov90 y %.3f z %.3f joint %.3f' % (np.mean([r['cov_y'] for r in rows]),
                                                  np.mean([r['cov_z'] for r in rows]),
                                                  np.mean([r['cov_2d'] for r in rows])),
              '| width y %.3f z %.3f' % (np.mean([r['wy'] for r in rows]), np.mean([r['wz'] for r in rows])),
              '| bias y %+.3f' % np.mean([r['by'] for r in rows]),
              '| P %.3f' % np.mean([r['p'] for r in rows]), flush=True)
    json.dump(out, open('../results/valid6.json', 'w'))
    print('written')

"""Fast scalar re-run of the six-parameter fit for the tail-sensitive probabilities.

Same model, priors, MAP and importance-sampling scheme as fit6.py, but the
propagation uses a pure-Python scalar RK4 (about one order of magnitude faster),
so the posterior can be sampled with 131 072 draws.
"""
import math
import numpy as np
import json
import fit6 as F

FACTOR = F.FACTOR
G = F.G
GOAL_X, HALF_B, H, RAD = 11.0, 3.66, 2.44, 0.110


def prop(th, dt=0.005):
    vx, vy, vz = th[0], th[1], th[2]
    cd = th[3]
    e0 = np.array([vx, vy, vz], float)
    e0 /= math.sqrt(vx*vx + vy*vy + vz*vz)
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), e0); e1 /= math.sqrt(e1 @ e1)
    e2 = np.cross(e0, e1)
    bx, by, bz = th[4]*e1 + th[5]*e2
    px, py, pz = 0.0, 0.0, 0.11
    kd = FACTOR*cd
    t = 0.0
    while px < GOAL_X and t < 0.9:
        # stage 1
        s = math.sqrt(vx*vx + vy*vy + vz*vz)
        ax = -kd*s*vx + s*(by*vz - bz*vy)
        ay = -kd*s*vy + s*(bz*vx - bx*vz)
        az = -kd*s*vz + s*(bx*vy - by*vx) - G
        vx2, vy2, vz2 = vx + 0.5*dt*ax, vy + 0.5*dt*ay, vz + 0.5*dt*az
        s = math.sqrt(vx2*vx2 + vy2*vy2 + vz2*vz2)
        ax = -kd*s*vx2 + s*(by*vz2 - bz*vy2)
        ay = -kd*s*vy2 + s*(bz*vx2 - bx*vz2)
        az = -kd*s*vz2 + s*(bx*vy2 - by*vx2) - G
        vx3, vy3, vz3 = vx + 0.5*dt*ax, vy + 0.5*dt*ay, vz + 0.5*dt*az
        s = math.sqrt(vx3*vx3 + vy3*vy3 + vz3*vz3)
        ax = -kd*s*vx3 + s*(by*vz3 - bz*vy3)
        ay = -kd*s*vy3 + s*(bz*vx3 - bx*vz3)
        az = -kd*s*vz3 + s*(bx*vy3 - by*vx3) - G
        vx4, vy4, vz4 = vx + dt*ax, vy + dt*ay, vz + dt*az
        s = math.sqrt(vx4*vx4 + vy4*vy4 + vz4*vz4)
        ax4 = -kd*s*vx4 + s*(by*vz4 - bz*vy4)
        ay4 = -kd*s*vy4 + s*(bz*vx4 - bx*vz4)
        az4 = -kd*s*vz4 + s*(bx*vy4 - by*vx4) - G
        npx = px + dt/6.0*(vx + 2*vx2 + 2*vx3 + vx4)
        npy = py + dt/6.0*(vy + 2*vy2 + 2*vy3 + vy4)
        npz = pz + dt/6.0*(vz + 2*vz2 + 2*vz3 + vz4)
        nvx = vx + dt/6.0*(ax + 2*ax + 2*ax + ax4)   # placeholder, replaced below
        # proper RK4 velocity update
        s1 = math.sqrt(vx*vx + vy*vy + vz*vz)
        k1x = -kd*s1*vx + s1*(by*vz - bz*vy); k1y = -kd*s1*vy + s1*(bz*vx - bx*vz); k1z = -kd*s1*vz + s1*(bx*vy - by*vx) - G
        s2 = math.sqrt(vx2*vx2 + vy2*vy2 + vz2*vz2)
        k2x = -kd*s2*vx2 + s2*(by*vz2 - bz*vy2); k2y = -kd*s2*vy2 + s2*(bz*vx2 - bx*vz2); k2z = -kd*s2*vz2 + s2*(bx*vy2 - by*vx2) - G
        s3 = math.sqrt(vx3*vx3 + vy3*vy3 + vz3*vz3)
        k3x = -kd*s3*vx3 + s3*(by*vz3 - bz*vy3); k3y = -kd*s3*vy3 + s3*(bz*vx3 - bx*vz3); k3z = -kd*s3*vz3 + s3*(bx*vy3 - by*vx3) - G
        nvx = vx + dt/6.0*(k1x + 2*k2x + 2*k3x + ax4)
        nvy = vy + dt/6.0*(k1y + 2*k2y + 2*k3y + ay4)
        nvz = vz + dt/6.0*(k1z + 2*k2z + 2*k3z + az4)
        if npx >= GOAL_X:
            f0, f1 = px - GOAL_X, npx - GOAL_X
            fr = f0/(f0 - f1)
            y = py + fr*(npy - py); z = pz + fr*(npz - pz)
            return t + fr*dt, y, z
        px, py, pz, vx, vy, vz = npx, npy, npz, nvx, nvy, nvz
        t += dt
    return float('nan'), float('nan'), float('nan')


def clear(y, z):
    return (abs(y) <= HALF_B - RAD) and (RAD <= z <= H - RAD)


def run(n=131072, seed=20261107, df=8):
    th, cost, J, r = F.fit_map()
    cov = np.linalg.inv(J.T@J)*1.15
    L = np.linalg.cholesky(cov); Linv = np.linalg.inv(L)
    print('MAP entry', F.cross_plane(th), flush=True)
    rng = np.random.default_rng(seed)
    keep_th, keep_r, keep_z = [], [], []
    drawn = 0
    while len(keep_th) < n:
        m = n - len(keep_th)
        Z = rng.standard_normal((m, 6))
        chi = np.sqrt(rng.chisquare(df, m)/df)
        ths = th + (Z@L.T)/chi[:, None]
        ok = np.all((ths > F.LOW) & (ths < F.HIGH), axis=1)
        for x, zz, cc in zip(ths[ok], Z[ok], chi[ok]):
            keep_th.append(x); keep_r.append(zz); keep_z.append(cc)
        drawn += m
        if len(keep_th) >= n: break
    ths = np.array(keep_th[:n]); Zk = np.array(keep_r[:n]); chik = np.array(keep_z[:n])
    rr = np.array([F.residuals(x) for x in ths])
    logpost = -0.5*np.sum(rr*rr, axis=1)
    from math import lgamma, pi, log
    d = Zk.shape[1]
    Rk = Zk/chik[:, None]          # r = L^-1(theta-map) = z/chi
    z2 = np.sum(Rk*Rk, axis=1)
    logq = (lgamma((df+d)/2) - lgamma(df/2) - 0.5*d*log(df*pi)
            - (df+d)/2*np.log1p(z2/df) - 0.5*np.linalg.slogdet(cov)[1])
    lw = logpost - logq
    w = np.exp(lw - lw.max()); w /= w.sum()
    print('ESS %.1f of %d' % (1.0/np.sum(w*w), n), flush=True)
    res = np.array([prop(x) for x in ths])
    good = np.isfinite(res[:, 0])
    ww = w[good]/w[good].sum(); r2 = res[good]
    def wq(v, q):
        o = np.argsort(v); vv = v[o]; wc = ww[o]
        c = (np.cumsum(wc) - 0.5*wc)/wc.sum()
        return float(np.interp(q, c, vv))
    out = {}
    for i, nm in enumerate(['t', 'y', 'z']):
        v = r2[:, i]; mu = float(np.sum(ww*v))
        sd = float(np.sqrt(np.sum(ww*(v - mu)**2)))
        out[nm] = dict(mean=mu, sd=sd, q90=[wq(v, .05), wq(v, .95)],
                       q95=[wq(v, .025), wq(v, .975)])
        print(nm, '%.4f +- %.4f  90%% [%.3f, %.3f]' % (mu, sd, *out[nm]['q90']), flush=True)
    clr = np.array([clear(y, z) for _, y, z in r2])
    p = float(np.sum(ww*clr)); mcse = float(np.sqrt(np.sum(ww**2*(clr - p)**2)))
    out['p_clear'] = p; out['p_clear_mcse'] = mcse; out['n'] = n
    out['ess'] = float(1.0/np.sum(w*w))
    print('P = %.5f +- %.5f' % (p, mcse), flush=True)
    json.dump(out, open('../results/fit6_fix2.json', 'w'), indent=1)
    return out


if __name__ == '__main__':
    import sys
    run(n=int(sys.argv[1]) if len(sys.argv) > 1 else 131072)

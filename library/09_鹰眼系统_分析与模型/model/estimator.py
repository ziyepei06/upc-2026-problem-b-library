"""Inverse problem: state from observations, then impact point prediction.

Three estimators, increasing sophistication:

  two_point_gravity   v = (p2 - p1) / h, then constant gravity
  three_point_accel   a from the second difference, then constant acceleration
  drag_model          v from two points, then the drag model integrated forward

Analytic (vectorised) propagation is used for Monte Carlo; the drag model is
integrated with RK4 from physics.py.  All angles/positions in SI units.
"""

from __future__ import annotations

import numpy as np

import physics as phys

GOAL_X = 11.0
GOAL_HALF_WIDTH = 7.32 / 2.0
GOAL_HEIGHT = 2.44


# --------------------------------------------------------------------- forward
def time_to_plane_gravity(p, v, x_plane=GOAL_X):
    """Time until x reaches the goal plane under gravity only (x-motion is linear)."""
    if v[0] <= 0:
        return np.inf
    return (x_plane - p[0]) / v[0]


def entry_point_gravity(p, v, x_plane=GOAL_X):
    t = time_to_plane_gravity(p, v, x_plane)
    if not np.isfinite(t):
        return np.array([np.nan, np.nan]), t
    y = p[1] + v[1] * t
    z = p[2] + v[2] * t - 0.5 * 9.81 * t * t
    return np.array([y, z]), t


def entry_point_const_accel(p, v, a, x_plane=GOAL_X):
    if v[0] <= 0:
        return np.array([np.nan, np.nan]), np.inf
    # solve 0 = x0 + vx t + ax t^2 / 2 for the smallest positive t
    roots = np.roots([0.5 * a[0], v[0], p[0] - x_plane]) if a[0] != 0 else [ (x_plane - p[0]) / v[0] ]
    cand = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    if not cand:
        return np.array([np.nan, np.nan]), np.inf
    t = min(cand)
    return np.array([p[1] + v[1] * t + 0.5 * a[1] * t * t,
                     p[2] + v[2] * t + 0.5 * a[2] * t * t]), t


def entry_point_drag(p, v, x_plane=GOAL_X, omega=None, dt=2e-3, tol=1e-4):
    """Integrate the drag (+ optional Magnus) model until x crosses the plane."""
    state = np.concatenate([np.asarray(p, float), np.asarray(v, float)])
    t = 0.0
    prev_x = state[0]
    while t < 2.0:
        nxt = phys.step_rk4(state, t, dt, omega)
        t += dt
        if nxt[0] >= x_plane:
            # linear interpolation inside the last step
            frac = (x_plane - prev_x) / max(nxt[0] - prev_x, 1e-12)
            pos = state[:3] + frac * (nxt[:3] - state[:3])
            return pos[1:], t
        prev_x = nxt[0]
        state = nxt
    return np.array([np.nan, np.nan]), np.inf


# -------------------------------------------------------------------- inverse
def velocity_from_two_point(p1, p2, h):
    return (np.asarray(p2, float) - np.asarray(p1, float)) / h


def accel_from_three_point(p0, p1, p2, h):
    p0, p1, p2 = (np.asarray(p, float) for p in (p0, p1, p2))
    return (p0 - 2 * p1 + p2) / (h * h)


def state_from_two_point(p1, p2, h):
    return np.asarray(p1, float), velocity_from_two_point(p1, p2, h)


def state_from_three_point(p0, p1, p2, h):
    return np.asarray(p1, float), velocity_from_two_point(p0, p1, h), accel_from_three_point(p0, p1, p2, h)


def covariance_two_point(sigma_pos, h):
    """Covariance of (p1, v) when both positions have independent noise sigma."""
    Cpp = np.eye(3) * sigma_pos ** 2
    Cvv = 2.0 * np.eye(3) * sigma_pos ** 2 / h ** 2
    Cpv = np.eye(3) * (-sigma_pos ** 2 / h)
    return np.block([[Cpp, Cpv], [Cpv, Cvv]])


def covariance_three_point(sigma_pos, h):
    """Covariance of v and a for the second-difference estimator."""
    Cvv = np.eye(3) * 2.0 * sigma_pos ** 2 / h ** 2
    Caa = np.eye(3) * 6.0 * sigma_pos ** 2 / h ** 4
    return Cvv, Caa


# ---------------------------------------------------------------- Monte Carlo
def monte_carlo_gravity(p0, p1, p2, sigma_pos, n=2000, seed=0, x_plane=GOAL_X,
                        free_initial=False):
    """Sample position noise, run the two-point estimator, return entry stats.

    p0 is treated as an exact initial condition (it is a rule, not a
    measurement).  With free_initial=True it is also perturbed, which is the
    three-point estimator.
    """
    rng = np.random.default_rng(seed)
    p1 = np.asarray(p1, float)[None, :] + rng.normal(0, sigma_pos, (n, 3))
    p2 = np.asarray(p2, float)[None, :] + rng.normal(0, sigma_pos, (n, 3))
    h = 0.05
    v = (p2 - p1) / h
    t = (x_plane - p1[:, 0]) / v[:, 0]
    good = (v[:, 0] > 0) & (t > 0) & (t < 1.0)
    y = p1[:, 1] + v[:, 1] * t
    z = p1[:, 2] + v[:, 2] * t - 0.5 * 9.81 * t ** 2
    return dict(t=t[good], y=y[good], z=z[good], v=v[good], n_good=good.sum(), n=n)


def monte_carlo_drag(p0, p1, p2, sigma_pos, n=200, seed=0, omega_rpm=0.0):
    """Same but propagating the drag model; slower, so default n is small."""
    rng = np.random.default_rng(seed)
    omega = np.array([0.0, 0.0, omega_rpm * phys.OMEGA_PER_RPM]) if omega_rpm else None
    ys, zs = [], []
    for _ in range(n):
        p1n = np.asarray(p1, float) + rng.normal(0, sigma_pos, 3)
        p2n = np.asarray(p2, float) + rng.normal(0, sigma_pos, 3)
        v = (p2n - p1n) / 0.05
        yz, _ = entry_point_drag(p1n, v, omega=omega)
        ys.append(yz[0])
        zs.append(yz[1])
    return np.asarray(ys), np.asarray(zs)


# ---------------------------------------------------------------- judge results
def inside_goal(y, z, margin=0.0):
    return (np.abs(y) < GOAL_HALF_WIDTH - margin) & (z > margin) & (z < GOAL_HEIGHT - margin)


def summarize(y, z, margin=0.0):
    return dict(
        sigma_y_cm=100 * np.std(y),
        sigma_z_cm=100 * np.std(z),
        p_inside=float(np.mean(inside_goal(y, z, margin))),
        p_wide=float(np.mean(np.abs(y) >= GOAL_HALF_WIDTH)),
        p_over=float(np.mean(z >= GOAL_HEIGHT)),
    )

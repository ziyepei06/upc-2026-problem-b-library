"""Football aerodynamics used by the IPP (impact point prediction) chain.

Units: m, s, kg, rad.  Spin is a vector omega in rad/s.

Model
-----
    dv/dt = -g z_hat  -  (1/2 rho A C_d |v| v) / m  +  (1/2 rho A C_l |v|^2 / m) (w_hat x v_hat)
    dp/dt = v
    C_d = C_d(Re),      Re = |v| D / nu
    C_l = 0.505 * S_p,  S_p = omega R / |v|      (Goff & Carre 2010)

The drag table is Carre's wind-tunnel data for a 32-panel football; the spin
decays by 5-15 % per 18 m of flight, which we model as a single exponential.

Everything here is a *forward* model: given initial state it integrates.  The
inverse problem (state from positions) lives in estimator.py.
"""

from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])

MASS = 0.424          # kg
DIAMETER = 0.22       # m
RADIUS = DIAMETER / 2
AREA = np.pi * RADIUS ** 2
RHO = 1.2             # kg/m^3
NU = 1.5e-5           # m^2/s  kinematic viscosity of air
OMEGA_PER_RPM = 2.0 * np.pi / 60.0

# Re vs C_d: drag crisis sits between 7e4 and 1.3e5 for a football.
_CD_TABLE = np.array([
    [2.0e4, 0.50],
    [4.0e4, 0.48],
    [6.0e4, 0.44],
    [7.0e4, 0.40],
    [8.0e4, 0.35],
    [9.0e4, 0.30],
    [1.0e5, 0.25],
    [1.3e5, 0.21],
    [2.0e5, 0.20],
    [4.0e5, 0.19],
    [1.0e6, 0.18],
])


def drag_coefficient_from_re(re: np.ndarray | float) -> np.ndarray | float:
    """Log-log interpolation of the wind-tunnel drag curve."""
    re = np.asarray(re, dtype=float)
    lr = np.log10(np.clip(re, _CD_TABLE[0, 0], _CD_TABLE[-1, 0]))
    return np.interp(lr, np.log10(_CD_TABLE[:, 0]), _CD_TABLE[:, 1])


def drag_coefficient(speed: np.ndarray | float) -> np.ndarray | float:
    speed = np.asarray(speed, dtype=float)
    return drag_coefficient_from_re(speed * DIAMETER / NU)


def spin_parameter(speed: np.ndarray | float, omega: np.ndarray | float) -> np.ndarray | float:
    """S_p = R omega / v, with omega the spin magnitude in rad/s."""
    speed = np.asarray(speed, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return RADIUS * np.abs(omega) / np.maximum(speed, 1e-6)


def lift_coefficient(speed, omega) -> np.ndarray | float:
    """Goff & Carre: C_l = 0.505 S_p in the linear range (S_p <~ 0.3)."""
    return 0.505 * spin_parameter(speed, omega)


def spin_decay(omega0, t, tau: float = 6.0):
    """Exponential spin decay; tau = 6 s reproduces 5-15 % loss per 18 m."""
    return np.asarray(omega0, dtype=float) * np.exp(-np.asarray(t, dtype=float) / tau)


def acceleration(v, omega=None, cd=None, cl=None):
    """Total acceleration: gravity + drag (+ Magnus if omega is given)."""
    v = np.asarray(v, dtype=float)
    speed = float(np.linalg.norm(v))
    cd = drag_coefficient(speed) if cd is None else cd
    a = GRAVITY - 0.5 * RHO * AREA * cd * speed * v / MASS
    if omega is not None:
        w = np.asarray(omega, dtype=float)
        wmag = float(np.linalg.norm(w))
        if wmag > 0.0:
            cl = lift_coefficient(speed, wmag) if cl is None else cl
            a = a + 0.5 * RHO * AREA * cl * speed ** 2 * np.cross(w / wmag, v / speed) / MASS
    return a


def rhs(state, t, omega0=None, tau: float = 6.0):
    """Right-hand side for integrate; omega decays with time."""
    p, v = state[:3], state[3:]
    omega = None if omega0 is None else spin_decay(omega0, t, tau)
    return np.concatenate([v, acceleration(v, omega)])


def step_rk4(state, t, dt, omega0=None, tau: float = 6.0):
    k1 = rhs(state, t, omega0, tau)
    k2 = rhs(state + 0.5 * dt * k1, t + 0.5 * dt, omega0, tau)
    k3 = rhs(state + 0.5 * dt * k2, t + 0.5 * dt, omega0, tau)
    k4 = rhs(state + dt * k3, t + dt, omega0, tau)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def propagate(p0, v0, t_end, omega0=None, dt: float = 1e-3, tau: float = 6.0):
    """Integrate forward; returns (positions, velocities, times)."""
    state = np.concatenate([np.asarray(p0, float), np.asarray(v0, float)])
    n = max(1, int(np.ceil(t_end / dt)))
    dt = t_end / n
    ts = np.linspace(0.0, t_end, n + 1)
    ps = np.empty((n + 1, 3))
    vs = np.empty((n + 1, 3))
    ps[0], vs[0] = state[:3], state[3:]
    t = 0.0
    for i in range(n):
        state = step_rk4(state, t, dt, omega0, tau)
        t += dt
        ps[i + 1], vs[i + 1] = state[:3], state[3:]
    return ps, vs, ts


def exit_velocity(p0, v0, t_end, omega0=None, dt: float = 1e-3, tau: float = 6.0):
    ps, vs, _ = propagate(p0, v0, t_end, omega0, dt, tau)
    return ps[-1], vs[-1]

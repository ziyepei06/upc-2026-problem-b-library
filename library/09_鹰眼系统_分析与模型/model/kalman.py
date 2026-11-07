"""A constant-velocity Kalman filter with gravity as a known input.

This is the "delay filter" of a tracking system in its simplest form: keep a
state [p, v], predict through the flight dynamics, update with each new
triangulated position.  It is what turns noisy per-frame triangulation into a
smooth trajectory and, more importantly, what gives the *covariance* that the
impact point prediction has to carry forward.

    x_k = F x_{k-1} + B u        u = g (known)
    F   = [[I, dt I], [0, I]]
    B   = [dt^2/2 I ; dt I]
    P   = F P F^T + Q            Q from unmodelled acceleration (drag/Magnus)
    K   = P H^T (H P H^T + R)^-1
    H   = [I, 0]                 we observe positions only
"""

from __future__ import annotations

import numpy as np

G = np.array([0.0, 0.0, -9.81])


class BallFilter:
    def __init__(self, sigma_pos=0.005, sigma_acc=1.0):
        self.sigma_pos = sigma_pos
        self.sigma_acc = sigma_acc
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e6
        self.R = np.eye(3) * sigma_pos ** 2

    # ------------------------------------------------------------------ init
    def init_two_points(self, p1, p2, h):
        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        v = (p2 - p1) / h
        self.x = np.concatenate([p1, v])
        s2 = self.sigma_pos ** 2
        Cpv = -s2 / h * np.eye(3)
        self.P = np.block([
            [s2 * np.eye(3), Cpv],
            [Cpv, 2 * s2 / h ** 2 * np.eye(3)],
        ])
        return self

    # --------------------------------------------------------------- predict
    def predict(self, dt, use_gravity=True):
        I, Z = np.eye(3), np.zeros((3, 3))
        F = np.block([[I, dt * I], [Z, I]])
        u = np.zeros(6)
        if use_gravity:
            u[:3] = 0.5 * dt * dt * G
            u[3:] = dt * G
        self.x = F @ self.x + u
        # process noise: acceleration uncertainty folded into the state
        q = self.sigma_acc ** 2
        Q = np.block([
            [0.25 * dt ** 4 * q * np.eye(3), 0.5 * dt ** 3 * q * np.eye(3)],
            [0.5 * dt ** 3 * q * np.eye(3), dt ** 2 * q * np.eye(3)],
        ])
        self.P = F @ self.P @ F.T + Q
        return self

    # ---------------------------------------------------------------- update
    def update(self, z):
        z = np.asarray(z, float)
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P
        return self

    @property
    def position(self):
        return self.x[:3]

    @property
    def velocity(self):
        return self.x[3:]

    def state_covariance(self):
        return self.P


def fuse_three_samples(p0, p1, p2, h=0.05, sigma_pos=0.005, sigma_acc=1.0,
                       p0_is_exact=True):
    """Run the filter over the three samples of the 2026 B problem."""
    f = BallFilter(sigma_pos, sigma_acc)
    # initialise at t = 0 with the two-point velocity guess, then walk forward
    p0 = np.asarray(p0, float)
    f.x = np.concatenate([p0, (np.asarray(p1, float) - p0) / h])
    s2 = sigma_pos ** 2
    f.P = np.block([
        [s2 * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), 2 * s2 / h ** 2 * np.eye(3)],
    ])
    for z in ((p1, p2) if p0_is_exact else (p0, p1, p2)):
        f.predict(h)
        f.update(z)
    return f


def demo():
    p0 = np.array([0.0, 0.0, 0.11])
    p1 = np.array([1.50, 0.45, 0.44])
    p2 = np.array([2.96, 0.86, 0.74])
    for sigma in (0.001, 0.005, 0.010):
        f = fuse_three_samples(p0, p1, p2, sigma_pos=sigma)
        C = f.state_covariance()
        print(f"sigma_pos = {1000 * sigma:4.1f} mm")
        print(f"  p = {np.round(f.position, 4)}   (at t = 0.10 s)")
        print(f"  v = {np.round(f.velocity, 3)} m/s  +- {np.round(np.sqrt(np.diag(C)[3:]), 3)}")
        print(f"  cov(vx,vy) = {C[3, 4]:.3e}")


if __name__ == "__main__":
    demo()

"""Multi-camera triangulation: the front half of Hawk-Eye.

A camera is a 3x4 matrix P = K [R | t] mapping homogeneous world points to
image points, x ~ P X.  Given the same ball in >= 2 calibrated views, the 3D
position is the least-squares solution of the homogeneous system A X = 0,
obtained from the SVD -- the textbook DLT solution.  Any optical tracking
system reduces to this once the cameras are calibrated.

In our 2026 B problem this stage is already done: the statement hands us the
triangulated positions.  The module is here so the chain is complete and so the
position covariance can be derived the same way a real system would.
"""

from __future__ import annotations

import numpy as np


def look_at(center, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera rotation for a camera at `center` looking at `target`."""
    center = np.asarray(center, float)
    forward = np.asarray(target, float) - center
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, float)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])       # rows: camera axes in world frame
    return R, center


def camera_matrix(K, R, center):
    """P = K [R | -R C] for a camera with rotation R and centre C."""
    center = np.asarray(center, float)
    t = -np.asarray(R, float) @ center
    return np.asarray(K, float) @ np.hstack([np.asarray(R, float), t.reshape(3, 1)])


def project(P, X):
    x = np.asarray(P, float) @ np.append(np.asarray(X, float), 1.0)
    return x[:2] / x[2]


def triangulate(uv_list, P_list):
    """Linear (DLT) triangulation of one point from N >= 2 views."""
    if len(uv_list) != len(P_list):
        raise ValueError("need one 2D observation per camera")
    if len(uv_list) < 2:
        raise ValueError("at least two views are required")
    A = []
    for (u, v), P in zip(uv_list, P_list):
        P = np.asarray(P, float)
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]


def reprojection_errors(X, uv_list, P_list):
    return np.array([project(P, X) - np.asarray(uv, float)
                     for uv, P in zip(uv_list, P_list)])


def triangulation_covariance(P_list, sigma_px, X, h=1e-6):
    """First-order position covariance from pixel noise, 3 unknowns.

    J is 2N x 3 (finite-difference Jacobian of the reprojection map);
    C = sigma_px^2 (J^T J)^-1.
    """
    X = np.asarray(X, float)
    cols = []
    for k in range(3):
        dX = np.zeros(3)
        dX[k] = h
        r_plus = np.concatenate([project(P, X + dX) for P in P_list])
        r_minus = np.concatenate([project(P, X - dX) for P in P_list])
        cols.append((r_plus - r_minus) / (2 * h))
    J = np.stack(cols, axis=1)
    return sigma_px ** 2 * np.linalg.inv(J.T @ J)


def demo():
    f = 2000.0
    K = np.array([[f, 0.0, 960.0], [0.0, f, 540.0], [0.0, 0.0, 1.0]])
    X_true = np.array([5.0, 0.6, 0.4])

    # two cameras on opposite halves of the pitch, both looking at the action
    centers = [np.array([0.0, -18.0, 12.0]), np.array([11.0, -14.0, 9.0])]
    target = np.array([5.0, 0.0, 0.5])
    P_list = []
    for c in centers:
        R, C = look_at(c, target)
        P_list.append(camera_matrix(K, R, C))

    uv = [project(P, X_true) for P in P_list]
    rng = np.random.default_rng(0)
    noisy = [u + rng.normal(0, 1.0, 2) for u in uv]   # 1 pixel noise
    X = triangulate(noisy, P_list)

    print("true      ", np.round(X_true, 4))
    print("recovered ", np.round(X, 4))
    print("error (mm)", round(1000 * float(np.linalg.norm(X - X_true)), 2))
    C = triangulation_covariance(P_list, 1.0, X)
    print("1-sigma (mm) per axis:", np.round(1000 * np.sqrt(np.diag(C)), 2))
    print("baseline (m):", round(float(np.linalg.norm(centers[0] - centers[1])), 2))


if __name__ == "__main__":
    demo()

"""Reproduce the numbers quoted in the paper from the two given positions.

    python3 predict.py

Prints the constant-acceleration estimate, the physics-constrained Bayesian
estimate, their equal-weight mixture and the goal-plane probabilities.
"""
import numpy as np
import bkick as B
import estimators as E

R_OBS = np.array([[1.50, 0.45, 0.44],      # ball centre at t = 0.05 s (m)
                  [2.96, 0.86, 0.74]])     # ball centre at t = 0.10 s (m)
T_OBS = np.array([0.05, 0.10])
SIGMA = 0.010                              # position noise per coordinate (m)


def outcome(y, z):
    """Split the predictive mass into five mutually exclusive outcomes.

    clean  the whole ball passes inside the frame
    wide   the centre misses outside a post
    over   the centre clears the crossbar (and is not wide)
    under  the centre passes below the goal line (and is not wide)
    clips  the centre is inside |y| < b but the ball touches post or bar
    """
    r = B.BALL_RADIUS
    b, h = B.GOAL_HALF_WIDTH, B.GOAL_HEIGHT
    wide = np.abs(y) > b
    over = (z > h) & ~wide
    under = (z < 0.0) & ~wide & ~over
    clean = (np.abs(y) <= b - r) & (z >= r) & (z <= h - r)
    clips = ~(clean | wide | over | under)
    f = lambda a: float(np.mean(a))
    return dict(clean=f(clean), wide=f(wide), over=f(over),
                under=f(under), clips=f(clips))


# ---------------------------------------------------------------- estimator E1
e1 = E.e1_predict(R_OBS, T_OBS, SIGMA, n_mc=40000, seed=0)
print("E1  a  =", np.round(e1["a"], 2), "m/s^2")
print("E1  v0 =", np.round(e1["v0"], 2), "m/s")
print("E1  entry: t = %.4f s,  y = %.3f +- %.3f m,  z = %.3f +- %.3f m"
      % (e1["t"], e1["mean"]["y"], e1["std"]["y"],
         e1["mean"]["z"], e1["std"]["z"]))
print("    delta-method sigma: y = %.3f m, z = %.3f m"
      % (e1["std_analytic"]["y"], e1["std_analytic"]["z"]))
print("E1  P(clean goal) = %.3f" % e1["p_inside"])

# ---------------------------------------------------------------- estimator E2
post = E.PhysicalPosterior(R_OBS, T_OBS, SIGMA)
th, _ = post.map()
print("E2  MAP: v0 =", np.round(th[:3], 2), "m/s,  C_D = %.3f," % np.exp(th[3]),
      " |omega| = %.1f rad/s" % np.linalg.norm(th[4:]))
out = post.sample(n=40000, burn=8000, thin=2, seed=1)
t2, y2, z2 = post.predict(out["chain"])
s2 = E.summarize(t2, y2, z2)
print("E2  entry: y = %.3f +- %.3f m,  z = %.3f +- %.3f m,  P = %.3f"
      % (s2["mean"]["y"], s2["std"]["y"], s2["mean"]["z"], s2["std"]["z"],
         s2["p_inside"]))
print("    acceptance = %.2f,  C_D = %.3f +- %.3f"
      % (out["acc"], np.exp(out["chain"][:, 3]).mean(),
         np.exp(out["chain"][:, 3]).std()))

# ---------------------------------------------------------------- mixture
m = min(len(e1["_y"]), len(y2))          # equal weights need equal counts
y = np.concatenate([e1["_y"][:m], y2[:m]])
z = np.concatenate([e1["_z"][:m], z2[:m]])
oc = outcome(y, z)
print("MIX entry: y = %.3f +- %.3f m,  z = %.3f +- %.3f m"
      % (y.mean(), y.std(), z.mean(), z.std()))
print("MIX P(clean goal) = %.3f,  wide = %.3f, over = %.3f, "
      "under = %.3f, clips = %.3f"
      % (oc["clean"], oc["wide"], oc["over"], oc["under"], oc["clips"]))

"""Strict audit of every number the paper quotes.

Two layers:
  A. independent recomputation from the raw result files (npz / json),
  B. internal consistency checks (sums, orderings, scalings).
Every line prints PASS/FAIL with the recomputed value, so a mismatch is visible
without trusting any single downstream file.
"""
import json
import numpy as np

import paths

GPT = paths.GPT
B, H, R = 3.66, 2.44, 0.110
rows = []


def check(name, ok, detail=""):
    rows.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + ("   " + detail if detail else ""))


def wq(v, w, q):
    o = np.argsort(v)
    vv, ww = v[o], w[o]
    c = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
    return float(np.interp(q, c, vv))


nom = np.load(GPT + "/geometry_nominal.npz")
ok = np.isfinite(nom["center_yz"]).all(axis=1) & np.isfinite(nom["center_time"])
w = nom["w"][ok]
w = w / w.sum()
yz = nom["center_yz"][ok]
t = nom["center_time"][ok]
fc = nom["full_clear"][ok].astype(float)
my = float(np.sum(w * yz[:, 0])); mz = float(np.sum(w * yz[:, 1]))
sy = float(np.sqrt(np.sum(w * (yz[:, 0] - my) ** 2)))
sz = float(np.sqrt(np.sum(w * (yz[:, 1] - mz) ** 2)))
mt = float(np.sum(w * t)); st = float(np.sqrt(np.sum(w * (t - mt) ** 2)))
p = float(np.sum(w * fc))
check("nominal mean y = 2.785", abs(my - 2.78505572733303) < 5e-4, f"{my:.6f}")
check("nominal mean z = 1.945", abs(mz - 1.9446659759959097) < 5e-4, f"{mz:.6f}")
check("nominal sd y = 0.2027", abs(sy - 0.20265896874018124) < 5e-4, f"{sy:.5f}")
check("nominal sd z = 0.2002", abs(sz - 0.20017716761576698) < 5e-4, f"{sz:.5f}")
check("nominal t mean = 0.3993", abs(mt - 0.399302238090811) < 5e-4, f"{mt:.6f}")
check("nominal t sd = 0.00754", abs(st - 0.00754456) < 5e-4, f"{st:.6f}")
check("P(clearance) = 0.9708", abs(p - 0.9707814411977606) < 2e-4, f"{p:.5f}")
# hand recomputation of the clearance criterion from y,z alone
crude = (np.abs(yz[:, 0]) <= B - R) & (yz[:, 1] >= R) & (yz[:, 1] <= H - R)
pc = float(np.sum(w * crude))
check("disc criterion reproduces the stored flag", abs(pc - p) < 2e-3, f"{pc:.5f}")
q90y = (wq(yz[:, 0], w, .05), wq(yz[:, 0], w, .95))
q90z = (wq(yz[:, 1], w, .05), wq(yz[:, 1], w, .95))
check("90% y [2.45, 3.12]", abs(q90y[0] - 2.45) < 5e-3 and abs(q90y[1] - 3.12) < 5e-3,
      f"[{q90y[0]:.3f}, {q90y[1]:.3f}]")
check("90% z [1.62, 2.27]", abs(q90z[0] - 1.62) < 5e-3 and abs(q90z[1] - 2.27) < 5e-3,
      f"[{q90z[0]:.3f}, {q90z[1]:.3f}]")
q95y = (wq(yz[:, 0], w, .025), wq(yz[:, 0], w, .975))
q95z = (wq(yz[:, 1], w, .025), wq(yz[:, 1], w, .975))
check("95% y [2.394, 3.187]", abs(q95y[0] - 2.394) < 5e-3 and abs(q95y[1] - 3.187) < 5e-3,
      f"[{q95y[0]:.3f}, {q95y[1]:.3f}]")
check("95% z [1.555, 2.341]", abs(q95z[0] - 1.555) < 5e-3 and abs(q95z[1] - 2.341) < 5e-3,
      f"[{q95z[0]:.3f}, {q95z[1]:.3f}]")
check("order: q05 < q50 < q95 (y,z)",
      wq(yz[:, 0], w, .05) < wq(yz[:, 0], w, .5) < wq(yz[:, 0], w, .95)
      and wq(yz[:, 1], w, .05) < wq(yz[:, 1], w, .5) < wq(yz[:, 1], w, .95))

scen = json.load(open(GPT + "/scenario_predictions.json"))
single = [s for s in scen if s["group"] == "single_factor"]
joint = [s for s in scen if s["group"] == "combined_stress"]
ps = np.array([s["summary"]["full_sphere_clearance_probability"] for s in single])
ys = np.array([s["summary"]["y"]["mean"] for s in single])
zs = np.array([s["summary"]["z"]["mean"] for s in single])
lo95 = np.array([s["summary"]["y"]["q025_q50_q975"][0] for s in single])
hi95 = np.array([s["summary"]["y"]["q025_q50_q975"][2] for s in single])
check("26 single-factor + 4 joint-stress", len(single) == 26 and len(joint) == 4,
      f"{len(single)}+{len(joint)}")
check("scenario P range 0.731-0.9997", abs(ps.min() - 0.7306) < 2e-3 and ps.max() > 0.9995,
      f"{ps.min():.4f}-{ps.max():.4f}")
check("single-factor y mean 2.687-3.130", abs(ys.min() - 2.687) < 5e-3 and abs(ys.max() - 3.130) < 5e-3,
      f"{ys.min():.3f}-{ys.max():.3f}")
pjs = np.array([s["summary"]["full_sphere_clearance_probability"] for s in joint])
yjs = np.array([s["summary"]["y"]["mean"] for s in joint])
check("joint-stress P 0.561-0.987", abs(pjs.min() - 0.561) < 3e-3 and abs(pjs.max() - 0.987) < 3e-3,
      f"{pjs.min():.3f}-{pjs.max():.3f}")
check("joint-stress y mean 2.733-3.339", abs(yjs.min() - 2.733) < 5e-3 and abs(yjs.max() - 3.339) < 5e-3,
      f"{yjs.min():.3f}-{yjs.max():.3f}")
check("scenario z mean 1.846-2.042", abs(zs.min() - 1.846) < 5e-3 and abs(zs.max() - 2.042) < 5e-3,
      f"{zs.min():.3f}-{zs.max():.3f}")
env_y = (np.percentile(lo95, 0), np.percentile(hi95, 100))
check("95% envelope y [2.259, 3.935]", abs(lo95.min() - 2.259) < 5e-3 and abs(hi95.max() - 3.935) < 5e-3,
      f"[{lo95.min():.3f}, {hi95.max():.3f}]")

# DS numbers: calibration and validation, if the files are present
import glob, os
def maybe(path, fn):
    if os.path.exists(path):
        fn(json.load(open(path)))
    else:
        check("missing " + os.path.basename(path), False)

# our own control results
v = json.load(open("../results/valid.json"))
for arm_name in ("physics", "kinematic"):
    tr = v[arm_name]["trials"]; sc = v[arm_name]["score"]
    e = np.array([r["e2_y"] - r["y_true"] for r in tr])
    sig = np.array([r["e2_sy"] for r in tr])
    cov68 = float(np.mean(np.abs(e) <= sig))
    lam = float(np.percentile(np.abs(e / sig), 90) / 1.645)
    check(f"E2 {arm_name}: bias_y = {e.mean():+.3f}", abs(e.mean() - sc["e2"]["bias_y"]) < 1e-9,
          f"{e.mean():+.4f} vs stored {sc['e2']['bias_y']:+.4f}")
    check(f"E2 {arm_name}: 68% coverage {cov68:.3f} vs stored {sc['e2']['cov']['y68']:.3f}",
          abs(cov68 - sc["e2"]["cov"]["y68"]) < 1e-9)
    check(f"E2 {arm_name}: lambda_y {lam:.3f}", (abs(lam - 1.66) < 0.12) if arm_name == "kinematic" else (abs(lam - 0.85) < 0.12))

print("\n%d checks, %d failed" % (len(rows), sum(1 for r in rows if not r[1])))

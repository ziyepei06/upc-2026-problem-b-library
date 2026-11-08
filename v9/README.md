# v9 — merged version (paper + statistical additions)

`paper/230B_v9.pdf` — 36 pages, 15 figures, 10 tables, 26 references. Set in 11 pt with 0.86 in margins to keep the length near the gold-paper median.
Build: `cd paper && xelatex main && bibtex main && xelatex main && xelatex main`.

## What v9 is

v9 = the submitted paper (40 pp, six-parameter transverse-force model, M0–M6
ladder, observation-design study, 24 references) **plus** the parts of the
statistical line (v8) that are strictly better, merged in place:

| addition | where in v9 |
|---|---|
| observation covariance with velocity-dependent and time-stamp terms, `C_i = σ_pos² I + σ_t² v vᵀ` (+ common-mode term) | §5.1, Eq. (6) |
| four-arm validation (common-mode offset, time-stamp error, force mismatch, low-launch) with Brier scores | §6.11, Table 12 |
| marginal intervals ≠ joint region, with the 0.81–0.90 factor and crossing-time Lagrange weights | Appendix F |
| exact whole-sphere clearance test on the sampled segment (analytic quadratic minimum) | Appendix G |
| one clause in the summary stating that the answer survives the error-model changes | Abstract |
| **new §6.8 "Drag from real trajectories": the drag coefficient measured on 350 one-second windows of two real 2-D tracking datasets, with a synthetic control that fixes the ±0.02 noise floor; the two fixes demand a deceleration above the 95th percentile of real flights | §6.8, Fig. 15, Table — |

## What is unchanged

The answer: `y = 2.785 m, z = 1.945 m, t = 0.399 s`, marginal 90 % intervals
`[2.455, 3.121] m` and `[1.617, 2.276] m`, whole-sphere clearance
`P = 0.971 ± 0.001`; the model ladder M0–M6; the design study (211 mm → 12 mm);
the three external checks against measured data.

## Provenance

* `paper/` — LaTeX sources of v9 (same layout as the submitted paper).
* `code/fit6_fix2.py` — the corrected independent sampler (see `../paper/230B.pdf`
  appendix D and the changelog); `fit6.json` in the submitted package was
  produced by a version whose importance weights were biased, which narrowed the
  posterior by ~8 %; the corrected run gives `2.7845 ± 0.2043 m` and
  `P = 0.9715`, i.e. the quoted numbers.
* `code/` — figure scripts, design/Fisher study, validation runs, number audit.
* `results/` — the JSON files behind the quoted tables.

## Version history

* `paper/230B.pdf` — submitted 2026-11-07 (40 pp).
* `v9/paper/230B_v9.pdf` — this merged version (43 pp).
* `../paper_rewrite_v8/` (outside this repo) — the 11 pp statistical rewrite that
  contributed the observation model, Appendix F/G and Table 12.

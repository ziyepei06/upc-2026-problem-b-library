# UPC 2026 Problem B — Source Library

Reference library assembled while solving **Problem B of the 2026 University Physics
Competition** (Team 230): *predicting the goal-plane entry point of a football penalty
kick from two early ball positions, with an uncertainty estimate.*

This repository collects the primary sources, the commercial-systems documentation, the
public tracking datasets and the methodological literature that the paper's numbers were
taken from — plus the submitted paper itself. It is published so that every quoted number
can be traced back to a document, and so that a later project (a journal version, or a
different ball-sport problem) can start from the same shelf.

- Problem statement: `library/00_赛题原文/2026-B-题目原文.txt`
- Submitted paper: `paper/230B.pdf` (source archive `paper/230B_打包.zip`)
- Full file list with sizes and SHA-256 prefixes: `MANIFEST.csv`
- Licensing and copyright: `LICENSE-NOTICE.md` — **read this before redistributing**

---

## Versions of the paper

| file | version | pages | what it is |
|---|---|---|---|
| `paper/230B.pdf` | submitted | 40 | the paper as submitted on 2026-11-07 |
| `v9/paper/230B_v9.pdf` | merged (v9) | 45 | submitted paper + observation-covariance model, four-arm validation, marginal-vs-joint and whole-sphere appendices |
| `v9/README.md` | --- | --- | what changed and where, plus provenance of every added number |

The 11-page statistical rewrite that contributed the observation model is not
included here; its additions are documented in `v9/README.md`.

## 1. Layout

| Folder | Contents | Role in the paper |
|---|---|---|
| `00_赛题原文` | 2026 A/B problem text and page snapshot; 2022 B problem text | Defines the geometry, the two fixes and the scenario |
| `01_同题金奖论文_2022B` | 3 Gold papers for the same problem (2022) | Benchmark for scope and length; audited for what they *do not* do |
| `02_同结构金奖论文_2025B_火炮弹道` | 13 Gold papers on projectile-ballistics structure | Structure and validation conventions |
| `03_其他球类飞行金奖论文` | Basketball spin, table-tennis ball size, golf Magnus | Cross-sport checks of the force model |
| `04_足球空气动力学` | Mehta 1985; Carré 2005; Goff & Carré 2012; JSME 2014; Tsukada 2008; Asai 2020; Nathan 2008 | $C_D(Re)$, $C_L(S_p)$, spin decay, flight time |
| `05_轨迹重建与跟踪` | Monocular/multi-camera 3-D ball reconstruction and prediction (2024–2026) | Motivation: what tracking cannot recover |
| `06_商业系统_鹰眼` | FIFA GLT test manual (2020), FIFA EPTS manual (2025), Hawk-Eye documents | Accuracy and sampling-rate anchors (±1.5 cm, 10/25 Hz) |
| `07_预测与不确定性方法` | Sloppy models (Gutenkunst 2007, White 2017); re-entry and missile trajectory UQ | Framing of model discrepancy and interval reporting |
| `08_数据集` | ISSIA-3D, IDSSE, Metrica, SoccerNet-v3D (see §3) | Real measurement noise; real 3-D reconstruction residuals |
| `09_鹰眼系统_分析与模型` | Index notes on how the commercial system works | Context |
| `10_中文文献` | Five Chinese-language papers on curved-ball flight | Cross-language sanity checks |
| `11_官方规则与规格` | IFAB Laws of the Game 2025–26 | Goal 7.32 × 2.44 m, penalty mark 11 m, ball 68–70 cm / 410–450 g |
| `12_点球与门将行为` | Keeper anticipation, penalty placement distributions, save strategies | Timing budget and empirical placement |
| `13_反演与估计方法` | Spin from 3-D tracks, PITCHf/x, landing-point prediction (badminton 2023), grey-box EKF with unidentifiable spin (table tennis 2023), monocular football (2025/2026) | Closest prior art for the inverse problem |
| `14_机器学习与最小二乘` | KalmanNet, PINN, differentiable optimisation, amortised inference | Why a physics-constrained least-squares fit was preferred |

## 2. Numbers the paper takes from this library

| Quantity | Value used | Source file | Page |
|---|---|---|---|
| Drag crisis of a football | $C_D$ 0.50 → 0.20 over $Re=9\times10^4$–$1.3\times10^5$ | `04/Carre2005` | 2, 6–7 |
| Drag at match speed | $C_D=0.22$ at $Re=2.7\times10^5$ and $3.9\times10^5$ | `04/Carre2005` | 11–12 |
| Drag from trajectory fits | $C_D\approx0.17$ at $Re=2.2$–$2.8\times10^5$ (plateau 0.17–0.22) | `04/GoffCarre2012` | 5 |
| Lift coefficient | $C_L$ = 0.21 / 0.24 / 0.29 at $S_p$ = 0.18 / 0.22 / 0.34 | `04/Asai2020` | 6 |
| Spin decay | 5–15 % per 18 m, axis unchanged | `04/Tsukada2008` | 1 |
| Typical spin | 2.3–10.1 rev/s (power kick 1.9, curled 10.2) | `04/Tsukada2008`, `04/Carre2005` | 1; 11–12 |
| Penalty flight time | 525 ± 64 ms (N = 88) | `12/FrontPsychol-2020` | 7 |
| Placement distribution | 31.3 % low, 36.0 % middle, 32.7 % high (N = 638) | `12/FrontPsychol-2020` | 7 |
| Keeper side decision | fixed ≈115 ms **before** contact; height resolved ≈245 ms **after** | `12/FrontPsychol-2020` | 7, 10 |
| Goal-line technology accuracy | ±1.5 cm, goal signal within 1 s | `06/FIFA-GLT-test-manual-2020` | 16–17 |
| Tracking certification | 10/25 Hz resampled to 50 Hz; ball certified in 2-D only | `06/FIFA-EPTS-manual-2025` | 6, 8–9, 26 |
| Theory of sloppy models | Fisher spectrum spanning >10⁶, predictive widths narrow | `07/Gutenkunst2007` | — |
| Model discrepancy | a fitted model can be *less* predictive | `07/White2017` | — |
| Spin unidentifiable from positions | table-tennis grey-box study | `13/2023-arXiv-乒乓灰箱EKF自旋不可辨识` | 3 |

## 3. Datasets (`library/08_数据集`)

| File | Rows | Content | Use |
|---|---|---|---|
| `IDSSE-J03WPY-ball.csv` | 146 211 | 25 Hz ball position in metres (`frame,time,x,y,speed,accel`), 2-D | Per-axis residual 0.056 / 0.054 m → real tracking noise |
| `IDSSE-J03WPY-shots.csv` | 26 | Peak speed (9.6–19.6 m/s) and distance to goal (11–14.6 m) | Sanity range for shot speed |
| `IDSSE-J03WPY-events.xml` | — | Event annotations | Context |
| `Metrica-第1场-球轨迹.csv` | 145 006 | 25 Hz, normalised pitch coordinates (`period,frame,time_s,ball_x,ball_y`) → multiply by 105 m × 68 m | Second noise estimate: 0.065 / 0.064 m |
| `ISSIA-3D.csv` | 2 465 | Six camera views plus a per-frame `ball_3D` and per-camera reprojection errors | Genuine 3-D positions, in camera coordinates; ground-plane play |
| `SoccerNet-v3D.csv` | 4 051 | Annotated 3-D ball point per action with camera calibration | One point per action, not a trajectory |
| `SoccerNet-v3D-parsed.csv` | 4 051 | Same, flattened: `x,y,z`, `rep_error`, `optimized_error` | Optimised reconstruction residual, median 0.050 m (p10 0.007, p90 0.27) vs 2.85 m raw reprojection |

**What this library does not contain** — and why the paper's end-to-end test is synthetic:
no public dataset combines (i) a penalty or free kick, (ii) a continuous free-flight 3-D
trajectory, (iii) a measured spin rate, and (iv) a labelled goal-plane crossing. The
measurement half of the problem is therefore calibrated on real data (noise, reconstruction
residual), while the estimator half is tested on synthetic truths generated from a broader
model than the fit.

## 4. Using it

```bash
python3 - <<'EOF'
import pandas as pd
b = pd.read_csv("library/08_数据集/IDSSE-J03WPY-ball.csv", parse_dates=["time"])
print(b[["x", "y"]].diff().abs().median())      # typical frame-to-frame step
EOF
```

PDFs are text-extractable except `04/Mehta1985` (OCR scan) and `02/2025-GM-B-460`
(failed extraction); read those as images.

## 5. Provenance

Collected 2026-11-04 to 2026-11-07. Gold-medal papers (`01`, `02`, `03`) are the
competition's own published archive. Journal articles and standards documents were
downloaded from publisher/issuer pages; the DOIs are listed in `LICENSE-NOTICE.md`.
`MANIFEST.csv` records size and a SHA-256 prefix for every file, so a copy can be verified.

## 6. 中文说明

本仓库是解 2026 年 UPC B 题（点球落点预测）时整理的资料库：76 篇文献/标准 PDF、7 个公开
跟踪数据集、以及最终论文与源码包。论文里每一个引用的数字都能在 `§2` 的对照表中追溯回具体
文件和页码；数据集在 `§3` 说明了列含义与实际用途（真实跟踪噪声 5–6 cm、真实三维重建残差
中位数 5 cm）。第三方 PDF 的版权归原作者/出版方，本仓库仅作个人研究存档，**请勿公开再分发**，
详见 `LICENSE-NOTICE.md`。

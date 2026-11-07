# 03 半自动越位（SAOT）与 EPTS

这一块和 2026 B 题的关系比门线技术更近：**它处理的是「球在运动、人在运动、如何用少量观测还原瞬时几何」**，而且它明确承认自己是「半自动」——测量自动化、判断留给人。

## SAOT 的构成

2022 年卡塔尔世界杯版本（FIFA 公开材料）：

- **12 台专用跟踪相机**，吊装在球场顶棚下；
- 每名球员跟踪 **29 个骨骼点**，**50 Hz**；
- 比赛用球 Adidas Al Rihla 内嵌 **500 Hz 惯性测量单元**（IMU），用于定位触球瞬间；
- 系统给出越位线建议，VAR 复核后由裁判最终判定 → 所以叫「半自动」。

2026 世界杯版本（媒体报道 + FIFA 材料）：

- 光学部分升级，报道称为 **16 台**高分辨率相机；
- 比赛用球 Adidas Trionda，500 Hz 传感器芯片，实时把球的数据送给 VAR；
- 引入**球员专属三维替身（digital twin）**，用扫描得到的身体比例修正骨骼点；
- 清晰的越位情形可以向场上裁判音频系统直接告警。

UEFA Euro 2024 的版本用了 **10 台专用相机**、同样 29 个身体点，并接入联网球（connected ball）确定触球点。

### 工程流水线（可直接借鉴的骨架）

```
标定相机 → 2D 检测（球 + 球员 + 关键点）→ 跨视角关联 → 三角测量成 3D
   → 用球内 IMU 的事件时间戳做触球时刻同步（必要时在帧间插值）
   → 计算越位线 → 比较攻击方合法部位 → 生成证据交给 VAR → 人做最终裁决
```

三点值得抄：

1. **时间戳比空间精度更关键**。球内 IMU 的作用不是测位置，而是把「触球瞬间」定到毫秒级；几何再准，时刻错了也白算。换成我们的题：`t = 0.05 s` 与 `t = 0.10 s` 这两个时刻本身如果有误差，会直接变成速度误差。
2. **在帧间插值**。系统明确使用「相邻跟踪样本插值」来处理触球落在两帧之间的情况。
3. **输出证据而不是断言**。半自动的「半」字是刻意保留的：它给出可复核的 3D 证据链，判断权留给裁判。我们的论文同样应该给「方法 + 不确定度 + 适用边界」，而不是一个精确到小数点后三位的落点。

## EPTS：FIFA 对跟踪系统的认证

FIFA Quality Programme for Electronic Performance & Tracking Systems 管球员与球的跟踪精度。测试手册 2025 版要点：

- **真值系统**：VICON 三维动作捕捉（30×30 m 区域，足球专项动作：速度、位置、加速度、技术动作）+ IMU（VICON Blue Trident，用于负荷指标）+ **全站仪（Total Station）**测场地关键点。
- **球的定位精度**：用全站仪测定的 10–20 个已知场地位置，测试者把球带到每个点、静置 3 秒，比较系统输出的 **二维 XY** 位置。
- **误差度量**：RMSD（与 VICON 的差值的均方根），按速度区间分档排名，公开测试报告。
- **认证有效期**：FIFA Quality 认证两年。
- 覆盖系统类型：GPS/GNSS、LPS、光学跟踪（OTS）、LiDAR、RTK。

注意：EPTS 的**球定位精度只测 XY**（水平面）。这与我们在数据侧遇到的现实一致——公开数据里 z（高度）几乎不存在真值。这也从侧面说明：工业界对球的垂直精度是单独处理的（用物理模型而非测量直接给出）。

## 对我们的直接用处

1. **误差度量用 RMSD + 分档而不是单一数字**：我们的不确定度表可以按「初速档 / 自旋档 / 观测噪声档」给出 RMSD 与最坏值，和 FIFA 报告同构，评委一看就懂。
2. **承认 z 方向没有测量真值**：拿 EPTS 只测 XY 当依据，说明「z 靠模型」是行业惯例而非我们的偷懒。
3. **触球时刻/采样时刻的精度优先**：题面给的两个时刻若有 ±10 ms 误差，按 30 m/s 算就是 ±0.3 m 的位置误差，足以改变「进/不进」的结论——这必须出现在敏感性分析里。

## 来源

- Wikipedia, *Semi-automated offside technology*: <https://en.wikipedia.org/wiki/Semi-automated_offside_technology>
- LearnOpenCV, *World Cup 2026 Offside Technology: AI, Computer Vision, and the Connected Ball*: <https://learnopencv.com/world-cup-2026-offside-technology>
- FIFA, *EPTS Testing Process*: <https://www.fifa.com/technical/football-technology/standards/epts/epts-testing-process>
- `papers/fifa-epts-test-manual-2025.pdf`、`papers/fifa-epts-how-to-obtain-certification.pdf`
- Hawk-Eye 官方 *Officiating*（SkeleTRACK 29 点、32,000+ 场；SAOT 900+ 场）

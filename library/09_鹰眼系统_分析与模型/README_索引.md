# Hawk-Eye（鹰眼）资料包

整理时间：2026-11-07。为 2026 UPC B 题（Predicting FIFA Penalty Trajectories）准备：题目要求在只给两帧观测的条件下预测球门落点并给出不确定度，这与 Hawk-Eye 这类「球轨迹测量系统」的工程实践是同一个问题——只是它的观测密度高几个数量级。看完这包资料，你应该能回答三个问题：

1. 商业系统凭什么敢说「毫米级」，它们的误差预算在哪里？
2. 只有两帧的时候，哪些误差项是 Hawk-Eye 用硬件消灭、而我们只能用假设替代的？
3. 我们的不确定度报告应该怎么写，才和工业界的验收口径对得上？

## 目录

```
hawkeye/
├── README.md                       本文件
├── notes/
│   ├── 01-原理与工作流.md            三角测量、标定、延迟滤波、输出频率
│   ├── 02-足球门线技术GLT.md         IFAB 四条要求、FIFA 测试手册、7/14 相机、±1.5 cm 与 <1 mm
│   ├── 03-半自动越位与EPTS.md        29 个关节点、50 Hz、500 Hz 球内 IMU、EPTS 测试方法（VICON 真值）
│   ├── 04-网球与旋转估计.md          10 相机 25 Hz、3.6 mm 说法、旋转反演的实测偏差
│   ├── 05-争议与局限.md              第二落点、阴影漏检、红土、板球 DRS 的独立测试
│   ├── 06-对2026B题的启示.md         两帧可辨识性、误差放大公式、验收指标怎么定
│   ├── 07-计算推理模型.md           三角测量→卡尔曼→飞行模型→落点；公式 + 实测数字 + 代码位置
│   └── 08-公开文献与专利边界.md     Hawk-Eye 没发表模型；专利/验收手册/独立验证各公开了什么
├── model/                          可直接运行的参考实现（纯 numpy）
│   ├── README.md                   模型链说明与输出样例
│   ├── triangulate.py              DLT 三角测量 + 位置协方差
│   ├── kalman.py                   常速度 KF（重力作为已知输入）
│   ├── physics.py                  Cd(Re)/Cl(Sp)/自旋衰减 + RK4
│   ├── estimator.py                两帧/三帧估计器、落点求解、蒙特卡洛
│   └── demo_2026b.py               2026 B 题端到端数值
├── papers/
│   ├── GLT_HowItWorks3.pdf                Hawk-Eye 官方 GLT 说明（7 相机/球门、2 of 7 冗余、mm 精度）
│   ├── Hawk-Eye's_GLT_How_it_Works.pdf    同上的另一版（2015，含监控手表信号链路）
│   ├── fifa-glt-testing-manual-2020.pdf   FIFA GLT 测试手册 2020 版（±1.5 cm、~55 次射门、目标墙、滑块测试）
│   ├── fifa-epts-test-manual-2025.pdf     FIFA EPTS 测试手册 2025 版（VICON 真值、RMSD 排名、球的 XY 精度）
│   ├── fifa-epts-how-to-obtain-certification.pdf  EPTS 认证流程说明
│   └── arxiv-2511.04126-tennis-tracking.pdf       ArXiv 学生级复现管线（YOLOv8/YOLOv5/ResNet50），可作开源参考
└── （打包产物在上一级）../hawk-eye-materials.zip
```

## 一句话结论

- **Hawk-Eye 的高精度来自「多相机 + 高帧率 + 标定 + 物理模型 + 延迟滤波」的整套系统**，没有任何单项是可以被两帧观测替代的。
- 足球门线系统用每球门 7 台（现代方案全场 14 台）高速相机、500 fps；即使只有 2 台看到球也能定位，这是冗余设计而不是精度来源。
- 官方验收口径：FIFA GLT 要求误差 ≤ ±1.5 cm（2014 年起；2012 年版是 ±3 cm），并且要在 1 秒内把信号送到裁判手表；Hawk-Eye 自述安装测试中「<1 mm」。
- 网球侧最常被引用的数字是 3.6 mm 平均误差（球直径 67 mm，约 5%）；独立论文测得的落点平均误差是 2.6 mm。
- **旋转没有直接测量**：Hawk-Eye 的旋转是由轨迹反推的，独立验证发现其系统性偏差明显（见 `notes/04`）；用轨迹模型反演反而更准——这一点对我们「用两帧反演 7–8 个未知量」的可行性判断非常关键。

## 怎么引用

论文里引用时优先用**可核查的一手文件**，不要引二手博客：

- 规则与验收：`papers/fifa-glt-testing-manual-2020.pdf`（FIFA 知识产权，公开 PDF），以及 IFAB 的四条要求（转引于该手册第 2 节）。
- 厂商声称：Hawk-Eye 官方页面与官方 PDF（要标明「厂商自述」）。
- 独立验证：Cant, Kovalchik, Cross & Reid (2019), *Validation of ball spin estimates in tennis from multi-camera tracking data*, J. Sports Sci. 37(22) — DOI 10.1080/02640414.2019.1697189（旋转偏差数据出自该文）。
- 半自动越位：FIFA 公开材料 + Hawk-Eye 官方 SkeleTRACK/SAOT 页面（29 点、50 Hz、900+ 场）。

模型实现与全部数值结果见 `notes/07-计算推理模型.md` 与 `model/README.md`；改动数据只需修改 `model/demo_2026b.py` 顶部的 `P0/P1/P2`。

各条目的原始链接见 `notes/` 内文与各文件末尾的「来源」小节。

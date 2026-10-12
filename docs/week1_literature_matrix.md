# 第 1 周：已有评估做了什么、缺什么

目标：一页对比表，确认本课题的差异点（出处分解 + 先验汇率 + 稳健校准）确实没被已有工作覆盖。如果第 1 周发现已有工作覆盖了其中两项以上，就回到启动方案重新定位。

> **注意**：启动方案里的"EGUsphere 2026-822 等降尺度评估工作"与 arXiv:2512.05377 是**同一篇论文**（Sun et al., 中国区域 3 km，已发表于 GMD 2026），它是模型论文，不是评估方法论文。评估类文献需要另找，下面的候选列表是起点。

## 对比表

列含义：✓ 做了 / ✗ 没做 / ~ 部分做了 / ? 待读

| 工作 | 模型 | 区域 / 分辨率 | 变量 | 真值来源 | 独立站点 | 按尺度（谱） | 集合校准（秩直方图 / SSR） | 区分输入约束与生成细节 | 极端事件分层 | 稳健 / 离群评分 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mardani et al., CorrDiff（arXiv:2309.15214） | 回归 UNet + 残差扩散 | 台湾，25 km → 2 km | t2m、u10、v10、雷达反射率 | CWA RWRF（WRF 同化） | ? | ? | ? | ? | ? | ? | 本课题审计的对象；查原文个例是否包含 2021 年台风（如璨树） |
| Sun et al., 中国 3 km CorrDiff（arXiv:2512.05377 = EGUsphere-2026-822，GMD 2026） | CorrDiff | 中国，25 km → 3 km（1600×2400） | 地面 + 6 个气压层 | CMA-RRA | ? | ? | ? | ? | ? | ? | 以 MAE 对比 CMA-MESO；代码在 Zenodo |
| Stochastic Flow Matching（arXiv:2410.19814） | 流匹配 | 台湾（CWA 数据集） | 同 CorrDiff | CWA RWRF | ? | ? | ? | ? | ? | ? | 与 CorrDiff 用同一数据，可作为第二个审计对象 |
| RainShift（地理泛化基准，PMC12852169） | 多种深度学习方法 | 多区域 | 降水 | ? | ? | ? | ? | ? | ? | ? | 评估基准类 |
| （待补）生成式降尺度的谱 / 幻觉评估 | | | | | | | | | | | 检索词：hallucination, spectral, generative downscaling evaluation |
| （待补）集合校准评估 | | | | | | | | | | | 检索词：spread-skill, rank histogram, diffusion downscaling |
| （待补）对站点的独立验证 | | | | | | | | | | | 检索词：station verification, km-scale, diffusion |
| （待补）分辨率 / 平均核思路用于机器学习降尺度 | | | | | | | | | | | 检索词：averaging kernel, degrees of freedom for signal, super-resolution |

## 检索建议

- 来源：arXiv（physics.ao-ph、cs.LG）、GMD、AIES、JAMES、MWR、QJRMS，以及 EGUsphere 讨论稿（审稿意见里常有评估方法的争论）
- 关键词组合：{generative, diffusion, GAN, flow matching} × {downscaling, super-resolution} × {evaluation, verification, calibration, spread-skill, hallucination, spectra, extremes, typhoon}
- 对每篇填表时，重点看三件事：
  1. 有没有区分"输入能约束的部分"和"模型生成的部分"
  2. 校准是整体算的还是分尺度 / 分子空间算的
  3. 用的评分对离群值是否稳健

## 本周结论（第 1 周末填写）

- 已有工作普遍覆盖的：
- 没有人做的：
- 差异点是否成立：是 / 否 / 需调整为

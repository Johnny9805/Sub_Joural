# 前 4 周执行计划

在启动方案的 4 周清单基础上，按本次核查结果细化。每周末在本文件末尾的日志里记一行。

## 第 0 周：工作部署（已完成，2026-10-12）

- [x] 仓库骨架、Python 包 `dsaudit`、测试（27 项通过）
- [x] 数据与模型可得性核查 → [data_availability.md](data_availability.md)
- [x] 方法定义 v0 与待决问题 → [method_notes.md](method_notes.md)
- [x] 合成玩具问题端到端跑通（三子空间、分子空间校准、稳健评分）
- [x] IBTrACS 下载并筛出 2021 年台风个例；GHCNh 台湾站点清点
- [x] ERA5 下载脚本（已 dry-run）、CorrDiff 获取脚本与 GPU 冒烟测试脚本（未运行）

## 第 1 周：文献对比表

- [ ] 精读 CorrDiff 原文：训练 / 测试年份划分、个例（是否有璨树）、评估指标
- [ ] 精读 Sun et al.（GMD 2026）的评估部分
- [ ] 补全 [week1_literature_matrix.md](week1_literature_matrix.md)，至少 8 篇
- [ ] 回答 method_notes §7 第 1 条：TAP 核的公式是否与 v0 一致
- **判据**：差异点（出处分解 + 先验汇率 + 稳健校准）中至少两项没人做过，否则重新定位

## 第 2 周：数据与推理跑通

需要你本人操作的前置步骤：
- [ ] 注册 CDS → `~/.cdsapirc` → 接受两个 ERA5 数据集的许可
- [ ] 申请 CWA 开放数据会员，确认 2021 年历史逐时自动站数据能否批量下载
- [ ] 确认 GPU 型号与显存

然后：
- [ ] `python scripts/data/download_era5.py`（2021 年 7–10 月，约几百 MB）
- [ ] `bash scripts/setup/get_corrdiff.sh` → `python scripts/setup/corrdiff_smoke_test.py`，记录：显存峰值、单样本耗时、能否求梯度
- [ ] 用 ERA5 驱动 CorrDiff，对璨树（2021-09-10 至 09-12）逐时生成 16–32 个成员
- **判据**：冒烟测试通过；如果显存不够，退到子区域或只用回归 UNet

## 第 3 周：最小可行结果

- [ ] 选 64×64 子区域（台湾中部山区，含中央山脉）
- [ ] 先验 B：用 CWA 训练数据（2018–2020）估计经验协方差 + 收缩（需要下载 CWA 数据集，97.65 GB，可只取子区域）
- [ ] 用 `assignment_operator` 构造 Lambert 网格 → ERA5 0.25° 的粗化算子
- [ ] 路线 B（统计估计，无需梯度）：用 `empirical_averaging_kernel` 估计"数据 + 耦合"平均核；用 `build_provenance_basis` 得到 S_d
- [ ] 输出一个样本的三子空间物理能量占比和分尺度占比
- **判据**：三部分占比数值稳定（对 τ 和收缩参数不敏感到量级层面）

## 第 4 周：站点对齐与决策

- [ ] 把 GHCNh（8 站）以及能拿到的 CWA 自动站对齐到 2 km 网格（最近格点 + 高度订正说明）
- [ ] 按事件强度分层（IBTrACS 距离 / 强度，ERA5 tcwv 或降水）
- [ ] 画第一张"可信细节比例 vs 事件强度"草图，叠加分子空间 spread–skill
- [ ] 同时报告 MSE、CRPS、核评分；核宽用双线性插值基线误差的 MAD 定一次
- **决策**：先验填充部分的离散度在强事件中是否明显低估（H2 的雏形）→ 决定是否继续投入

## 日志

| 日期 | 进展 | 卡点 |
|---|---|---|
| 2026-10-12 | 工作部署完成 | 需要你提供：CDS 账号、CWA 会员、GPU 信息、TAP 核公式 |

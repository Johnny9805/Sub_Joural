# 课题 1：生成式气象降尺度的细节出处审计

**How much of AI km-scale weather detail is in the data? Provenance and calibration of generative downscaling**

AI 千米级气象降尺度中有多少细节来自数据？本课题把生成式降尺度（首先是 CorrDiff-Taiwan）输出的每个样本、误差和集合离散度，分到三个子空间：**数据可信**（粗分辨率输入直接约束）、**耦合可信**（经由地形和多变量耦合约束）、**先验填充**（模型自己生成）。然后在每个子空间里分别检验校准，并用基于相关熵的稳健评分对独立站点做验证。

- 启动方案与假设 H1–H4：[docs/project1_kickoff.md](docs/project1_kickoff.md)
- 数据与模型可得性核查（2026-10-12）：[docs/data_availability.md](docs/data_availability.md)
- 方法定义 v0 与待决问题：[docs/method_notes.md](docs/method_notes.md)
- 前 4 周执行计划：[docs/workplan.md](docs/workplan.md)
- 第 1 周文献对比表：[docs/week1_literature_matrix.md](docs/week1_literature_matrix.md)

## 目录

```
src/dsaudit/          分析核心（只依赖 numpy/scipy，CPU 即可）
  operators.py        粗粒化前向算子（规则网格 / 曲线网格 → ERA5）
  priors.py           高斯先验与矩阵平方根
  decomposition.py    噪声加权 GN 核、三子空间、能量占比、无雅可比的平均核估计
  calibration.py      秩直方图、spread–skill（整体与分子空间）
  scores.py           MSE、CRPS、相关熵与高斯核评分、自适应核宽
  spectra.py          径向能谱与分尺度能量占比
  toy.py              合成地形 / 递减率问题与精确后验（流程验证用）
experiments/          实验脚本（toy_terrain_demo.py：端到端演示）
scripts/data/         ERA5、IBTrACS、GHCNh 站点下载
scripts/setup/        CorrDiff 权重获取与 GPU 冒烟测试
configs/              研究区域与时段（台湾，2021 年台风季）
tests/                单元测试
data/, outputs/       本地数据与结果（不进 git）
```

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" matplotlib
pytest                                          # 27 项测试
python experiments/toy_terrain_demo.py          # 约 1 分钟，结果在 outputs/toy/

# 数据（见 docs/data_availability.md 的前置条件）
python scripts/data/download_ibtracs.py         # 无需账号
python scripts/data/download_stations.py        # 无需账号
python scripts/data/download_era5.py --dry-run  # 实际下载需 CDS 令牌
```

GPU 工作站（CorrDiff 推理）：

```bash
conda env create -f environment.yml && conda activate dsaudit
pip install -e ".[data,dev]" "earth2studio[corrdiff]"
bash scripts/setup/get_corrdiff.sh
python scripts/setup/corrdiff_smoke_test.py --time 2021-09-11T21:00 --samples 4
```

## 当前状态

第 0 周（工作部署）已完成：分析核心经测试验证，合成问题上端到端跑通，IBTrACS 与站点数据脚本已在真实数据上运行。下一步需要你提供的东西列在 [docs/workplan.md](docs/workplan.md) 的日志里。

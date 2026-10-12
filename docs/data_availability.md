# 数据与模型可得性核查

核查日期：2026-10-12。逐项对应 [启动方案](project1_kickoff.md) 中"数据与模型"清单。
状态标记：✅ 已在本环境实测可下载 / ☑️ 官方页面确认、需你本人注册或操作 / ⚠️ 有限制，影响方案 / ❓ 未核实。

## 结论速览

| 项目 | 状态 | 一句话结论 |
|---|---|---|
| ERA5（CDS） | ☑️ | 可得；需注册 CDS、把令牌写进 `~/.cdsapirc`、在两个数据集页面接受许可。冒烟测试可先用免注册的 ARCO-ERA5 |
| CorrDiff 台湾权重 | ✅ ⚠️ | NGC 包可下载（Apache-2.0）；**输出没有降水，只有最大雷达反射率**；Earth2Studio 标注 40 GB 级 GPU；前向带 `inference_mode`，求雅可比要绕开 |
| 中国区域 3 km（arXiv:2512.05377） | ⚠️ | **与 EGUsphere-2026-822 是同一篇论文**；代码与部分样例数据在 Zenodo，权重未确认；3 km 目标数据（CMA-RRA）不公开 |
| 独立站点：GHCN-Daily | ⚠️ | **台湾本岛没有 GHCN-D 站点**，台湾区域不可用；大陆扩展时可用 |
| 独立站点：ISD | ⚠️ | 2025-08-24 后停更，NCEI 原下载服务计划 2026-07-31 关闭；改用 GHCNh |
| 独立站点：GHCNh | ✅ ⚠️ | 可下载；但 2021 年 7–10 月台湾**只有 8 个站有数据**，且没有山区站点 |
| 台湾气象署（CWA）站点 | ❓ | 自动站逐时资料公开，但站点数据须向 CWA 申请会员下载；**建议作为主力独立观测** |
| 中国气象局站点 | ❓ | 未核实；大陆扩展阶段再办 |
| IBTrACS | ✅ | v04r01 WP 文件已下载并筛出 2021 年影响台湾的 5 个台风 |
| GPM IMERG | ❓ | V07 Final（0.1°、半小时）存在；下载需 NASA Earthdata 账号，未实测 |
| GPU | ⚠️ | 本云端环境无 GPU（4 核 CPU、15 GB 内存）；分解原型够用，CorrDiff 推理需你的工作站 |

## 逐项说明

### 1. ERA5 再分析 ☑️
- 新版 CDS 要求 `cdsapi>=0.7.2`，`~/.cdsapirc` 写入个人访问令牌（旧的 `UID:key` 格式已失效）。令牌在登录后的 [how-to-api 页面](https://cds.climate.copernicus.eu/how-to-api)。
- **必须手动**在 `reanalysis-era5-single-levels` 和 `reanalysis-era5-pressure-levels` 两个数据集页面底部接受许可，否则请求会被拒。
- 脚本 `scripts/data/download_era5.py` 已写好；`--dry-run` 已验证请求格式（每月 2 个请求，2021 年 7–10 月共 8 个）。实际下载需要你的令牌，未在本环境运行。
- 下载区域取已发布权重的输入网格：19.25–28°N、116–125.75°E（36×40 个 0.25° 格点），数据量很小（四个月约几百 MB）。
- 免注册替代：Earth2Studio 的 `ARCO` 数据源（Google 云上的 ARCO-ERA5）。GPU 冒烟测试脚本默认用它。

### 2. CorrDiff 公开代码与权重 ✅ ⚠️
- **权重**：NGC `nvidia/modulus/corrdiff_inference_package` v1（2024-04-10，684 MB）。检查点许可 Apache-2.0；包内样例数据 CC BY-NC-ND 4.0。下载链接已实测可解析（302 跳转）。
- **代码**：PhysicsNeMo（原 Modulus，NGC 页面固定在提交 `ee6b3ac`）；Earth2Studio 提供 `CorrDiffTaiwan` 封装（本次读的是 2026-10-07 的 main 分支源码）。
- **已发布权重的变量**（来自 Earth2Studio 源码）：
  - 输入 12 个：`tcwv`，500/850 hPa 的 `z t u v`，`t2m u10m v10m`。训练数据集说明写的是 20 个（另有 700/925 hPa），已发布的这版权重只用 12 个。
  - 输出 4 个：`mrr`（1 小时最大雷达反射率，dBZ）、`t2m`、`u10m`、`v10m`；2 km 的 448×448 Lambert 网格，Earth2Studio 输出时去掉了 1 个像元的边框。
- **训练数据**：NGC `modulus_datasets_cwa`，2018-01 至 2022-04 逐时，Zarr 格式 97.65 GB，CC BY-NC-ND 4.0（可用于研究，不可再分发）。训练/测试年份划分还需在第 1 周对照论文确认（配置里假设 2021 年为测试年）。
- **对研究设计的影响**：
  1. **没有降水输出。** H2 里的"强降水"只能用 `mrr` 代替：可以对雷达观测检验，或经 Z–R 关系换算后对雨量站检验（会额外引入换算误差），也可以把 H2 的检验变量改成 `t2m`、`u10m`、`v10m`、阵风。
  2. `mrr` 在输入里**没有粗分辨率对应量**，按定义它的"数据可信"子空间为空，信息只能来自耦合或先验。这反而是 H3 最干净的测试变量。
  3. Earth2Studio 给该模型标的是 `gpu:40gb`；方案里假设的 24 GB 能否跑通，要看冒烟测试结果。
  4. `CorrDiffTaiwan._forward` 带 `@torch.inference_mode()`，公开调用没有梯度。冒烟测试会试着调用未装饰的函数；如果行不通，有两条路：第 3 周改用不需要雅可比的统计估计（见 [method_notes.md](method_notes.md) §3），或者直接对 PhysicsNeMo 的回归 UNet 求导。

### 3. 中国区域 3 km 降尺度（arXiv:2512.05377） ⚠️
- **arXiv:2512.05377 与 EGUsphere-2026-822 是同一篇论文**：Sun, Jing 等，"China Regional 3 km Downscaling Based on Residual Corrective Diffusion Model"。启动方案把两者分别列为"中国 3 km 工作"和"降尺度评估工作"，但它们是同一项工作。EGUsphere 页面显示讨论已关闭，终稿发表于 GMD 19, 9235–9262（2026-09-29）。
- 代码与可用性：作者回复中称"模型代码和部分样例数据"放在 Zenodo [10.5281/zenodo.19244294](https://doi.org/10.5281/zenodo.19244294)（v2，单个 6.8 GB 的 tar，CC BY 4.0）。**页面没有提到训练好的权重**，要解开 tar 才能确认。
- 3 km 目标数据 CMA-RRA 不公开，正式发布前须向 CMA 地球系统数值预报中心（CEMC）申请。
- 影响：第 1 周对比表要另找评估类文献（见 [week1_literature_matrix.md](week1_literature_matrix.md)）；大陆扩展取决于 tar 里有没有权重，以及能否拿到 CMA-RRA。

### 4. 独立站点观测
- **GHCN-Daily ⚠️**：在 19.5–28°N、116–126°E 范围内只有 15 个站，分布在福建、广东沿海、日本八重山/宫古群岛和菲律宾巴丹群岛，**台湾本岛一个都没有**。按站下载的路径可用（`/pub/data/ghcn/daily/by_station/<ID>.csv.gz`），留给大陆扩展。
- **ISD ⚠️**：据 NCEI 2026-06-23 的公告，ISD 已被 GHCNh 取代，2025-08-24 之后不再更新；NCEI 自己的 HTTPS/FTP 下载服务计划于 2026-07-31 关闭，之后只通过 NOAA 开放数据分发（NODD）云端提供。
- **GHCNh ✅ ⚠️**：按站文件 `access/by-station/GHCNh_<ID>_por.psv` 实测可下载；脚本 `scripts/data/download_stations.py` 已在全部 75 个台湾站点条目上跑通。**2021-07-01 至 10-31 只有 8 个站有数据**：
  - 逐时（机场 METAR）：高雄国际机场、台南、清泉岗，各约 5600–6200 条
  - 3 小时（天气站）：台北、台南、恒春、台中、兰屿，各约 850–980 条
  - 阿里山（2406 m）、玉山（3850 m）等山区站在 GHCNh 里这段时间没有数据
  - 这个密度不够第 4 周按事件强度分层统计。
- **台湾气象署（CWA）站点 ❓（建议主力）**：气象资料开放平台提供全台自动气象站逐时资料（气温、风、阵风、降水、气压等），见 [data.gov.tw 9176](https://data.gov.tw/dataset/9176) 与 [9177](https://data.gov.tw/dataset/9177)（自动雨量站）。历史资料走 CODiS，批量下载需申请会员。**需要你注册后确认 2021 年历史逐时资料能否批量取得。**
  - 独立性：CorrDiff 推理时只看 ERA5。没有上 GTS 的 CWA 自动站不进 ERA5，相对模型输入是独立的；但它们可能被同化进 RWRF（训练目标），这一点要在文中说明。
- **中国气象局站点 ❓**：国家气象科学数据中心需注册，高密度站点数据有限制；放到大陆扩展阶段处理。GHCNh 站点表中 CH 开头的条目有 1620 个。

### 5. IBTrACS ✅
- v04r01 WP 文件（115 MB）已下载，每周更新三次。脚本按单一机构取风速（默认 JTWC 1 分钟风，可改为 RSMC Tokyo 10 分钟风），避免不同机构的风速混用。
- 2021-07-01 至 10-31 进入域（含 1° 外扩）且风速 ≥34 kt 的台风：

  | 台风 | 在域内时间 | 域内最大风速（JTWC / Tokyo） | 距台湾中心最近 |
  |---|---|---|---|
  | 烟花 In-fa | 07-21 至 07-24 | 95 / 85 kt | 421 km |
  | 卢碧 Lupit | 08-04 至 08-07 | 45 / 40 kt | 150 km |
  | 奥麦斯 Omais | 08-21 至 08-22 | 50 / 45 kt | 460 km |
  | **璨树 Chanthu** | 09-10 至 09-12 | **155 / 115 kt** | **130 km**（09-11 21 UTC） |
  | 圆规 Kompasu | 10-10 至 10-12 | 60 / 55 kt | 545 km |

  璨树是首选个例：强度最大、离台湾最近，GPU 冒烟测试默认取它最接近台湾的时刻。

### 6. GPM IMERG ❓
- V07 Final Run（GPM_3IMERGHH，0.1°、半小时，延迟约 3.5 个月）。官方从 NASA GES DISC 下载，需要 Earthdata 账号；NCAR GDEX（d731000）有镜像。未实测。

### 7. 计算资源 ⚠️
- 本云端环境：无 GPU、4 核 CPU、15 GB 内存。稠密矩阵实现的分解能处理约 10⁴ 维（64×64 网格、1–2 个变量），玩具实验约 1 分钟跑完。
- CorrDiff 推理和求雅可比要在你的 GPU 工作站上做；先跑 `scripts/setup/corrdiff_smoke_test.py`，看实际显存占用和能否求梯度。

# data/（不进 git）

```
data/raw/era5/          ERA5 月文件 era5_{sl,pl}_YYYYMM.nc（download_era5.py）
data/raw/ibtracs/       IBTrACS v04r01 原始 CSV（download_ibtracs.py）
data/raw/stations/      GHCNh 站点表（download_stations.py）
data/processed/         筛选后的台风列表、台风路径点、站点时段子集
```

所有文件都可以用 `scripts/data/` 下的脚本重新生成，脚本会跳过已存在的文件。

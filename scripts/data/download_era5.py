"""Download ERA5 inputs for the study domain and period from the Copernicus CDS.

One-time setup
--------------
1. Register at https://cds.climate.copernicus.eu and copy the two lines shown at
   https://cds.climate.copernicus.eu/how-to-api into ~/.cdsapirc
   (url + personal access token; the old "UID:key" format no longer works).
2. Accept the licence at the bottom of the download form of BOTH datasets:
   reanalysis-era5-single-levels and reanalysis-era5-pressure-levels.
3. pip install "cdsapi>=0.7.2"

Usage
-----
    python scripts/data/download_era5.py --config configs/taiwan_typhoon2021.toml --dry-run
    python scripts/data/download_era5.py --config configs/taiwan_typhoon2021.toml

Writes data/raw/era5/era5_{sl,pl}_YYYYMM.nc, one file per month and dataset;
existing files are skipped, so the script can be re-run after interruptions.
"""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path

from _common import REPO, load_config, period


def monthly_requests(cfg: dict) -> list[tuple[str, str, dict]]:
    start, end = period(cfg)
    e = cfg["era5"]
    times = [f"{h:02d}:00" for h in cfg["period"]["hours"]]
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        last = calendar.monthrange(y, m)[1]
        d0 = start.day if (y, m) == (start.year, start.month) else 1
        d1 = end.day if (y, m) == (end.year, end.month) else last
        common = {
            "product_type": ["reanalysis"],
            "year": [f"{y}"],
            "month": [f"{m:02d}"],
            "day": [f"{d:02d}" for d in range(d0, d1 + 1)],
            "time": times,
            "area": e["area"],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        tag = f"{y}{m:02d}"
        out.append(("reanalysis-era5-single-levels", f"era5_sl_{tag}.nc", {**common, "variable": e["single_levels"]}))
        out.append(
            (
                "reanalysis-era5-pressure-levels",
                f"era5_pl_{tag}.nc",
                {
                    **common,
                    "variable": e["pressure_levels"],
                    "pressure_level": [str(p) for p in e["levels_hpa"]],
                },
            )
        )
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=REPO / "configs/taiwan_typhoon2021.toml")
    p.add_argument("--out", type=Path, default=REPO / "data/raw/era5")
    p.add_argument("--dry-run", action="store_true", help="print the requests without submitting")
    args = p.parse_args()

    requests = monthly_requests(load_config(args.config))
    if args.dry_run:
        for dataset, fname, req in requests:
            print(f"# {dataset} -> {fname}\n{json.dumps(req)}")
        return

    import cdsapi  # imported here so --dry-run works without it

    client = cdsapi.Client()
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset, fname, req in requests:
        target = args.out / fname
        if target.exists():
            print(f"exists, skipping: {target}")
            continue
        print(f"requesting {dataset} -> {target}")
        client.retrieve(dataset, req).download(str(target))


if __name__ == "__main__":
    main()

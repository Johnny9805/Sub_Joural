"""Fetch independent station observations (NOAA GHCNh, hourly) for the study box.

GHCNh is NCEI's successor of ISD (ISD stopped updating in Aug 2025 and its
NCEI HTTPS service was scheduled to close on 31 Jul 2026). GHCN-Daily has no
stations on Taiwan itself, so it is not used for the Taiwan domain.

Per-station "period of record" files can be ~100 MB; they are streamed and
only rows inside the study period and the configured columns are kept.

    python scripts/data/download_stations.py --config configs/taiwan_typhoon2021.toml --list-only
    python scripts/data/download_stations.py --config configs/taiwan_typhoon2021.toml

Outputs
  data/raw/stations/ghcnh-station-list.txt
  data/processed/stations/ghcnh_<ID>_<start>_<end>.csv   (one per active station)
  data/processed/stations/ghcnh_inventory.csv            (rows kept per station)
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from _common import REPO, download, load_config, open_url, period

BASE = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly"
META = ["STATION", "Station_name", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION"]


def stations_in_box(station_list: Path, box: dict) -> list[dict]:
    out = []
    with open(station_list) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                lat, lon, elev = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            if box["lat_min"] <= lat <= box["lat_max"] and box["lon_min"] <= lon <= box["lon_max"]:
                out.append({"id": parts[0], "lat": lat, "lon": lon, "elev": elev, "name": line[41:80].strip()})
    return out


def fetch_station(sid: str, start, end, columns: list[str], dest: Path) -> int:
    """Stream one por file, keep rows within [start, end] and the selected columns."""
    url = f"{BASE}/access/by-station/GHCNh_{sid}_por.psv"
    years = {str(y) for y in range(start.year, end.year + 1)}
    lo, hi = start.isoformat(), end.isoformat() + "T23:59:59"
    n = 0
    with open_url(url, timeout=120) as r:
        text = io.TextIOWrapper(r, encoding="utf-8", errors="replace", newline="")
        header = text.readline().rstrip("\r\n").split("|")
        idx = {name: i for i, name in enumerate(header)}
        keep = [c for c in META if c in idx]
        for c in columns:
            keep += [k for k in (c, f"{c}_Quality_Code") if k in idx]
        i_year, i_date = idx["Year"], idx["DATE"]
        tmp = dest.with_suffix(".part")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keep)
            for line in text:
                row = line.rstrip("\r\n").split("|")
                if len(row) <= i_date or row[i_year] not in years or not (lo <= row[i_date] <= hi):
                    continue
                w.writerow([row[idx[k]] for k in keep])
                n += 1
    if n:
        tmp.rename(dest)
    else:
        tmp.unlink()
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=REPO / "configs/taiwan_typhoon2021.toml")
    p.add_argument("--raw", type=Path, default=REPO / "data/raw/stations")
    p.add_argument("--out", type=Path, default=REPO / "data/processed/stations")
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--max-stations", type=int, default=None, help="for testing")
    args = p.parse_args()

    cfg = load_config(args.config)
    start, end = period(cfg)
    lst = download(f"{BASE}/doc/ghcnh-station-list.txt", args.raw / "ghcnh-station-list.txt")
    stations = stations_in_box(lst, cfg["stations"])
    print(f"{len(stations)} GHCNh stations in the box")
    if args.list_only:
        for s in stations:
            print(f"  {s['id']}  {s['lat']:8.4f} {s['lon']:9.4f} {s['elev']:7.1f} m  {s['name']}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    inventory = []
    for s in stations[: args.max_stations]:
        dest = args.out / f"ghcnh_{s['id']}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        if dest.exists():
            with open(dest) as f:
                n = sum(1 for _ in f) - 1
        else:
            try:
                n = fetch_station(s["id"], start, end, cfg["stations"]["ghcnh_columns"], dest)
            except OSError as err:
                print(f"  {s['id']}: failed ({err})")
                n = -1
        print(f"  {s['id']} {s['name']:<30} rows in period: {n}")
        inventory.append({**s, "rows": n})
    if inventory:
        with open(args.out / "ghcnh_inventory.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(inventory[0]))
            w.writeheader()
            w.writerows(inventory)
        active = sum(1 for r in inventory if r["rows"] > 0)
        print(f"{active}/{len(inventory)} stations have data in {start}..{end}")


if __name__ == "__main__":
    main()

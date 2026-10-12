"""Download IBTrACS v04r01 and list the storms that affect the study domain.

    python scripts/data/download_ibtracs.py --config configs/taiwan_typhoon2021.toml

Outputs
  data/raw/ibtracs/ibtracs.<BASIN>.list.v04r01.csv   (~115 MB for WP)
  data/processed/ibtracs_storms.csv        one row per storm entering the box
  data/processed/ibtracs_track_points.csv  the storms' 3-hourly points in the box

Intensity comes from one agency only (config [ibtracs] wind_agency): "USA"
(JTWC, 1-min sustained) or "TOKYO" (RSMC Tokyo, 10-min). Mixing agencies
across track points, as a WMO-then-USA fallback would, makes intensities
inconsistent between storms.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path

from _common import REPO, download, load_config, period

BASE = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv"
TAIWAN_CENTRE = (23.7, 121.0)


def _float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


def _distance_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return 6371.0 * math.acos(max(-1.0, min(1.0, c)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=REPO / "configs/taiwan_typhoon2021.toml")
    p.add_argument("--raw", type=Path, default=REPO / "data/raw/ibtracs")
    p.add_argument("--out", type=Path, default=REPO / "data/processed")
    args = p.parse_args()

    cfg = load_config(args.config)
    basin = cfg["ibtracs"]["basin"]
    min_wind = cfg["ibtracs"]["min_wind_kt"]
    agency = cfg["ibtracs"].get("wind_agency", "USA")
    d = cfg["domain"]
    lat0, lat1 = d["lat_min"] - d["margin"], d["lat_max"] + d["margin"]
    lon0, lon1 = d["lon_min"] - d["margin"], d["lon_max"] + d["margin"]
    start, end = period(cfg)
    t_start = dt.datetime.combine(start, dt.time(0))
    t_end = dt.datetime.combine(end, dt.time(23, 59))

    src = download(f"{BASE}/ibtracs.{basin}.list.v04r01.csv", args.raw / f"ibtracs.{basin}.list.v04r01.csv")

    points, header = [], None
    with open(src, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        next(reader)  # units row
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            season = int(row[col["SEASON"]])
            if season < start.year or season > end.year:
                continue
            t = dt.datetime.fromisoformat(row[col["ISO_TIME"]])
            if not (t_start <= t <= t_end):
                continue
            lat, lon = _float(row[col["LAT"]]), _float(row[col["LON"]])
            if not (lat0 <= lat <= lat1 and lon0 <= lon <= lon1):
                continue
            wind = _float(row[col[f"{agency}_WIND"]])
            points.append(
                {
                    "SID": row[col["SID"]],
                    "NAME": row[col["NAME"]],
                    "ISO_TIME": t.isoformat(sep=" "),
                    "LAT": lat,
                    "LON": lon,
                    "WIND_KT": wind,
                    "PRES_HPA": _float(row[col[f"{agency}_PRES"]]),
                    "DIST_TW_KM": round(_distance_km(lat, lon, *TAIWAN_CENTRE), 1),
                }
            )

    storms = {}
    for pt in points:
        s = storms.setdefault(pt["SID"], {"SID": pt["SID"], "NAME": pt["NAME"], "points": []})
        s["points"].append(pt)
    rows = []
    for s in storms.values():
        pts = s["points"]
        winds = [q["WIND_KT"] for q in pts if not math.isnan(q["WIND_KT"])]
        max_wind = max(winds) if winds else math.nan
        if not (max_wind >= min_wind):
            continue
        closest = min(pts, key=lambda q: q["DIST_TW_KM"])
        rows.append(
            {
                "SID": s["SID"],
                "NAME": s["NAME"],
                "first_in_box": pts[0]["ISO_TIME"],
                "last_in_box": pts[-1]["ISO_TIME"],
                "n_points": len(pts),
                "max_wind_kt_in_box": max_wind,
                "min_dist_taiwan_km": closest["DIST_TW_KM"],
                "time_of_closest": closest["ISO_TIME"],
            }
        )
    rows.sort(key=lambda r: r["first_in_box"])
    kept = {r["SID"] for r in rows}

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "ibtracs_storms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["SID"])
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "ibtracs_track_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0]) if points else ["SID"])
        w.writeheader()
        w.writerows(pt for pt in points if pt["SID"] in kept)

    print(f"{len(rows)} storms with >= {min_wind} kt ({agency}) inside the box, {start} to {end}:")
    for r in rows:
        print(
            f"  {r['NAME']:<12} {r['first_in_box'][:10]} .. {r['last_in_box'][:10]}  "
            f"max {r['max_wind_kt_in_box']:>5.0f} kt  closest {r['min_dist_taiwan_km']:>6.0f} km ({r['time_of_closest']})"
        )


if __name__ == "__main__":
    main()

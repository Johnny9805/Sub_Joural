"""Shared helpers for the data scripts (stdlib only)."""

from __future__ import annotations

import datetime as dt
import shutil
import time
import tomllib
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def period(cfg: dict) -> tuple[dt.date, dt.date]:
    p = cfg["period"]
    return dt.date.fromisoformat(p["start"]), dt.date.fromisoformat(p["end"])


def open_url(url: str, retries: int = 4, timeout: float = 60.0):
    """urlopen with exponential backoff (2, 4, 8, 16 s) on network errors."""
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except OSError as err:
            if attempt == retries or getattr(err, "code", 500) < 500:
                raise
            time.sleep(2 ** (attempt + 1))


def download(url: str, dest: Path, overwrite: bool = False) -> Path:
    """Download url to dest (atomic rename); skip if dest exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        print(f"exists, skipping: {dest}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open_url(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)
    print(f"downloaded {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest

"""CorrDiff-Taiwan smoke test via Earth2Studio, for the GPU workstation.

Not yet run: the cloud dev container has no GPU and no torch. Run it once on
the workstation and keep the printed report (week 2 deliverable).

    pip install "earth2studio[corrdiff]"
    python scripts/setup/corrdiff_smoke_test.py --time 2021-09-11T21:00 --samples 4

Checks
  1. the NGC package downloads and loads; input/output variables and grids
  2. ERA5 input from the public ARCO store (no CDS account needed for this test)
  3. one ensemble forward pass: wall time and peak GPU memory
  4. whether autograd reaches the input. CorrDiffTaiwan._forward is wrapped in
     @torch.inference_mode(), so the public call has no gradients; the test
     tries the undecorated function. The result decides how the coupling
     Jacobian (week 3) is computed.
"""

from __future__ import annotations

import argparse
import time as timer
from pathlib import Path

import numpy as np
import torch
from earth2studio.data import ARCO, prep_data_array
from earth2studio.models.dx import CorrDiffTaiwan
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", default="2021-09-11T21:00", help="UTC, ERA5 hour (default: Chanthu closest to Taiwan)")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/corrdiff_smoke"))
    p.add_argument("--skip-grad", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    model = CorrDiffTaiwan.load_model(CorrDiffTaiwan.load_default_package()).to(device)
    model.number_of_samples = args.samples
    model.seed = args.seed
    ic = model.input_coords()
    oc = model.output_coords(ic)
    print("input variables :", list(ic["variable"]))
    print(f"input grid      : lat {ic['lat'][0]}..{ic['lat'][-1]} ({len(ic['lat'])}), "
          f"lon {ic['lon'][0]}..{ic['lon'][-1]} ({len(ic['lon'])})")
    print("output variables:", list(oc["variable"]))
    print("output grid     :", oc["lat"].shape, f"steps={model.number_of_steps} solver={model.solver}")

    t = to_time_array([args.time])
    x, coords = prep_data_array(ARCO()(t, ic["variable"]), device=device)
    x, coords = map_coords(x, coords, ic)
    print("input tensor    :", tuple(x.shape))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = timer.time()
    y, ycoords = model(x, coords)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = timer.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    print(f"forward         : {tuple(y.shape)} in {dt:.1f} s, peak GPU memory {mem:.1f} GiB")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / f"corrdiff_{args.time.replace(':', '')}.npz",
        y=y.detach().cpu().numpy(),
        x=x.detach().cpu().numpy(),
        lat=ycoords["lat"],
        lon=ycoords["lon"],
        variables=np.array(ycoords["variable"]),
    )

    if args.skip_grad:
        return
    raw = getattr(type(model)._forward, "__wrapped__", None)
    if raw is None:
        print("autograd        : _forward has no __wrapped__; re-implement the forward pass for Jacobians")
        return
    model.number_of_samples = 1
    # _forward takes one batch element, shape (variable, lat, lon)
    xg = x.detach().reshape(-1, *x.shape[-3:])[0].clone().requires_grad_(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        with torch.enable_grad():
            out = raw(model, xg)
            i, j = out.shape[-2] // 2, out.shape[-1] // 2
            out[..., 1, i, j].sum().backward()  # t2m at the domain centre
        g = xg.grad
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
        print(f"autograd        : OK, |d t2m(centre) / d input| = {g.norm().item():.3e}, peak {mem:.1f} GiB")
        print("                  input sensitivity per variable:",
              {v: f"{g[k].norm().item():.2e}" for k, v in enumerate(ic["variable"])})
    except Exception as err:  # report rather than crash: this is a capability probe
        print(f"autograd        : FAILED ({type(err).__name__}: {err})")


if __name__ == "__main__":
    main()

"""End-to-end dry run of the audit on the synthetic terrain / lapse-rate problem.

What it shows, before any real data:
  * the three-subspace split (dims, DOFS, prior-fill -> coupled conversion, H3)
  * energy fractions of truth, members, errors and spread, overall and by scale
  * per-subspace calibration of three "downscalers":
      linear_only   exact posterior given the coarse field only
      coupled       exact posterior given coarse field + lapse-rate information
      overconfident coupled posterior mean, prior-fill spread shrunk by --alpha
                    (the failure mode H2 looks for in CorrDiff)
  * MSE / CRPS / kernel (correntropy) scores with and without gross station
    errors (H4)

Run:  python experiments/toy_terrain_demo.py --out outputs/toy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dsaudit.calibration import rank_flatness, subspace_rank_histograms, subspace_spread_skill
from dsaudit.decomposition import build_provenance_basis
from dsaudit.operators import block_average_operator
from dsaudit.priors import gaussian_prior
from dsaudit.scores import crps_ensemble, kernel_score_ensemble, kernel_width, mse
from dsaudit.spectra import band_fractions
from dsaudit.toy import (
    LinearGaussianModel,
    block_terrain_patterns,
    lapse_rate_operator,
    ridge_terrain,
    toy_covariance,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=64, help="HR grid size (n x n), 2 km cells")
    p.add_argument("--factor", type=int, default=8, help="coarsening factor")
    p.add_argument("--cases", type=int, default=200)
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--noise-coarse", type=float, default=0.2, help="coarse input error std [K]")
    p.add_argument("--noise-lapse", type=float, default=0.7, help="lapse-rate estimate error std [K/km]")
    p.add_argument("--lapse-std", type=float, default=2.0, help="prior lapse-rate anomaly std [K/km]")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.4, help="prior-fill spread factor of the overconfident sampler")
    p.add_argument("--outlier-rate", type=float, default=0.02)
    p.add_argument("--outlier-size", type=float, default=8.0, help="gross station error [K]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/toy"))
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    h = ridge_terrain(a.n, seed=a.seed)
    U, _ = block_terrain_patterns(h, a.factor)
    prior = gaussian_prior(toy_covariance(a.n, U, lapse_std=a.lapse_std))
    A = block_average_operator((a.n, a.n), a.factor)
    Jc = lapse_rate_operator(U)
    basis = build_provenance_basis(prior, A, a.noise_coarse, Jc, a.noise_lapse, tau=a.tau)
    print(f"setup {time.time() - t0:.1f}s  dims={basis.dims}")

    linear_model = LinearGaussianModel(prior, [(A, a.noise_coarse)])
    coupled_model = LinearGaussianModel(prior, [(A, a.noise_coarse), (Jc, a.noise_lapse)])
    truths = prior.sample(rng, a.cases)
    ens = {"linear_only": [], "coupled": [], "overconfident": []}
    for x in truths:
        y = A @ x + a.noise_coarse * rng.standard_normal(A.shape[0])
        c = Jc @ x + a.noise_lapse * rng.standard_normal(Jc.shape[0])
        lin = linear_model.posterior([y])
        cpl = coupled_model.posterior([y, c])
        ens["linear_only"].append(lin.sample(rng, a.members))
        ens["coupled"].append(cpl.sample(rng, a.members))
        dz = basis.split_whitened(cpl.whitened_deviations(rng, a.members))
        dz = dz["data"] + dz["coupled"] + a.alpha * dz["prior"]
        ens["overconfident"].append(cpl.mean + prior.color(dz))
    ens = {k: np.stack(v, axis=1) for k, v in ens.items()}  # (K, cases, n)
    print(f"sampling {time.time() - t0:.1f}s")

    res = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "dims": basis.dims,
        "dofs": {
            "data": basis.dofs_data,
            "data_plus_coupling": basis.dofs_total,
            "converted": basis.dofs_converted,
            "prior_fill_dims_converted_fraction": basis.dims["coupled"] / (basis.prior.size - basis.dims["data"]),
        },
    }

    def mean_energy(x, anomaly):
        e = basis.energy(x, anomaly=anomaly)
        return {norm: {k: float(np.mean(v)) for k, v in d.items()} for norm, d in e.items()}

    res["energy"] = {"truth": mean_energy(truths, False)}
    edges = [np.inf, 4 * a.factor, a.factor, 2]  # wavelength bands in cells
    res["band_edges_cells"] = [str(e) for e in edges]
    res["bands_truth"] = _bands(basis, truths, False, a.n, edges)

    # gross station errors on a random 'outlier-rate' fraction of grid points
    contaminated = truths.copy()
    mask = rng.random(truths.shape) < a.outlier_rate
    contaminated[mask] += a.outlier_size * rng.choice([-1.0, 1.0], size=mask.sum())
    width = kernel_width(truths - prior.mean, "mad")
    res["kernel_width"] = width

    for name, E in ens.items():
        mean = E.mean(axis=0)
        r = {
            "energy_member": mean_energy(E[0], False),
            "energy_error": mean_energy(mean - truths, True),
            "energy_spread": mean_energy(E - mean, True),
            "bands_spread": _bands(basis, (E - mean)[0], True, a.n, edges),
            "spread_skill": subspace_spread_skill(basis, E, truths),
            "rank_flatness": {
                s: rank_flatness(c) for s, c in subspace_rank_histograms(basis, E, truths, rng=rng).items()
            },
        }
        for label, obs in (("clean", truths), ("outliers", contaminated)):
            r[f"scores_{label}"] = {
                "mse_of_mean": float(np.mean(mse(mean, obs))),
                "crps": float(np.mean(crps_ensemble(E, obs))),
                "kernel": float(np.mean(kernel_score_ensemble(E, obs, width))),
            }
        res[name] = r

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=2))
    _print_summary(res)
    if not a.no_plot:
        _plot(a, h, basis, truths, ens, res)
    print(f"done {time.time() - t0:.1f}s -> {a.out}")


def _bands(basis, x, anomaly, n, edges):
    parts = basis.split(x, anomaly=anomaly)
    fr = band_fractions(parts, (n, n), edges)
    return {k: np.mean(v, axis=tuple(range(v.ndim - 1))).tolist() for k, v in fr.items()}


def _print_summary(res):
    d, f = res["dims"], res["dofs"]
    print(f"\nsubspace dims {d}")
    print(
        f"DOFS data {f['data']:.1f}, +coupling {f['data_plus_coupling']:.1f} "
        f"(converted {f['converted']:.1f}; {100 * f['prior_fill_dims_converted_fraction']:.1f}% of prior-fill dims)"
    )
    t = res["energy"]["truth"]["physical"]
    print("truth physical energy: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in t.items()))
    hdr = f"{'sampler':<14}{'subspace':<9}{'spread/skill':>13}{'rank chi2':>11}"
    print("\n" + hdr)
    for name in ("linear_only", "coupled", "overconfident"):
        for s, v in res[name]["spread_skill"].items():
            print(f"{name:<14}{s:<9}{v['ratio']:>13.2f}{res[name]['rank_flatness'].get(s, np.nan):>11.2f}")
    print(f"\n{'sampler':<14}{'obs':<9}{'MSE':>8}{'CRPS':>8}{'kernel':>9}")
    for name in ("linear_only", "coupled", "overconfident"):
        for label in ("clean", "outliers"):
            s = res[name][f"scores_{label}"]
            print(f"{name:<14}{label:<9}{s['mse_of_mean']:>8.3f}{s['crps']:>8.3f}{s['kernel']:>9.4f}")


def _plot(a, h, basis, truths, ens, res):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    n = a.n
    member = ens["coupled"][0, 0]
    parts = basis.split(member)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.5), layout="constrained")
    panels = [
        (h, "terrain [km]", "terrain"),
        (truths[0], "truth anomaly [K]", "RdBu_r"),
        (member, "coupled sampler, member 1 [K]", "RdBu_r"),
        (ens["coupled"][:, 0].std(axis=0), "ensemble spread [K]", "viridis"),
        (parts["data"], "data-trusted part [K]", "RdBu_r"),
        (parts["coupled"], "coupling-trusted part [K]", "RdBu_r"),
        (parts["prior"], "prior-fill part [K]", "RdBu_r"),
    ]
    vmax = np.abs(truths[0]).max()
    for axis, (f, title, cmap) in zip(ax.flat, panels):
        kw = dict(vmin=-vmax, vmax=vmax) if cmap == "RdBu_r" else {}
        im = axis.imshow(f.reshape(n, n), origin="lower", cmap=cmap, **kw)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(im, ax=axis, shrink=0.8)
    axis = ax.flat[7]
    samplers = ["linear_only", "coupled", "overconfident"]
    subspaces = ["data", "coupled", "prior"]
    width = 0.25
    for i, s in enumerate(subspaces):
        vals = [res[m]["spread_skill"].get(s, {}).get("ratio", np.nan) for m in samplers]
        axis.bar(np.arange(3) + (i - 1) * width, vals, width, label=s)
    axis.axhline(1.0, color="k", lw=0.8)
    axis.set_xticks(np.arange(3), samplers, fontsize=9)
    axis.set_ylabel("spread / skill")
    axis.set_title("calibration per subspace", fontsize=10)
    axis.legend(fontsize=8, loc="lower left")
    fig.savefig(a.out / "toy_overview.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

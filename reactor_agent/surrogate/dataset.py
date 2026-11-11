"""Generate the surrogate training set by sampling the exact adiabatic PFR model.

Inputs:  T0 [1000, 1700] K, P [1, 10] atm, phi [0.5, 1.5], log10(tau) [-4.5, -0.5]
Outputs: CH4 conversion, outlet temperature
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..sim import PFRInput, run_pfr

BOUNDS = {
    "T_K": (1000.0, 1700.0),
    "P_atm": (1.0, 10.0),
    "phi": (0.5, 1.5),
    "log10_tau": (-4.5, -0.5),
}
DATA_PATH = Path(__file__).with_name("data.npz")


def sample_inputs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = [rng.uniform(lo, hi, n) for lo, hi in BOUNDS.values()]
    return np.stack(cols, axis=1)


def _run(row: np.ndarray) -> tuple[float, float]:
    T, P, phi, log_tau = row
    try:
        r = run_pfr(PFRInput(T_K=T, P_atm=P, phi=phi, residence_time_s=10**log_tau))
        return r.conversion_CH4, r.T_out_K
    except Exception:
        return np.nan, np.nan


def build(n: int = 4000, seed: int = 0, workers: int | None = None, out: Path = DATA_PATH) -> Path:
    X = sample_inputs(n, seed)
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        Y = np.asarray(pool.map(_run, X, chunksize=16))
    ok = ~np.isnan(Y).any(axis=1)
    np.savez(out, X=X[ok], Y=Y[ok], bounds=np.asarray(list(BOUNDS.values())))
    print(f"saved {ok.sum()}/{n} samples to {out} in {time.perf_counter() - t0:.0f} s")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    build(a.n, a.seed, a.workers)

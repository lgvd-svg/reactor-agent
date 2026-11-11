"""Build the golden eval set: random tasks with reference answers computed by bisection on
the exact simulator. Run once; the output is committed so evals are reproducible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..sim import BatchInput, EquilibriumInput, PFRInput, equilibrium, run_batch, run_pfr
from ..tasks import AdiabaticFlameTemp, MinPyrolysisTemperature, MinResidenceTime, Task

CASES_PATH = Path(__file__).with_name("cases.json")


def _bisect(f, lo: float, hi: float, target: float, iters: int = 30, log: bool = False) -> float:
    """Smallest x in [lo, hi] with f(x) >= target, assuming f is non-decreasing."""
    assert f(hi) >= target, "target unreachable at upper bound"
    for _ in range(iters):
        mid = np.sqrt(lo * hi) if log else 0.5 * (lo + hi)
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def golden_min_tau(spec: MinResidenceTime) -> dict:
    def conv(tau):
        return run_pfr(PFRInput(T_K=spec.T_K, P_atm=spec.P_atm, phi=spec.phi, residence_time_s=tau)).conversion_CH4

    tau = _bisect(conv, 1e-5, 1.0, spec.target_conversion, iters=25, log=True)
    r = run_pfr(PFRInput(T_K=spec.T_K, P_atm=spec.P_atm, phi=spec.phi, residence_time_s=tau))
    return {"residence_time_s": tau, "conversion_CH4": r.conversion_CH4, "T_max_K": r.T_max_K}


def golden_min_T(spec: MinPyrolysisTemperature) -> dict:
    def conv(T):
        return run_batch(BatchInput(T_K=T, P_atm=spec.P_atm, phi=None, t_end_s=spec.t_end_s, isothermal=True)).conversion_CH4

    T = _bisect(conv, 900.0, spec.T_max_K, spec.target_conversion, iters=22)
    return {"T_K": T, "conversion_CH4": conv(T)}


def golden_tad(spec: AdiabaticFlameTemp) -> dict:
    return {"T_K": equilibrium(EquilibriumInput(T_K=spec.T_K, P_atm=spec.P_atm, phi=spec.phi)).T_K}


def make(n_per_type: int = 10, seed: int = 42) -> list[Task]:
    rng = np.random.default_rng(seed)
    tasks: list[Task] = []
    for i in range(n_per_type):
        # Round so the prompt ("1421 K", "phi 1.129") and the spec carry identical numbers;
        # near the ignition front a 0.3 K difference changes the answer.
        spec = MinResidenceTime(
            T_K=float(round(rng.uniform(1150, 1500))),
            P_atm=float(rng.choice([1.0, 2.0, 5.0])),
            phi=round(float(rng.uniform(0.7, 1.2)), 3),
            target_conversion=float(rng.choice([0.5, 0.8, 0.9, 0.95])),
        )
        tasks.append(Task(id=f"tau-{i:02d}", spec=spec, golden=golden_min_tau(spec)))
    for i in range(n_per_type):
        t_end = float(rng.choice([0.1, 0.5, 1.0, 2.0]))
        target = float(rng.choice([0.1, 0.2, 0.3, 0.5]))
        spec = MinPyrolysisTemperature(P_atm=float(rng.choice([1.0, 3.0])), t_end_s=t_end,
                                       target_conversion=target, T_max_K=2000.0)
        try:
            g = golden_min_T(spec)
        except AssertionError:
            continue
        tasks.append(Task(id=f"pyro-{i:02d}", spec=spec, golden=g))
    for i in range(max(3, n_per_type // 2)):
        spec = AdiabaticFlameTemp(T_K=float(rng.choice([300.0, 500.0, 800.0])),
                                  P_atm=float(rng.choice([1.0, 5.0, 20.0])),
                                  phi=round(float(rng.uniform(0.6, 1.4)), 3))
        tasks.append(Task(id=f"tad-{i:02d}", spec=spec, golden=golden_tad(spec)))
    return tasks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="cases per task type")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    tasks = make(a.n, a.seed)
    CASES_PATH.write_text(json.dumps([t.model_dump() for t in tasks], indent=1))
    print(f"wrote {len(tasks)} cases to {CASES_PATH}")

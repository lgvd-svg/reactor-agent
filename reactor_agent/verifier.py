"""Independent verifier. It never trusts the LLM: it re-runs the exact simulator on the
submitted design and checks (1) the claimed numbers match, (2) constraints hold,
(3) the design is near-minimal when the task asks for a minimum."""

from __future__ import annotations

from pydantic import BaseModel

from .sim import BatchInput, EquilibriumInput, PFRInput, equilibrium, run_batch, run_pfr
from .tasks import AdiabaticFlameTemp, MinPyrolysisTemperature, MinResidenceTime, TaskSpec
from .tools import SubmitAnswer

REL_TOL_CLAIM = 0.02  # claimed vs recomputed outputs
MINIMALITY_SHRINK = 0.85  # design shrunk by 15% must violate the target to count as "minimal"
MINIMALITY_TEMP_STEP_K = 25.0


class Verdict(BaseModel):
    accepted: bool
    constraints_ok: bool
    claims_ok: bool
    minimal_ok: bool | None  # None when the task has no minimisation objective
    recomputed: dict[str, float]
    reasons: list[str]


def _close(a: float, b: float, rel: float, abs_floor: float = 1e-3) -> bool:
    return abs(a - b) <= max(rel * abs(b), abs_floor)


def verify(spec: TaskSpec, answer: SubmitAnswer) -> Verdict:
    reasons: list[str] = []
    if isinstance(spec, MinResidenceTime):
        tau = answer.design.get("residence_time_s")
        if tau is None or tau <= 0:
            return Verdict(accepted=False, constraints_ok=False, claims_ok=False, minimal_ok=None,
                           recomputed={}, reasons=["design lacks residence_time_s"])
        res = run_pfr(PFRInput(T_K=spec.T_K, P_atm=spec.P_atm, phi=spec.phi, residence_time_s=tau))
        recomputed = {"conversion_CH4": res.conversion_CH4, "T_max_K": res.T_max_K}
        constraints_ok = res.conversion_CH4 >= spec.target_conversion - 1e-6
        if not constraints_ok:
            reasons.append(f"conversion {res.conversion_CH4:.4f} < target {spec.target_conversion}")
        if spec.T_max_K and res.T_max_K > spec.T_max_K:
            constraints_ok = False
            reasons.append(f"T_max {res.T_max_K:.0f} K > limit {spec.T_max_K:.0f} K")
        smaller = run_pfr(PFRInput(T_K=spec.T_K, P_atm=spec.P_atm, phi=spec.phi,
                                   residence_time_s=tau * MINIMALITY_SHRINK))
        minimal_ok = smaller.conversion_CH4 < spec.target_conversion
        if not minimal_ok:
            reasons.append(
                f"not minimal: {MINIMALITY_SHRINK:.0%} of tau still gives conversion {smaller.conversion_CH4:.4f}"
            )
    elif isinstance(spec, MinPyrolysisTemperature):
        T = answer.design.get("T_K")
        if T is None or T <= 0:
            return Verdict(accepted=False, constraints_ok=False, claims_ok=False, minimal_ok=None,
                           recomputed={}, reasons=["design lacks T_K"])
        res = run_batch(BatchInput(T_K=T, P_atm=spec.P_atm, phi=None, t_end_s=spec.t_end_s, isothermal=True))
        recomputed = {"conversion_CH4": res.conversion_CH4}
        constraints_ok = res.conversion_CH4 >= spec.target_conversion - 1e-6 and T <= spec.T_max_K
        if res.conversion_CH4 < spec.target_conversion:
            reasons.append(f"conversion {res.conversion_CH4:.4f} < target {spec.target_conversion}")
        if T > spec.T_max_K:
            reasons.append(f"T {T:.0f} K exceeds limit {spec.T_max_K:.0f} K")
        cooler = run_batch(BatchInput(T_K=T - MINIMALITY_TEMP_STEP_K, P_atm=spec.P_atm, phi=None,
                                      t_end_s=spec.t_end_s, isothermal=True))
        minimal_ok = cooler.conversion_CH4 < spec.target_conversion
        if not minimal_ok:
            reasons.append(f"not minimal: T - {MINIMALITY_TEMP_STEP_K:.0f} K still meets the target")
    elif isinstance(spec, AdiabaticFlameTemp):
        eq = equilibrium(EquilibriumInput(T_K=spec.T_K, P_atm=spec.P_atm, phi=spec.phi))
        recomputed = {"T_K": eq.T_K}
        constraints_ok = True
        minimal_ok = None
    else:  # pragma: no cover
        raise TypeError(f"unknown spec {type(spec)}")

    claims_ok = True
    for key, claimed in answer.claimed.items():
        if key in recomputed and not _close(claimed, recomputed[key], REL_TOL_CLAIM):
            claims_ok = False
            reasons.append(f"claimed {key}={claimed:.4g} but simulator gives {recomputed[key]:.4g}")
    for key in recomputed:
        if key not in answer.claimed:
            claims_ok = False
            reasons.append(f"missing claim for {key}")

    accepted = constraints_ok and claims_ok and (minimal_ok is not False)
    return Verdict(accepted=accepted, constraints_ok=constraints_ok, claims_ok=claims_ok,
                   minimal_ok=minimal_ok, recomputed=recomputed, reasons=reasons)

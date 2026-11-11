from reactor_agent.sim import PFRInput, run_pfr
from reactor_agent.tasks import AdiabaticFlameTemp, MinPyrolysisTemperature, MinResidenceTime
from reactor_agent.tools import SubmitAnswer
from reactor_agent.verifier import verify

SPEC = MinResidenceTime(T_K=1200, P_atm=1, phi=1.0, target_conversion=0.9)


def _answer(tau: float, conv: float, tmax: float, ok: bool = True) -> SubmitAnswer:
    return SubmitAnswer(
        design={"residence_time_s": tau},
        claimed={"conversion_CH4": conv, "T_max_K": tmax},
        constraints_met=ok,
        justification="test",
    )


def test_accepts_honest_near_minimal_design():
    r = run_pfr(PFRInput(T_K=1200, P_atm=1, phi=1.0, residence_time_s=0.048))
    v = verify(SPEC, _answer(0.048, r.conversion_CH4, r.T_max_K))
    assert v.accepted, v.reasons


def test_rejects_fabricated_numbers():
    # Design is fine, but the claimed conversion is invented.
    v = verify(SPEC, _answer(0.048, 0.95, 2500.0))
    assert not v.accepted and not v.claims_ok
    assert any("claimed" in r for r in v.reasons)


def test_rejects_design_that_misses_target():
    r = run_pfr(PFRInput(T_K=1200, P_atm=1, phi=1.0, residence_time_s=0.03))
    v = verify(SPEC, _answer(0.03, r.conversion_CH4, r.T_max_K, ok=True))
    assert not v.accepted and not v.constraints_ok


def test_flags_non_minimal_design():
    r = run_pfr(PFRInput(T_K=1200, P_atm=1, phi=1.0, residence_time_s=0.5))
    v = verify(SPEC, _answer(0.5, r.conversion_CH4, r.T_max_K))
    assert v.constraints_ok and v.claims_ok
    assert v.minimal_ok is False and not v.accepted


def test_pyrolysis_temperature_limit_enforced():
    spec = MinPyrolysisTemperature(P_atm=1, t_end_s=1.0, target_conversion=0.2, T_max_K=1350)
    v = verify(spec, SubmitAnswer(design={"T_K": 1400}, claimed={"conversion_CH4": 0.25},
                                  constraints_met=True, justification="t"))
    assert not v.accepted and any("exceeds limit" in r for r in v.reasons)


def test_flame_temperature_claim_checked():
    spec = AdiabaticFlameTemp(T_K=300, P_atm=1, phi=1.0)
    good = verify(spec, SubmitAnswer(design={}, claimed={"T_K": 2225.5}, constraints_met=True, justification="t"))
    bad = verify(spec, SubmitAnswer(design={}, claimed={"T_K": 2100.0}, constraints_met=True, justification="t"))
    assert good.accepted and not bad.accepted

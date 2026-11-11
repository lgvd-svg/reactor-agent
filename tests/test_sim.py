import pytest
from pydantic import ValidationError

from reactor_agent.sim import (
    BatchInput,
    CSTRInput,
    EquilibriumInput,
    PFRInput,
    equilibrium,
    run_batch,
    run_cstr,
    run_pfr,
)


def test_adiabatic_flame_temperature_stoichiometric_methane():
    # Textbook value for CH4/air at 300 K, 1 atm is ~2225 K (GRI-Mech, HP equilibrium).
    T = equilibrium(EquilibriumInput(T_K=300, P_atm=1, phi=1.0)).T_K
    assert 2200 < T < 2250


def test_pfr_conversion_is_monotonic_in_residence_time():
    taus = [0.02, 0.04, 0.06, 0.1]
    convs = [run_pfr(PFRInput(T_K=1200, P_atm=1, phi=1.0, residence_time_s=t)).conversion_CH4 for t in taus]
    assert all(b >= a - 1e-9 for a, b in zip(convs, convs[1:], strict=False))
    assert convs[-1] > 0.99


def test_pfr_reports_ignition_delay_when_ignited():
    r = run_pfr(PFRInput(T_K=1200, P_atm=1, phi=1.0, residence_time_s=0.1))
    assert r.ignited and r.ignition_delay_s is not None
    assert 0.03 < r.ignition_delay_s < 0.06
    short = run_pfr(PFRInput(T_K=1200, P_atm=1, phi=1.0, residence_time_s=0.02))
    assert not short.ignited and short.ignition_delay_s is None


def test_isothermal_pyrolysis_has_no_temperature_rise():
    r = run_batch(BatchInput(T_K=1400, P_atm=1, phi=None, t_end_s=1.0, isothermal=True, include_composition=True))
    assert abs(r.T_out_K - 1400) < 1e-6
    assert 0.1 < r.conversion_CH4 < 0.6
    assert r.outlet.O2 == 0.0


def test_cstr_burning_branch_gives_high_conversion():
    r = run_cstr(CSTRInput(T_K=600, P_atm=1, phi=0.8, residence_time_s=0.05))
    assert r.conversion_CH4 > 0.95 and r.T_out_K > 2000


def test_input_validation_rejects_out_of_range():
    with pytest.raises(ValidationError):
        PFRInput(T_K=100, P_atm=1, phi=1.0, residence_time_s=0.1)

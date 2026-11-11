"""Cantera-backed reactor models for CH4/air (GRI-Mech 3.0).

All functions are pure: build a fresh gas object, integrate, return a typed result.
"""

from __future__ import annotations

import time
import warnings

import cantera as ct
import numpy as np

from .models import (
    BatchInput,
    Composition,
    CSTRInput,
    EquilibriumInput,
    EquilibriumResult,
    PFRInput,
    ReactorResult,
)

MECHANISM = "gri30.yaml"
OXIDIZER = "O2:1, N2:3.76"
SPECIES = ["CH4", "O2", "CO", "CO2", "H2O", "H2", "NO", "C2H2"]
IGNITION_DT_K = 400.0  # temperature rise that counts as thermal runaway

warnings.filterwarnings("ignore", category=DeprecationWarning, module="cantera")


def _make_gas(T_K: float, P_atm: float, phi: float | None) -> ct.Solution:
    gas = ct.Solution(MECHANISM)
    if phi is None:
        gas.TPX = T_K, P_atm * ct.one_atm, "CH4:1"
    else:
        gas.TP = T_K, P_atm * ct.one_atm
        gas.set_equivalence_ratio(phi, "CH4", OXIDIZER)
    return gas


def _composition(gas: ct.Solution) -> Composition:
    return Composition(**{s: float(gas[s].X[0]) for s in SPECIES})


def equilibrium(inp: EquilibriumInput) -> EquilibriumResult:
    gas = _make_gas(inp.T_K, inp.P_atm, inp.phi)
    gas.equilibrate("HP")
    return EquilibriumResult(T_K=float(gas.T), composition=_composition(gas))


def _integrate_const_pressure(
    T_K: float, P_atm: float, phi: float | None, t_end: float, isothermal: bool, label: str,
    include_composition: bool = True,
) -> ReactorResult:
    t0 = time.perf_counter()
    gas = _make_gas(T_K, P_atm, phi)
    x_ch4_in = float(gas["CH4"].X[0])
    reactor = ct.IdealGasConstPressureReactor(
        gas, energy="off" if isothermal else "on", clone=False
    )
    net = ct.ReactorNet([reactor])
    net.max_steps = 100_000

    # Geometric time grid: ~2% relative resolution on ignition delay, exact landing on t_end.
    grid = np.geomspace(t_end * 1e-4, t_end, 500)
    times = [0.0]
    temps = [reactor.T]
    for t in grid:
        net.advance(float(t))
        times.append(float(t))
        temps.append(reactor.T)

    phase = reactor.phase
    temps_arr = np.asarray(temps)
    times_arr = np.asarray(times)
    T_max = float(temps_arr.max())
    ignited = (T_max - T_K) > IGNITION_DT_K
    delay = None
    if ignited and len(times_arr) > 2:
        dTdt = np.gradient(temps_arr, times_arr, edge_order=1)
        delay = float(times_arr[int(np.argmax(dTdt))])

    x_ch4_out = float(phase["CH4"].X[0])
    return ReactorResult(
        reactor=label,
        conversion_CH4=max(0.0, 1.0 - x_ch4_out / x_ch4_in),
        T_out_K=float(reactor.T),
        T_max_K=T_max,
        ignited=ignited,
        ignition_delay_s=delay,
        outlet=_composition(phase) if include_composition else None,
        wall_time_s=time.perf_counter() - t0,
    )


def run_batch(inp: BatchInput) -> ReactorResult:
    """Constant-pressure batch reactor (adiabatic or isothermal)."""
    return _integrate_const_pressure(
        inp.T_K, inp.P_atm, inp.phi, inp.t_end_s, inp.isothermal, "batch_const_P",
        inp.include_composition,
    )


def run_pfr(inp: PFRInput) -> ReactorResult:
    """Plug-flow reactor in the Lagrangian frame: a fluid element moving down the tube sees
    the same history as a constant-pressure batch reactor, so residence time plays the role
    of reaction time. Valid for ideal plug flow (no axial mixing, negligible pressure drop)."""
    return _integrate_const_pressure(
        inp.T_K, inp.P_atm, inp.phi, inp.residence_time_s, inp.isothermal, "pfr_lagrangian",
        inp.include_composition,
    )


def run_cstr(inp: CSTRInput) -> ReactorResult:
    """Adiabatic continuous stirred-tank reactor at steady state.

    The reactor is initialised at the equilibrium (ignited) state so the integrator
    converges to the burning branch when one exists. If the residence time is too short
    the flame blows out and the result shows conversion near zero."""
    t0 = time.perf_counter()
    inlet_gas = _make_gas(inp.T_K, inp.P_atm, inp.phi)
    x_ch4_in = float(inlet_gas["CH4"].X[0])
    inlet = ct.Reservoir(inlet_gas)

    gas = _make_gas(inp.T_K, inp.P_atm, inp.phi)
    gas.equilibrate("HP")
    reactor = ct.IdealGasReactor(gas, clone=False)
    reactor.volume = 1.0
    exhaust = ct.Reservoir(gas)

    mass_flow = reactor.mass / inp.residence_time_s
    ct.MassFlowController(inlet, reactor, mdot=mass_flow)
    # A valve keeps the reactor near the inlet pressure while letting mass out.
    ct.Valve(reactor, exhaust, K=1.0)

    net = ct.ReactorNet([reactor])
    net.max_steps = 100_000
    net.advance_to_steady_state()

    phase = reactor.phase
    x_ch4_out = float(phase["CH4"].X[0])
    T_out = float(reactor.T)
    return ReactorResult(
        reactor="cstr_steady_state",
        conversion_CH4=max(0.0, 1.0 - x_ch4_out / x_ch4_in),
        T_out_K=T_out,
        T_max_K=T_out,
        ignited=(T_out - inp.T_K) > IGNITION_DT_K,
        ignition_delay_s=None,
        outlet=_composition(phase) if inp.include_composition else None,
        wall_time_s=time.perf_counter() - t0,
    )

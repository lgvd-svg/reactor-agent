"""Typed inputs and outputs for the simulator tools. Units are in field names."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Feed(BaseModel):
    T_K: float = Field(..., ge=300, le=3000, description="Initial / inlet temperature in kelvin.")
    P_atm: float = Field(1.0, ge=0.1, le=100, description="Pressure in atm (constant).")
    phi: float | None = Field(
        1.0,
        ge=0.2,
        le=5.0,
        description=(
            "Fuel/air equivalence ratio for CH4 in air (O2:1, N2:3.76). "
            "Use null for pure CH4 (pyrolysis, no oxygen)."
        ),
    )
    include_composition: bool = Field(
        False, description="Also return outlet mole fractions (CO, CO2, H2, NO, C2H2...)."
    )


class EquilibriumInput(Feed):
    pass


class BatchInput(Feed):
    t_end_s: float = Field(..., gt=0, le=100, description="Reaction time in seconds.")
    isothermal: bool = Field(
        False, description="True: hold temperature constant. False: adiabatic, constant pressure."
    )


class PFRInput(Feed):
    residence_time_s: float = Field(..., gt=0, le=100, description="Mean residence time in seconds.")
    isothermal: bool = Field(
        False, description="True: hold temperature constant. False: adiabatic, constant pressure."
    )


class CSTRInput(Feed):
    residence_time_s: float = Field(..., gt=0, le=100, description="Mean residence time in seconds.")


class Composition(BaseModel):
    CH4: float
    O2: float
    CO: float
    CO2: float
    H2O: float
    H2: float
    NO: float
    C2H2: float


class EquilibriumResult(BaseModel):
    T_K: float = Field(..., description="Equilibrium temperature (adiabatic flame temperature).")
    composition: Composition
    note: str = "Constant enthalpy and pressure (HP) equilibrium; the upper bound for any reactor."


class ReactorResult(BaseModel):
    reactor: str
    conversion_CH4: float = Field(..., description="1 - X_CH4,out / X_CH4,in. Zero for pyrolysis of pure CH4 means nothing reacted.")
    T_out_K: float
    T_max_K: float
    ignited: bool
    ignition_delay_s: float | None = Field(
        None, description="Time of maximum temperature rise rate; null if no thermal runaway."
    )
    outlet: Composition | None = None
    wall_time_s: float

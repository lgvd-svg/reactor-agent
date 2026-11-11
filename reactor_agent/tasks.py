"""Task specifications: a natural-language goal plus a machine-checkable spec so the
verifier can judge the agent's answer independently of the LLM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MinResidenceTime(BaseModel):
    """Find the smallest adiabatic-PFR residence time that reaches a CH4 conversion target."""

    kind: Literal["min_residence_time"] = "min_residence_time"
    T_K: float
    P_atm: float
    phi: float
    target_conversion: float = Field(..., gt=0, lt=1)
    T_max_K: float | None = None  # optional peak-temperature limit

    def prompt(self) -> str:
        s = (
            f"Design an adiabatic plug-flow reactor for CH4/air at inlet {self.T_K:.0f} K, "
            f"{self.P_atm:g} atm, equivalence ratio {self.phi:g}. Find the SMALLEST residence time "
            f"(in seconds) that achieves CH4 conversion >= {self.target_conversion:.2f}."
        )
        if self.T_max_K:
            s += f" The peak temperature must stay below {self.T_max_K:.0f} K."
        return s + " Report residence_time_s in `design` and conversion_CH4 and T_max_K in `claimed`."


class MinPyrolysisTemperature(BaseModel):
    """Find the lowest isothermal temperature that reaches a CH4 conversion target in pyrolysis."""

    kind: Literal["min_pyrolysis_temperature"] = "min_pyrolysis_temperature"
    P_atm: float
    t_end_s: float
    target_conversion: float = Field(..., gt=0, lt=1)
    T_max_K: float

    def prompt(self) -> str:
        return (
            f"Pure methane pyrolysis (no oxygen: phi = null) in an isothermal batch reactor at "
            f"{self.P_atm:g} atm for {self.t_end_s:g} s. Find the LOWEST temperature (K) that gives "
            f"CH4 conversion >= {self.target_conversion:.2f}, and it must not exceed {self.T_max_K:.0f} K. "
            "Report T_K in `design` and conversion_CH4 in `claimed`."
        )


class AdiabaticFlameTemp(BaseModel):
    """Report the adiabatic flame temperature."""

    kind: Literal["adiabatic_flame_temp"] = "adiabatic_flame_temp"
    T_K: float
    P_atm: float
    phi: float

    def prompt(self) -> str:
        return (
            f"What is the adiabatic flame temperature of CH4/air at {self.T_K:.0f} K initial temperature, "
            f"{self.P_atm:g} atm, equivalence ratio {self.phi:g}? Report T_K in `claimed` "
            "(design may be empty)."
        )


TaskSpec = MinResidenceTime | MinPyrolysisTemperature | AdiabaticFlameTemp


class Task(BaseModel):
    id: str
    spec: TaskSpec = Field(..., discriminator="kind")
    golden: dict[str, float] | None = None  # filled by evals/make_cases.py

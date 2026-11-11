"""Tool registry: Pydantic input models become OpenAI function schemas; calls are validated
and dispatched here, so the LLM can only reach the simulator through typed, range-checked
arguments."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .sim import (
    BatchInput,
    CSTRInput,
    EquilibriumInput,
    PFRInput,
    equilibrium,
    run_batch,
    run_cstr,
    run_pfr,
)


class SubmitAnswer(BaseModel):
    """Final answer. Every number here must come from a simulator tool result in this
    conversation, never from memory or estimation."""

    design: dict[str, float] = Field(
        ..., description="Design variables you chose, e.g. {'residence_time_s': 0.047} or {'T_K': 1380}."
    )
    claimed: dict[str, float] = Field(
        ...,
        description=(
            "Outputs you claim for that design, copied from the exact simulator tool "
            "(not the surrogate): e.g. {'conversion_CH4': 0.93, 'T_max_K': 2610}."
        ),
    )
    constraints_met: bool = Field(..., description="Your own assessment of whether all constraints hold.")
    justification: str = Field(..., max_length=1200, description="Two to four sentences: what you tried and why this design.")


@dataclass
class Tool:
    name: str
    description: str
    model: type[BaseModel]
    fn: Callable[[BaseModel], Any] | None  # None for submit_answer

    def schema(self) -> dict:
        params = self.model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


def _surrogate_tool() -> Tool | None:
    try:
        from .surrogate.predict import SurrogateInput, is_available, predict
    except Exception:  # surrogate deps missing
        return None
    if not is_available():
        return None
    return Tool(
        name="surrogate_predict",
        description=(
            "FAST APPROXIMATE prediction (neural surrogate, ~1 ms) of adiabatic PFR CH4 conversion "
            "and outlet temperature. Use it to explore many residence times quickly, then CONFIRM "
            "the chosen design with run_pfr before submitting. Never cite surrogate numbers as final."
        ),
        model=SurrogateInput,
        fn=predict,
    )


def build_registry(include_surrogate: bool = True) -> dict[str, Tool]:
    tools = [
        Tool(
            "equilibrium",
            "Chemical equilibrium at constant enthalpy and pressure for CH4/air (or pure CH4 if phi is null). "
            "Returns the adiabatic flame temperature and equilibrium composition: the theoretical limit "
            "any reactor can approach, useful to check whether a target is physically reachable.",
            EquilibriumInput,
            equilibrium,
        ),
        Tool(
            "run_batch",
            "Constant-pressure batch reactor with GRI-Mech 3.0 kinetics. Adiabatic by default; set "
            "isothermal=true to hold temperature (e.g. pyrolysis of pure CH4 with phi=null).",
            BatchInput,
            run_batch,
        ),
        Tool(
            "run_pfr",
            "Ideal plug-flow reactor (Lagrangian frame) with GRI-Mech 3.0 kinetics. Reports CH4 conversion, "
            "outlet and peak temperature, whether the mixture ignited and the ignition delay. "
            "Adiabatic by default; isothermal=true holds temperature.",
            PFRInput,
            run_pfr,
        ),
        Tool(
            "run_cstr",
            "Adiabatic continuous stirred-tank reactor at steady state (burning branch). Short residence "
            "times blow out the flame and give low conversion.",
            CSTRInput,
            run_cstr,
        ),
        Tool(
            "submit_answer",
            "Submit the final design. Call this exactly once, after confirming the design with an exact "
            "simulator tool. All numbers must be copied from tool results.",
            SubmitAnswer,
            None,
        ),
    ]
    if include_surrogate:
        s = _surrogate_tool()
        if s is not None:
            tools.insert(4, s)
    return {t.name: t for t in tools}


def dispatch(registry: dict[str, Tool], name: str, raw_args: str | dict) -> dict:
    """Validate and run a tool call. Errors are returned as data so the model can recover."""
    tool = registry.get(name)
    if tool is None:
        return {"error": f"unknown tool '{name}'", "available": sorted(registry)}
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        inp = tool.model.model_validate(args)
    except (json.JSONDecodeError, ValidationError) as e:
        return {"error": "invalid arguments", "detail": str(e)[:800]}
    if tool.fn is None:
        return {"submitted": inp.model_dump()}
    try:
        out = tool.fn(inp)
    except Exception as e:  # simulator failure (stiff case, blow-out, etc.)
        return {"error": f"simulation failed: {type(e).__name__}", "detail": str(e)[:800]}
    return out.model_dump() if isinstance(out, BaseModel) else out

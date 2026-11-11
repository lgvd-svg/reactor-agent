"""Offline stand-in for the LLM: a scripted engineer that drives the same tool loop.

It exists to exercise the agent loop, tool dispatch, verifier and eval runner end to end
without network or quota. Two modes:

- "oracle": careful engineer. Checks equilibrium, brackets the target by log-bisection,
  confirms with the exact tool and submits honest numbers. Should be accepted ~100 %.
- "sloppy": submits the first passing design it finds without narrowing the bracket and
  claims success. Constraints hold but the design is far from minimal, so the verifier
  should reject it: this is the "false accept" the evals are designed to surface.

Select with LLM_MODEL=oracle or LLM_MODEL=sloppy (any base URL / key is ignored).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _Call:
    id: str
    function: _Fn
    type: str = "function"


@dataclass
class _Msg:
    content: str | None = None
    tool_calls: list[_Call] = field(default_factory=list)


NUM = r"(\d+(?:\.\d+)?)"  # a number that does not swallow a sentence-ending period


def _num(pattern: str, text: str) -> float:
    m = re.search(pattern.replace("NUM", NUM), text)
    if not m:
        raise ValueError(f"could not parse {pattern!r} from prompt")
    return float(m.group(1))


class OracleLLM:
    """Implements the same `.chat(messages, tools)` surface as LLMClient."""

    def __init__(self, mode: str = "oracle") -> None:
        assert mode in {"oracle", "sloppy"}
        self.mode = mode
        self.model = mode
        self._calls = 0
        self._state: dict[str, Any] = {}

    # -- helpers ---------------------------------------------------------------------
    def _last_tool_result(self, messages: list[dict]) -> dict | None:
        for m in reversed(messages):
            if m["role"] == "tool":
                try:
                    return json.loads(m["content"])
                except json.JSONDecodeError:
                    return None
            if m["role"] == "user":
                return None
        return None

    def _call(self, name: str, args: dict) -> _Msg:
        self._calls += 1
        return _Msg(tool_calls=[_Call(f"c{self._calls}", _Fn(name, json.dumps(args)))])

    def _submit(self, design: dict, claimed: dict, ok: bool, why: str) -> _Msg:
        return self._call(
            "submit_answer",
            {"design": design, "claimed": claimed, "constraints_met": ok, "justification": why},
        )

    # -- main -------------------------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> _Msg:
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        if not self._state:
            self._state = self._parse(prompt)
        s = self._state
        last = self._last_tool_result(messages)
        kind = s["kind"]

        if kind == "tad":
            if last is None:
                return self._call("equilibrium", {"T_K": s["T_K"], "P_atm": s["P_atm"], "phi": s["phi"]})
            return self._submit({}, {"T_K": last["T_K"]}, True, "HP equilibrium from the simulator.")

        # Bracketing search on x (tau or T). State: lo (fails), hi (passes), pending x.
        if "pending" in s and last is not None and "conversion_CH4" in last:
            conv = last["conversion_CH4"]
            x = s.pop("pending")
            s["results"][x] = last
            if conv >= s["target"]:
                s["hi"] = x
            else:
                s["lo"] = x
            if self.mode == "sloppy" and s["hi"] is not None:
                r = s["results"][s["hi"]]
                return self._submit(*self._answer(s, s["hi"], r), True, "First design that met the target.")
            if s["hi"] is not None and s["lo"] is not None and self._tight(s):
                r = s["results"][s["hi"]]
                return self._submit(*self._answer(s, s["hi"], r), True,
                                    "Bracketed the target by bisection and confirmed with the exact tool.")

        x = self._next_x(s)
        if x is None:  # upper bound could not reach target: report infeasible
            best = s["hi"] or max(s["results"], default=None)
            if best is None:
                return _Msg(content="No feasible design found.")
            return self._submit(*self._answer(s, best, s["results"][best]), False,
                                "Target not reachable within the allowed range.")
        s["pending"] = x
        if kind == "tau":
            return self._call("run_pfr", {"T_K": s["T_K"], "P_atm": s["P_atm"], "phi": s["phi"],
                                          "residence_time_s": x})
        return self._call("run_batch", {"T_K": x, "P_atm": s["P_atm"], "phi": None,
                                        "t_end_s": s["t_end_s"], "isothermal": True})

    # -- search policy -------------------------------------------------------------------
    def _tight(self, s: dict) -> bool:
        if s["kind"] == "tau":
            return s["hi"] / s["lo"] < 1.05
        return s["hi"] - s["lo"] < 15.0

    def _next_x(self, s: dict) -> float | None:
        lo, hi = s["lo"], s["hi"]
        if hi is None:
            # Walk up from the upper probe until it passes.
            x = s["probe"]
            s["probe"] = x * 3 if s["kind"] == "tau" else x + 150
            if s["kind"] == "tau" and x > 1.0 or s["kind"] == "pyro" and x > s["T_cap"]:
                return None
            return min(x, s["T_cap"]) if s["kind"] == "pyro" else x
        if lo is None:
            x = hi / 3 if s["kind"] == "tau" else hi - 150
            return max(x, 900.0) if s["kind"] == "pyro" else x
        return (lo * hi) ** 0.5 if s["kind"] == "tau" else 0.5 * (lo + hi)

    def _answer(self, s: dict, x: float, r: dict) -> tuple[dict, dict]:
        if s["kind"] == "tau":
            return {"residence_time_s": x}, {"conversion_CH4": r["conversion_CH4"], "T_max_K": r["T_max_K"]}
        return {"T_K": x}, {"conversion_CH4": r["conversion_CH4"]}

    # -- prompt parsing ----------------------------------------------------------------
    def _parse(self, p: str) -> dict:
        if "adiabatic flame temperature" in p:
            return {"kind": "tad", "T_K": _num(r"at NUM K initial", p), "P_atm": _num(r"NUM atm", p),
                    "phi": _num(r"equivalence ratio NUM", p)}
        if "plug-flow" in p:
            return {"kind": "tau", "T_K": _num(r"inlet NUM K", p), "P_atm": _num(r"NUM atm", p),
                    "phi": _num(r"equivalence ratio NUM", p),
                    "target": _num(r"conversion >= NUM", p),
                    "lo": None, "hi": None, "probe": 1e-3, "results": {}}
        if "pyrolysis" in p:
            return {"kind": "pyro", "P_atm": _num(r"NUM atm", p), "t_end_s": _num(r"for NUM s", p),
                    "target": _num(r"conversion >= NUM", p), "T_cap": _num(r"not exceed NUM K", p),
                    "lo": None, "hi": None, "probe": 1300.0, "results": {}}
        raise ValueError("unrecognised task prompt")

"""Agent loop tests with a scripted fake LLM: no network, deterministic."""

import json
from dataclasses import dataclass, field

from reactor_agent.agent.loop import run_task
from reactor_agent.tasks import MinResidenceTime, Task
from reactor_agent.tools import build_registry, dispatch


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
    tool_calls: list = field(default_factory=list)


class ScriptedLLM:
    """Emits run_pfr for each scripted tau, then submits the last simulator result verbatim."""

    model = "fake"

    def __init__(self, taus):
        self.taus = list(taus)
        self.last_tau = None

    def chat(self, messages, tools=None):
        if self.taus:
            self.last_tau = self.taus.pop(0)
            args = json.dumps({"T_K": 1200, "P_atm": 1, "phi": 1.0, "residence_time_s": self.last_tau})
            return _Msg(tool_calls=[_Call("c1", _Fn("run_pfr", args))])
        last = next(json.loads(m["content"]) for m in reversed(messages) if m["role"] == "tool")
        sub = {
            "design": {"residence_time_s": self.last_tau},
            "claimed": {"conversion_CH4": last["conversion_CH4"], "T_max_K": last["T_max_K"]},
            "constraints_met": True,
            "justification": "scripted",
        }
        return _Msg(tool_calls=[_Call("c2", _Fn("submit_answer", json.dumps(sub)))])


TASK = Task(id="t", spec=MinResidenceTime(T_K=1200, P_atm=1, phi=1.0, target_conversion=0.9))


def test_agent_loop_runs_tools_and_verifies_good_answer():
    run = run_task(TASK, llm=ScriptedLLM([0.03, 0.06, 0.048]), include_surrogate=False)
    assert run.answer is not None and run.error is None
    assert [s.tool for s in run.steps] == ["run_pfr", "run_pfr", "run_pfr", "submit_answer"]
    assert run.verdict.accepted, run.verdict.reasons


def test_agent_loop_verifier_rejects_non_minimal_answer():
    run = run_task(TASK, llm=ScriptedLLM([0.5]), include_surrogate=False)
    assert run.answer is not None
    assert run.verdict.minimal_ok is False and not run.verdict.accepted


def test_dispatch_returns_errors_as_data():
    reg = build_registry(include_surrogate=False)
    assert "error" in dispatch(reg, "nope", "{}")
    assert "error" in dispatch(reg, "run_pfr", '{"T_K": 50}')
    assert "error" in dispatch(reg, "run_pfr", "not json")


def test_tool_schemas_are_openai_shaped():
    for t in build_registry(include_surrogate=False).values():
        s = t.schema()
        assert s["type"] == "function"
        assert s["function"]["parameters"]["type"] == "object"

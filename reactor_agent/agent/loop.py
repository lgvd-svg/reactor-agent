"""Agent loop: LLM proposes tool calls, we execute them against the simulator, the LLM
submits a structured answer, and the independent verifier judges it."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from ..tasks import Task
from ..tools import SubmitAnswer, build_registry, dispatch
from ..verifier import Verdict, verify
from .llm import LLMClient
from .prompts import NUDGE_SUBMIT, SYSTEM_PROMPT


class Step(BaseModel):
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]
    latency_s: float


class AgentRun(BaseModel):
    task_id: str
    model: str
    steps: list[Step]
    answer: SubmitAnswer | None
    verdict: Verdict | None
    final_text: str | None
    error: str | None
    tool_calls: int
    simulator_calls: int
    wall_time_s: float


def _round(obj: Any, sig: int = 4) -> Any:
    """Round floats to `sig` significant figures to keep the context small."""
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}") if obj != 0 else 0.0
    if isinstance(obj, dict):
        return {k: _round(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, sig) for v in obj]
    return obj


def _compact(result: dict, max_chars: int = 1200) -> str:
    slim = {k: v for k, v in _round(result).items() if v is not None and k != "wall_time_s"}
    s = json.dumps(slim, default=float, separators=(",", ":"))
    return s if len(s) <= max_chars else s[:max_chars] + "...(truncated)"


def run_task(
    task: Task,
    llm: LLMClient | None = None,
    max_rounds: int = 16,
    max_tool_calls: int = 14,
    include_surrogate: bool = True,
    verbose: bool = False,
) -> AgentRun:
    llm = llm or LLMClient()
    registry = build_registry(include_surrogate=include_surrogate)
    schemas = [t.schema() for t in registry.values()]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.spec.prompt()},
    ]
    steps: list[Step] = []
    answer: SubmitAnswer | None = None
    error: str | None = None
    final_text: str | None = None
    nudged = False
    t_start = time.perf_counter()

    for _ in range(max_rounds):
        try:
            msg = llm.chat(messages, tools=schemas)
        except Exception as e:
            error = f"llm error: {type(e).__name__}: {str(e)[:300]}"
            break

        tool_calls = msg.tool_calls or []
        if not tool_calls:
            final_text = msg.content
            if nudged:
                error = "model answered in prose instead of calling submit_answer"
                break
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": NUDGE_SUBMIT})
            nudged = True
            continue

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            name = tc.function.name
            t0 = time.perf_counter()
            if len(steps) >= max_tool_calls and name != "submit_answer":
                result = {"error": "tool budget exhausted; call submit_answer now"}
            else:
                result = dispatch(registry, name, tc.function.arguments)
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            steps.append(Step(tool=name, args=args, result=result, latency_s=time.perf_counter() - t0))
            if verbose:
                print(f"  -> {name}({json.dumps(args)}) = {_compact(result, 300)}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": _compact(result)})
            if name == "submit_answer" and "submitted" in result:
                try:
                    answer = SubmitAnswer.model_validate(result["submitted"])
                except ValidationError as e:  # pragma: no cover
                    error = f"bad submit_answer: {e}"
        if answer is not None or error:
            break
    else:
        error = "max rounds reached without submit_answer"

    verdict = verify(task.spec, answer) if answer is not None else None
    sim_calls = sum(1 for s in steps if s.tool in {"run_pfr", "run_batch", "run_cstr", "equilibrium"})
    return AgentRun(
        task_id=task.id,
        model=llm.model,
        steps=steps,
        answer=answer,
        verdict=verdict,
        final_text=final_text,
        error=error,
        tool_calls=len(steps),
        simulator_calls=sim_calls,
        wall_time_s=time.perf_counter() - t_start,
    )

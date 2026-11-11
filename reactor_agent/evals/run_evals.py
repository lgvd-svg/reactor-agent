"""Run the agent on the golden cases and report what the verifier catches.

Metrics:
- verified_success: answers the independent verifier accepted
- agent_claimed_success: answers where the agent said constraints_met=true
- false_accepts: agent claimed success but the verifier rejected (what you'd ship without a verifier)
- golden_gap: how far the agent's minimal design is from the bisection reference
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..agent import LLMClient, run_task
from ..tasks import Task
from .make_cases import CASES_PATH

RESULTS_PATH = Path(__file__).with_name("results.json")
SUMMARY_PATH = Path(__file__).with_name("RESULTS.md")


def _gap(task: Task, run) -> float | None:
    if not task.golden or run.answer is None:
        return None
    for key in ("residence_time_s", "T_K"):
        if key in task.golden and key in run.answer.design:
            g = task.golden[key]
            return (run.answer.design[key] - g) / g if key == "residence_time_s" else run.answer.design[key] - g
    if "T_K" in task.golden and "T_K" in run.answer.claimed:
        return (run.answer.claimed["T_K"] - task.golden["T_K"]) / task.golden["T_K"]
    return None


def main(limit: int | None, include_surrogate: bool, verbose: bool, only: str | None) -> None:
    tasks = [Task.model_validate(t) for t in json.loads(CASES_PATH.read_text())]
    if only:
        tasks = [t for t in tasks if t.id.startswith(only)]
    if limit:
        tasks = tasks[:limit]
    llm = LLMClient()
    rows = []
    t_start = time.perf_counter()
    for task in tasks:
        print(f"[{task.id}] {task.spec.prompt()[:90]}...")
        run = run_task(task, llm=llm, include_surrogate=include_surrogate, verbose=verbose)
        gap = _gap(task, run)
        row = {
            "task_id": task.id,
            "kind": task.spec.kind,
            "verified": bool(run.verdict and run.verdict.accepted),
            "agent_claimed_ok": bool(run.answer and run.answer.constraints_met),
            "reasons": run.verdict.reasons if run.verdict else [run.error or "no answer"],
            "golden_gap": gap,
            "tool_calls": run.tool_calls,
            "simulator_calls": run.simulator_calls,
            "wall_time_s": round(run.wall_time_s, 1),
            "error": run.error,
        }
        rows.append(row)
        status = "PASS" if row["verified"] else "FAIL"
        print(f"    {status}  calls={run.tool_calls}  gap={gap if gap is None else round(gap, 3)}  {row['reasons'][:2]}")

    n = len(rows)
    verified = sum(r["verified"] for r in rows)
    claimed = sum(r["agent_claimed_ok"] for r in rows)
    false_accepts = sum(1 for r in rows if r["agent_claimed_ok"] and not r["verified"])
    no_answer = sum(1 for r in rows if r["error"])
    summary = {
        "model": llm.model,
        "surrogate_enabled": include_surrogate,
        "n": n,
        "verified_success": verified,
        "agent_claimed_success": claimed,
        "false_accepts": false_accepts,
        "no_answer": no_answer,
        "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / max(n, 1), 1),
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    lines = [
        f"# Eval results — model `{llm.model}`, surrogate {'on' if include_surrogate else 'off'}",
        "",
        f"- Cases: **{n}**",
        f"- Verified success (independent verifier): **{verified}/{n}**",
        f"- Agent claimed success: {claimed}/{n}",
        f"- **False accepts** (agent said OK, verifier said no): **{false_accepts}**",
        f"- No answer / errors: {no_answer}",
        f"- Mean tool calls per task: {summary['mean_tool_calls']}",
        "",
        "| kind | n | verified | claimed | false accepts | median |gap| |",
        "|---|---|---|---|---|---|",
    ]
    for kind, rs in by_kind.items():
        gaps = sorted(abs(r["golden_gap"]) for r in rs if r["golden_gap"] is not None)
        med = gaps[len(gaps) // 2] if gaps else None
        lines.append(
            f"| {kind} | {len(rs)} | {sum(r['verified'] for r in rs)} | {sum(r['agent_claimed_ok'] for r in rs)} | "
            f"{sum(1 for r in rs if r['agent_claimed_ok'] and not r['verified'])} | "
            f"{'—' if med is None else f'{med:.3g}'} |"
        )
    lines += ["", "Gap: relative error in residence time, or kelvin for temperature tasks, versus the bisection reference.", ""]
    SUMMARY_PATH.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-surrogate", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--only", type=str, default=None, help="task id prefix, e.g. tau, pyro, tad")
    a = p.parse_args()
    main(a.limit, not a.no_surrogate, a.verbose, a.only)

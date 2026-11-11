"""Command-line entry point.

  reactor-agent ask "Design a PFR at 1300 K, 2 atm, phi 0.9 with conversion >= 0.95"
  reactor-agent task tau-00            # run one golden case by id
  reactor-agent evals [--limit N] [--no-surrogate] [--only tau]
  reactor-agent surrogate build|train
  reactor-agent cases                  # regenerate golden cases
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import LLMClient, run_task
from .tasks import Task


def _print_run(run) -> None:
    print(f"\nmodel: {run.model}   tool calls: {run.tool_calls}   wall: {run.wall_time_s:.1f}s")
    for s in run.steps:
        print(f"  {s.tool}({json.dumps(s.args)})")
    if run.answer:
        print("\nanswer:", json.dumps(run.answer.model_dump(), indent=1))
    if run.verdict:
        v = run.verdict
        print(f"\nverifier: {'ACCEPTED' if v.accepted else 'REJECTED'}  "
              f"constraints={v.constraints_ok} claims={v.claims_ok} minimal={v.minimal_ok}")
        for r in v.reasons:
            print("  -", r)
    if run.error:
        print("\nerror:", run.error)
    if run.final_text:
        print("\nmodel text:", run.final_text[:500])


def _free_form(goal: str, include_surrogate: bool) -> None:
    """Free-form question: no machine-checkable spec, so only the transcript is shown."""
    from .agent.loop import build_registry, dispatch
    from .agent.prompts import SYSTEM_PROMPT

    llm = LLMClient()
    registry = build_registry(include_surrogate)
    schemas = [t.schema() for t in registry.values()]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": goal}]
    for _ in range(16):
        msg = llm.chat(messages, tools=schemas)
        if not msg.tool_calls:
            print(msg.content)
            return
        messages.append({"role": "assistant", "content": msg.content or None,
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            result = dispatch(registry, tc.function.name, tc.function.arguments)
            print(f"  -> {tc.function.name}({tc.function.arguments})")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=float)})
            if tc.function.name == "submit_answer":
                print(json.dumps(result, indent=1))
                return


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reactor-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="free-form design question")
    a.add_argument("goal")
    a.add_argument("--no-surrogate", action="store_true")

    t = sub.add_parser("task", help="run a golden case by id")
    t.add_argument("task_id")
    t.add_argument("--no-surrogate", action="store_true")

    e = sub.add_parser("evals", help="run the eval suite")
    e.add_argument("--limit", type=int)
    e.add_argument("--only")
    e.add_argument("--no-surrogate", action="store_true")
    e.add_argument("--verbose", action="store_true")

    s = sub.add_parser("surrogate", help="build dataset / train model")
    s.add_argument("what", choices=["build", "train"])
    s.add_argument("--n", type=int, default=4000)
    s.add_argument("--epochs", type=int, default=3000)

    sub.add_parser("cases", help="regenerate golden cases")

    args = p.parse_args(argv)

    if args.cmd == "ask":
        _free_form(args.goal, not args.no_surrogate)
    elif args.cmd == "task":
        from .evals.make_cases import CASES_PATH

        tasks = {c["id"]: Task.model_validate(c) for c in json.loads(CASES_PATH.read_text())}
        if args.task_id not in tasks:
            print("unknown task id; available:", ", ".join(sorted(tasks)), file=sys.stderr)
            return 2
        task = tasks[args.task_id]
        print(task.spec.prompt())
        run = run_task(task, include_surrogate=not args.no_surrogate, verbose=True)
        _print_run(run)
        if task.golden:
            print("\nreference (bisection):", json.dumps(task.golden))
    elif args.cmd == "evals":
        from .evals.run_evals import main as evals_main

        evals_main(args.limit, not args.no_surrogate, args.verbose, args.only)
    elif args.cmd == "surrogate":
        if args.what == "build":
            from .surrogate.dataset import build

            build(args.n)
        else:
            from .surrogate.model import train

            train(epochs=args.epochs)
    elif args.cmd == "cases":
        from .evals.make_cases import CASES_PATH, make

        tasks = make()
        CASES_PATH.write_text(json.dumps([x.model_dump() for x in tasks], indent=1))
        print(f"wrote {len(tasks)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())

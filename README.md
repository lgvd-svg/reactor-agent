# reactor-agent

An LLM agent that designs methane reactors by **driving a real process simulator** (Cantera, GRI-Mech 3.0),
with an **independent verifier** that re-simulates every answer before it is accepted, and a
**JAX neural surrogate** the agent can use to explore quickly before confirming with the exact model.

The point is not that an LLM can talk about reactors. It is that every number the agent reports is
produced by a simulator, checked by code that does not trust the LLM, and measured against a
reference solution.

```
goal ──> LLM ──tool calls──> Cantera (exact)  /  JAX surrogate (fast, approximate)
          │                        │
          └── submit_answer ──> Verifier: re-simulate, check constraints, check minimality ──> verdict
```

## What it does

Given an engineering goal such as

> *Find the smallest residence time for an adiabatic plug-flow reactor at 1420 K, 2 atm, φ = 1.13
> that reaches 80 % CH₄ conversion.*

the agent:

1. checks feasibility with an equilibrium calculation (adiabatic flame temperature is the ceiling),
2. brackets the target with simulator runs (bisection on residence time or temperature),
3. optionally screens many candidates in ~2 ms each with the surrogate,
4. confirms the chosen design with the exact Cantera model,
5. submits a structured answer (`design`, `claimed` outputs, `constraints_met`, `justification`).

Then the **verifier** — plain Python, no LLM — re-runs the exact simulator on the submitted design and
checks three things:

| check | what it catches |
|---|---|
| **claims** | numbers the model invented or mis-copied (tolerance 2 %) |
| **constraints** | conversion target not met, temperature limit exceeded |
| **minimality** | a design that works but is far from the minimum (85 % of τ still passes, or T − 25 K still passes) |

An answer is accepted only if all three pass. The eval suite reports how often the agent *claimed*
success versus how often the verifier *confirmed* it; the difference is what you would have shipped
without a verifier.

## Simulator tools (Cantera)

| tool | model |
|---|---|
| `equilibrium` | HP equilibrium: adiabatic flame temperature and composition |
| `run_batch` | constant-pressure batch reactor, adiabatic or isothermal (pyrolysis with `phi = null`) |
| `run_pfr` | ideal plug-flow reactor in the Lagrangian frame (residence time ≡ reaction time) |
| `run_cstr` | adiabatic CSTR at steady state on the burning branch; short τ blows out |
| `surrogate_predict` | JAX MLP, ~2 ms, flagged *approximate*; never accepted as a final claim |

All inputs are Pydantic models with physical ranges; out-of-range or malformed calls return an
error object the model can read and recover from.

## Surrogate (JAX + Optax)

4-64-64-64-2 MLP trained on 4,000 adiabatic PFR runs sampled over
T₀ ∈ [1000, 1700] K, P ∈ [1, 10] atm, φ ∈ [0.5, 1.5], τ ∈ [3×10⁻⁵, 0.3] s.
Conversion head uses a sigmoid + BCE loss because conversion vs τ is almost a step at the ignition delay.

Held-out test set (800 samples):

| metric | value |
|---|---|
| CH₄ conversion MAE | 0.0075 |
| conversion P90 abs. error | 0.0031 |
| T_out MAE | 34 K |
| ignited / not-ignited classification | 99.5 % |
| training time (CPU) | 8.6 s |

Where it fails, honestly: right at the ignition boundary. At 1200 K, 1 atm, φ = 1 the exact model ignites
at τ ≈ 0.047 s; the surrogate predicts ignition at τ ≈ 0.040 s. That is exactly why the agent is required to
confirm with `run_pfr` and why the verifier never looks at surrogate output.

## Evals

`reactor_agent/evals/cases.json` holds 25 tasks of three kinds with reference answers computed by
bisection on the exact simulator:

- `min_residence_time` (10): smallest τ for a conversion target in an adiabatic PFR
- `min_pyrolysis_temperature` (10): lowest isothermal T for a conversion target in CH₄ pyrolysis
- `adiabatic_flame_temp` (5): report T_ad

Run `reactor-agent evals` to produce `evals/RESULTS.md` with verified success, agent-claimed success,
**false accepts**, tool calls per task and the gap to the bisection reference.

### Harness validation (offline, no LLM)

Two scripted stand-ins drive the exact same loop, tools and verifier, so the pipeline can be proven
without network or quota (`--model oracle` / `--model sloppy`):

| agent | verified | claimed success | **false accepts** | mean tool calls | median gap (τ) |
|---|---|---|---|---|---|
| `oracle` — brackets by bisection, confirms, copies numbers honestly | **25/25** | 25/25 | **0** | 6.8 | 2.1 % |
| `sloppy` — submits the first passing design and claims success | 7/25 | 25/25 | **18** | 3.2 | 43 % |

Both agents *claim* success on every task. The verifier accepts the careful one and rejects 18 of the
sloppy one's answers (all for non-minimality: "85 % of τ still meets the target"). That difference is
the whole point of having a verifier.

Results with a real LLM (`codestral-latest` via llm7.io, or any OpenAI-compatible model):
*pending quota — run `reactor-agent evals` with your key and the table lands in RESULTS.md.*

## Quick start

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env          # defaults to llm7.io free tier; any OpenAI-compatible endpoint works

JAX_PLATFORMS=cpu pytest -q   # offline tests: simulator physics, verifier, agent loop with a fake LLM

reactor-agent task tau-00     # one golden case, verbose, compared to the bisection reference
reactor-agent ask "Lowest isothermal temperature for 30% CH4 pyrolysis conversion in 0.5 s at 1 atm, below 1800 K"
reactor-agent evals --limit 5
reactor-agent surrogate build && reactor-agent surrogate train   # regenerate the surrogate
```

Docker: `docker build -t reactor-agent . && docker run --env-file .env reactor-agent task tau-00`

## Design notes

- **Verification is independent of the model.** The verifier imports the simulator, not the agent. A
  different LLM, or no LLM, gives the same verdict for the same answer.
- **Errors are data.** Simulator exceptions and validation failures are returned to the model as JSON so it
  can adjust instead of crashing the loop.
- **Token economy.** Tool results are rounded to 4 significant figures and composition is opt-in; a full
  task costs roughly 15–25k tokens.
- **Physics shortcuts are stated.** The PFR is the Lagrangian batch analogue (ideal plug flow, no pressure
  drop); the CSTR is initialised on the burning branch. Both are documented in the tool descriptions the
  model reads.

## Layout

```
reactor_agent/
  sim/          Cantera models + Pydantic I/O
  tools.py      tool registry, JSON schemas, validated dispatch
  tasks.py      machine-checkable task specs
  verifier.py   independent re-simulation and checks
  agent/        OpenAI-compatible client, prompts, loop
  surrogate/    dataset sampling, JAX training, inference
  evals/        golden cases (bisection), eval runner, results
tests/          offline: physics sanity, verifier, agent loop with scripted LLM
```

## Author

Luis Guillermo Virues Delgadillo — PhD in Engineering. Six years building process-simulation software,
now building LLM systems that are measured before they ship.

# Eval results — model `oracle`, surrogate on

- Cases: **25**
- Verified success (independent verifier): **25/25**
- Agent claimed success: 25/25
- **False accepts** (agent said OK, verifier said no): **0**
- No answer / errors: 0
- Mean tool calls per task: 6.8
- Total runtime: 14.1 s

| kind | n | verified | claimed | false accepts | median |gap| |
|---|---|---|---|---|---|
| min_residence_time | 10 | 10 | 10 | 0 | 0.0212 |
| min_pyrolysis_temperature | 10 | 10 | 10 | 0 | 5.43 |
| adiabatic_flame_temp | 5 | 5 | 5 | 0 | 0.000127 |

Gap: relative error in residence time, or kelvin for temperature tasks, versus the bisection reference.

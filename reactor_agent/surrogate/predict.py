"""Inference for the trained surrogate; exposed to the agent as `surrogate_predict`."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pydantic import BaseModel, Field

from .dataset import BOUNDS
from .model import PARAMS_PATH


class SurrogateInput(BaseModel):
    T_K: float = Field(..., ge=BOUNDS["T_K"][0], le=BOUNDS["T_K"][1])
    P_atm: float = Field(..., ge=BOUNDS["P_atm"][0], le=BOUNDS["P_atm"][1])
    phi: float = Field(..., ge=BOUNDS["phi"][0], le=BOUNDS["phi"][1])
    residence_times_s: list[float] = Field(
        ..., min_length=1, max_length=25,
        description="One or more residence times to evaluate at once (seconds, 3e-5 to 0.3).",
    )


class SurrogatePrediction(BaseModel):
    residence_time_s: float
    conversion_CH4_approx: float
    T_out_K_approx: float


class SurrogateOutput(BaseModel):
    predictions: list[SurrogatePrediction]
    warning: str = (
        "APPROXIMATE neural surrogate. Confirm any design with run_pfr before submitting."
    )


def is_available() -> bool:
    return PARAMS_PATH.exists()


@lru_cache(maxsize=1)
def _load():
    d = np.load(PARAMS_PATH)
    n = int(d["n_layers"])
    layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
    return d["mean"], d["std"], layers


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def predict(inp: SurrogateInput) -> SurrogateOutput:
    mean, std, layers = _load()
    taus = np.asarray(inp.residence_times_s, dtype=float)
    lo, hi = 10 ** BOUNDS["log10_tau"][0], 10 ** BOUNDS["log10_tau"][1]
    taus = np.clip(taus, lo, hi)
    X = np.column_stack(
        [np.full_like(taus, inp.T_K), np.full_like(taus, inp.P_atm), np.full_like(taus, inp.phi), np.log10(taus)]
    )
    h = (X - mean) / std
    for w, b in layers[:-1]:
        h = _gelu(h @ w + b)
    out = h @ layers[-1][0] + layers[-1][1]
    conv = 1 / (1 + np.exp(-out[:, 0]))
    T_out = out[:, 1] * 1000
    return SurrogateOutput(
        predictions=[
            SurrogatePrediction(residence_time_s=float(t), conversion_CH4_approx=float(c), T_out_K_approx=float(T))
            for t, c, T in zip(taus, conv, T_out, strict=True)
        ]
    )

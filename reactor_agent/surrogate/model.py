"""Small MLP surrogate in pure JAX + Optax.

Inputs are standardised; outputs are (conversion in [0,1] via sigmoid, T_out/1000 K).
Conversion vs residence time is nearly a step at the ignition delay, so a plain MLP
smooths the transition; the README reports where it is and is not trustworthy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .dataset import DATA_PATH

PARAMS_PATH = Path(__file__).with_name("params.npz")
METRICS_PATH = Path(__file__).with_name("metrics.json")
HIDDEN = (64, 64, 64)


def init_params(key, sizes):
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        params.append({"w": w, "b": jnp.zeros(n_out)})
    return params


def forward(params, x):
    h = x
    for layer in params[:-1]:
        h = jax.nn.gelu(h @ layer["w"] + layer["b"])
    out = h @ params[-1]["w"] + params[-1]["b"]
    conv = jax.nn.sigmoid(out[:, 0])
    T_out = out[:, 1]
    return jnp.stack([conv, T_out], axis=1)


def loss_fn(params, x, y):
    pred = forward(params, x)
    # Conversion: BCE handles the near-step behaviour better than MSE. Temperature: MSE on T/1000.
    eps = 1e-6
    bce = -(y[:, 0] * jnp.log(pred[:, 0] + eps) + (1 - y[:, 0]) * jnp.log(1 - pred[:, 0] + eps))
    mse_T = (pred[:, 1] - y[:, 1]) ** 2
    return jnp.mean(bce) + jnp.mean(mse_T)


def standardise(X, mean, std):
    return (X - mean) / std


def train(
    data_path: Path = DATA_PATH,
    epochs: int = 3000,
    lr: float = 2e-3,
    seed: int = 0,
    test_frac: float = 0.2,
) -> dict:
    d = np.load(data_path)
    X, Y = d["X"], d["Y"].copy()
    Y[:, 1] = Y[:, 1] / 1000.0
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr, Xte = standardise(X[tr], mean, std), standardise(X[te], mean, std)
    Xtr, Ytr, Xte, Yte = map(jnp.asarray, (Xtr, Y[tr], Xte, Y[te]))

    params = init_params(jax.random.PRNGKey(seed), (X.shape[1], *HIDDEN, 2))
    sched = optax.cosine_decay_schedule(lr, epochs)
    opt = optax.adamw(sched, weight_decay=1e-5)
    state = opt.init(params)

    @jax.jit
    def step(params, state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    t0 = time.perf_counter()
    for ep in range(epochs):
        params, state, loss = step(params, state, Xtr, Ytr)
        if ep % 500 == 0 or ep == epochs - 1:
            print(f"epoch {ep:5d} loss {float(loss):.4f}")
    train_s = time.perf_counter() - t0

    pred = np.asarray(forward(params, Xte))
    Yte_np = np.asarray(Yte)
    conv_err = np.abs(pred[:, 0] - Yte_np[:, 0])
    T_err_K = np.abs(pred[:, 1] - Yte_np[:, 1]) * 1000
    ignited_true = Yte_np[:, 0] > 0.5
    ignited_pred = pred[:, 0] > 0.5
    metrics = {
        "n_train": len(tr),
        "n_test": len(te),
        "epochs": epochs,
        "train_seconds": round(train_s, 1),
        "conversion_MAE": float(conv_err.mean()),
        "conversion_P90_abs_err": float(np.percentile(conv_err, 90)),
        "T_out_MAE_K": float(T_err_K.mean()),
        "T_out_P90_abs_err_K": float(np.percentile(T_err_K, 90)),
        "ignition_classification_accuracy": float((ignited_true == ignited_pred).mean()),
        "note": "Largest errors sit on the ignition boundary (tau ~ ignition delay), where conversion jumps from ~0 to ~1.",
    }
    flat = {f"w{i}": np.asarray(p["w"]) for i, p in enumerate(params)}
    flat.update({f"b{i}": np.asarray(p["b"]) for i, p in enumerate(params)})
    np.savez(PARAMS_PATH, mean=mean, std=std, n_layers=len(params), **flat)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=2e-3)
    a = p.parse_args()
    train(epochs=a.epochs, lr=a.lr)

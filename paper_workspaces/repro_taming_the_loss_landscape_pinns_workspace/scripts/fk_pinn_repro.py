#!/usr/bin/env python3
# /// script
# dependencies = ["numpy", "scipy", "torch", "plotly"]
# ///
"""Numerical audits for FK-PINN reproduction.

The script is intentionally self-contained so it can run locally and as an
`hf jobs uv run` payload.  It tests reduced one-dimensional PDEs that admit
Feynman-Kac labels; the goal is to audit mechanisms and directional claims,
not to replace the paper's full 2-D benchmark suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import plotly.graph_objects as go
import torch
from scipy import stats


DTYPE = torch.float64


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_np(problem: str, x: np.ndarray) -> np.ndarray:
    if problem == "met":
        return x * (1.0 - x)
    if problem == "committor":
        return x
    if problem == "schrodinger":
        return np.sin(np.pi * x)
    raise ValueError(problem)


def v_np(problem: str, x: np.ndarray) -> np.ndarray:
    if problem == "schrodinger":
        return 5.0 + 4.0 * (x - 0.35) ** 2
    return np.zeros_like(x)


def f_np(problem: str, x: np.ndarray) -> np.ndarray:
    if problem == "met":
        return np.ones_like(x)
    if problem == "committor":
        return np.zeros_like(x)
    if problem == "schrodinger":
        return (0.5 * np.pi**2 + v_np(problem, x)) * np.sin(np.pi * x)
    raise ValueError(problem)


def exact_t(problem: str, x: torch.Tensor) -> torch.Tensor:
    if problem == "met":
        return x * (1.0 - x)
    if problem == "committor":
        return x
    if problem == "schrodinger":
        return torch.sin(math.pi * x)
    raise ValueError(problem)


def v_t(problem: str, x: torch.Tensor) -> torch.Tensor:
    if problem == "schrodinger":
        return 5.0 + 4.0 * (x - 0.35) ** 2
    return torch.zeros_like(x)


def f_t(problem: str, x: torch.Tensor) -> torch.Tensor:
    if problem == "met":
        return torch.ones_like(x)
    if problem == "committor":
        return torch.zeros_like(x)
    if problem == "schrodinger":
        return (0.5 * math.pi**2 + v_t(problem, x)) * torch.sin(math.pi * x)
    raise ValueError(problem)


class TanhNet(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, 1),
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def residual(problem: str, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    u = model(x)
    du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]
    return 0.5 * d2u - v_t(problem, x) * u + f_t(problem, x)


def boundary_loss(problem: str, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    xb = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=device)
    yb = exact_t(problem, xb)
    return torch.mean((model(xb) - yb) ** 2)


def rel_l2_error(problem: str, model: torch.nn.Module, device: torch.device, n: int = 512) -> float:
    x = torch.linspace(0.0, 1.0, n, dtype=DTYPE, device=device).reshape(-1, 1)
    with torch.no_grad():
        pred = model(x)
        y = exact_t(problem, x)
        return float(torch.linalg.norm(pred - y) / torch.linalg.norm(y))


def fk_mc_labels(
    problem: str,
    xs: np.ndarray,
    n_mc: int,
    dt: float,
    tmax: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    out = np.zeros_like(xs)
    max_steps = int(math.ceil(tmax / dt))
    sqrt_dt = math.sqrt(dt)
    for i, x0 in enumerate(xs):
        x = np.full(n_mc, x0, dtype=np.float64)
        alive = np.ones(n_mc, dtype=bool)
        acc = np.zeros(n_mc, dtype=np.float64)
        discount = np.ones(n_mc, dtype=np.float64)
        terminal = np.zeros(n_mc, dtype=np.float64)
        for _ in range(max_steps):
            if not alive.any():
                break
            idx = np.where(alive)[0]
            x_alive = x[idx]
            if problem in ("met", "schrodinger"):
                acc[idx] += discount[idx] * f_np(problem, x_alive) * dt
                discount[idx] *= np.exp(-v_np(problem, x_alive) * dt)
            x[idx] = x_alive + sqrt_dt * rng.standard_normal(size=len(idx))
            exited_left = x[idx] <= 0.0
            exited_right = x[idx] >= 1.0
            exited = exited_left | exited_right
            if problem == "committor" and exited.any():
                terminal[idx[exited]] = exited_right[exited].astype(np.float64)
            alive[idx[exited]] = False
        if problem == "committor":
            # Paths that fail to exit by tmax get the exact harmonic value as a
            # small tail-control proxy; the non-exit fraction is reported below.
            terminal[alive] = x[alive]
            out[i] = terminal.mean()
        else:
            out[i] = acc.mean()
    return out


def fk_mc_samples_single(problem: str, x0: float, n_mc: int, dt: float, tmax: float, seed: int) -> np.ndarray:
    labels = []
    for j in range(n_mc):
        labels.append(fk_mc_labels(problem, np.array([x0]), 1, dt, tmax, seed + j)[0])
    return np.asarray(labels)


@dataclass
class TrainConfig:
    steps: int
    width: int
    n_int: int
    n_fk: int
    n_mc: int
    dt: float
    tmax: float
    lr: float


def train_one(problem: str, use_fk: bool, cfg: TrainConfig, seed: int, device: torch.device) -> dict:
    seed_all(seed)
    model = TanhNet(cfg.width).to(device=device, dtype=DTYPE)
    params = list(model.parameters())
    log_vars = torch.nn.Parameter(torch.zeros(3 if use_fk else 2, dtype=DTYPE, device=device))
    opt = torch.optim.Adam(params + [log_vars], lr=cfg.lr)
    x_int_np = np.linspace(0.0, 1.0, cfg.n_int + 2, dtype=np.float64)[1:-1]
    x_int = torch.tensor(x_int_np.reshape(-1, 1), dtype=DTYPE, device=device)
    x_fk_np = np.linspace(0.0, 1.0, cfg.n_fk + 2, dtype=np.float64)[1:-1]
    if use_fk:
        y_fk_np = fk_mc_labels(problem, x_fk_np, cfg.n_mc, cfg.dt, cfg.tmax, seed + 1000)
        x_fk = torch.tensor(x_fk_np.reshape(-1, 1), dtype=DTYPE, device=device)
        y_fk = torch.tensor(y_fk_np.reshape(-1, 1), dtype=DTYPE, device=device)
    else:
        y_fk_np = None
        x_fk = y_fk = None

    history = []
    start = time.time()
    for step in range(cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        r = residual(problem, model, x_int)
        lpde = torch.mean(r**2)
        lbc = boundary_loss(problem, model, device)
        if use_fk:
            lfk = torch.mean((model(x_fk) - y_fk) ** 2)
            s = torch.clamp(log_vars, -6.0, 6.0)
            loss = torch.exp(-s[0]) * lpde + s[0] + torch.exp(-s[1]) * lbc + s[1] + torch.exp(-s[2]) * lfk + s[2]
        else:
            s = torch.clamp(log_vars, -6.0, 6.0)
            loss = torch.exp(-s[0]) * lpde + s[0] + torch.exp(-s[1]) * lbc + s[1]
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_vars.clamp_(-6.0, 6.0)
        if step % max(1, cfg.steps // 10) == 0 or step == cfg.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "pde_loss": float(lpde.detach().cpu()),
                    "bc_loss": float(lbc.detach().cpu()),
                    "fk_loss": float(lfk.detach().cpu()) if use_fk else None,
                    "rel_l2": rel_l2_error(problem, model, device),
                    "weights": [float(math.exp(-float(z))) for z in log_vars.detach().cpu()],
                }
            )
    return {
        "problem": problem,
        "method": "fk_pinn" if use_fk else "vanilla_pinn",
        "seed": seed,
        "rel_l2": rel_l2_error(problem, model, device),
        "history": history,
        "elapsed_s": time.time() - start,
        "fk_label_rmse": float(np.sqrt(np.mean((y_fk_np - exact_np(problem, x_fk_np)) ** 2))) if use_fk else None,
        "final_weights": history[-1]["weights"],
    }


def basis_condition_audit(n_values: list[int], lam_values: list[float]) -> list[dict]:
    rows = []
    for n in n_values:
        k = max(8, int(round(math.sqrt(n))))
        x = (np.arange(n, dtype=np.float64) + 0.5) / n
        modes = np.arange(1, k + 1, dtype=np.float64)
        phi = np.sqrt(2.0) * np.sin(np.pi * x[:, None] * modes[None, :])
        lphi = (np.pi * modes[None, :]) ** 2 * phi
        gpde = lphi.T @ lphi / n
        gdata = phi.T @ phi / n
        evals = np.linalg.eigvalsh(gpde)
        rows.append(
            {
                "audit": "sine_operator",
                "N": n,
                "K": k,
                "lambda_fk": 0.0,
                "condition": float(evals[-1] / max(evals[0], 1e-300)),
            }
        )
        for lam in lam_values:
            g = gpde + lam * gdata
            e = np.linalg.eigvalsh(g)
            rows.append(
                {
                    "audit": "sine_operator_fk",
                    "N": n,
                    "K": k,
                    "lambda_fk": lam,
                    "condition": float(e[-1] / max(e[0], 1e-300)),
                }
            )
    return rows


def flatten_grads(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> np.ndarray:
    parts = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(g.reshape(-1))
    return torch.cat(parts).detach().cpu().numpy()


def neural_gn_condition(n_values: list[int], seed: int = 0, width: int = 8, n_fk: int = 24) -> list[dict]:
    seed_all(seed)
    device = torch.device("cpu")
    rows = []
    for n in n_values:
        model = TanhNet(width).to(device=device, dtype=DTYPE)
        params = list(model.parameters())
        xvals = np.linspace(0.0, 1.0, n + 2)[1:-1]
        jac_rows = []
        for x0 in xvals:
            x = torch.tensor([[x0]], dtype=DTYPE, device=device, requires_grad=True)
            r = residual("met", model, x).sum()
            jac_rows.append(flatten_grads(torch.autograd.grad(r, params, retain_graph=False, allow_unused=True), params))
        # Boundary rows.
        for x0 in [0.0, 1.0]:
            x = torch.tensor([[x0]], dtype=DTYPE, device=device)
            u = model(x).sum()
            jac_rows.append(flatten_grads(torch.autograd.grad(u, params, retain_graph=False, allow_unused=True), params))
        j_pde = np.vstack(jac_rows)
        gpde = j_pde.T @ j_pde / j_pde.shape[0]

        fk_rows = []
        for x0 in np.linspace(0.0, 1.0, n_fk + 2)[1:-1]:
            x = torch.tensor([[x0]], dtype=DTYPE, device=device)
            u = model(x).sum()
            fk_rows.append(flatten_grads(torch.autograd.grad(u, params, retain_graph=False, allow_unused=True), params))
        j_fk = np.vstack(fk_rows)
        gfk = gpde + (j_fk.T @ j_fk / j_fk.shape[0])
        for name, g in [("neural_gn_pinn", gpde), ("neural_gn_fk", gfk)]:
            e = np.linalg.eigvalsh(g)
            epos = e[e > max(e[-1] * 1e-10, 1e-12)]
            cond = float(e[-1] / epos[0]) if len(epos) else float("inf")
            rows.append({"audit": name, "N": n, "K": width, "lambda_fk": 1.0 if "fk" in name else 0.0, "condition": cond, "rank": int(len(epos))})
    return rows


def mc_bias_noise_audit(mode: str) -> list[dict]:
    xgrid = np.linspace(0.1, 0.9, 17)
    dts = [1e-2, 5e-3, 2.5e-3] if mode == "smoke" else [1e-2, 5e-3, 2.5e-3, 1.25e-3]
    n_mc = 250 if mode == "smoke" else 900
    rows = []
    for problem in ["met", "committor"]:
        for dt in dts:
            y = fk_mc_labels(problem, xgrid, n_mc=n_mc, dt=dt, tmax=2.0, seed=71 + int(1e6 * dt))
            err = y - exact_np(problem, xgrid)
            rows.append(
                {
                    "problem": problem,
                    "dt": dt,
                    "n_mc": n_mc,
                    "mean_abs_bias": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "scaled_by_sqrt_dt": float(np.mean(np.abs(err)) / math.sqrt(dt)),
                }
            )
        samples = fk_mc_samples_single(problem, 0.5, n_mc=700 if mode == "smoke" else 1800, dt=2.5e-3, tmax=2.0, seed=900)
        centered = samples - samples.mean()
        ts = np.quantile(np.abs(centered), [0.6, 0.7, 0.8, 0.9, 0.95])
        probs = np.array([(np.abs(centered) > t).mean() for t in ts])
        mask = probs > 0
        if int(mask.sum()) >= 2 and float(np.max(ts[mask]) - np.min(ts[mask])) > 0:
            slope_t = float(stats.linregress(ts[mask], np.log(probs[mask])).slope)
            slope_t2 = float(stats.linregress(ts[mask] ** 2, np.log(probs[mask])).slope)
        else:
            slope_t = None
            slope_t2 = None
        rows.append(
            {
                "problem": problem,
                "noise_dt": 2.5e-3,
                "sample_n": len(samples),
                "centered_mean": float(centered.mean()),
                "centered_std": float(centered.std()),
                "tail_log_slope_t": slope_t,
                "tail_log_slope_t2": slope_t2,
                "exp_moment_abs_5": float(np.mean(np.exp(5.0 * np.abs(centered)))),
            }
        )
    return rows


def error_bound_scaling(mode: str, device: torch.device) -> list[dict]:
    cfg_base = TrainConfig(
        steps=350 if mode == "smoke" else 900,
        width=18 if mode == "smoke" else 28,
        n_int=160 if mode == "smoke" else 320,
        n_fk=16,
        n_mc=180 if mode == "smoke" else 420,
        dt=3e-3,
        tmax=2.0,
        lr=2e-3,
    )
    nfks = [12, 24] if mode == "smoke" else [16, 32, 64, 128]
    rows = []
    for nfk in nfks:
        cfg = TrainConfig(**{**cfg_base.__dict__, "n_fk": nfk})
        run = train_one("met", True, cfg, seed=123 + nfk, device=device)
        rows.append(
            {
                "N_FK": nfk,
                "width": cfg.width,
                "steps": cfg.steps,
                "rel_l2": run["rel_l2"],
                "fk_label_rmse": run["fk_label_rmse"],
            }
        )
    if len(rows) >= 3:
        xs = np.log([r["N_FK"] for r in rows])
        ys = np.log([r["rel_l2"] for r in rows])
        slope = float(stats.linregress(xs, ys).slope)
        for r in rows:
            r["fit_loglog_slope"] = slope
    return rows


def benchmark_audit(mode: str, device: torch.device) -> list[dict]:
    cfg = TrainConfig(
        steps=400 if mode == "smoke" else 1200,
        width=20 if mode == "smoke" else 36,
        n_int=160 if mode == "smoke" else 384,
        n_fk=36 if mode == "smoke" else 72,
        n_mc=180 if mode == "smoke" else 360,
        dt=3e-3 if mode == "smoke" else 2e-3,
        tmax=2.0,
        lr=2e-3,
    )
    seeds = [0] if mode == "smoke" else [0, 1]
    rows = []
    histories = []
    for problem in ["schrodinger", "met", "committor"]:
        for seed in seeds:
            for use_fk in [False, True]:
                run = train_one(problem, use_fk, cfg, seed=seed, device=device)
                rows.append({k: run[k] for k in ["problem", "method", "seed", "rel_l2", "elapsed_s", "fk_label_rmse", "final_weights"]})
                for h in run["history"]:
                    histories.append({"problem": problem, "method": run["method"], "seed": seed, **h})
    return rows, histories


def iteration_proxy(condition_rows: list[dict], eps: float = 1e-3) -> list[dict]:
    rows = []
    for r in condition_rows:
        kappa = r["condition"]
        if not np.isfinite(kappa) or kappa <= 1:
            iters = None
        else:
            iters = int(math.ceil(math.log(eps) / math.log(1.0 - 1.0 / kappa)))
        rows.append({**r, "epsilon": eps, "predicted_gd_iters": iters})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_figures(out: Path, condition_rows: list[dict], bias_rows: list[dict], bench_rows: list[dict], scaling_rows: list[dict]) -> None:
    fig = go.Figure()
    for audit in sorted({r["audit"] for r in condition_rows}):
        rs = [r for r in condition_rows if r["audit"] == audit]
        fig.add_trace(go.Scatter(x=[r["N"] for r in rs], y=[r["condition"] for r in rs], mode="lines+markers", name=audit))
    fig.update_layout(title="Condition number audits", xaxis_title="N", yaxis_title="condition", xaxis_type="log", yaxis_type="log")
    fig.write_html(out / "condition_audit.html", include_plotlyjs="cdn")

    fig = go.Figure()
    for problem in sorted({r["problem"] for r in bias_rows if "dt" in r}):
        rs = [r for r in bias_rows if r.get("problem") == problem and "dt" in r]
        fig.add_trace(go.Scatter(x=[r["dt"] for r in rs], y=[r["mean_abs_bias"] for r in rs], mode="lines+markers", name=problem))
    fig.update_layout(title="FK Monte Carlo bias vs Euler dt", xaxis_title="dt", yaxis_title="mean |bias|", xaxis_type="log", yaxis_type="log")
    fig.write_html(out / "mc_bias.html", include_plotlyjs="cdn")

    fig = go.Figure()
    problems = ["schrodinger", "met", "committor"]
    for method in ["vanilla_pinn", "fk_pinn"]:
        vals = []
        err = []
        for p in problems:
            xs = [r["rel_l2"] for r in bench_rows if r["problem"] == p and r["method"] == method]
            vals.append(float(np.mean(xs)))
            err.append(float(np.std(xs)) if len(xs) > 1 else 0.0)
        fig.add_trace(go.Bar(x=problems, y=vals, error_y={"type": "data", "array": err}, name=method))
    fig.update_layout(title="Scaled 1D FK-PINN benchmarks", yaxis_title="relative L2 error", barmode="group")
    fig.write_html(out / "benchmark_errors.html", include_plotlyjs="cdn")

    if scaling_rows:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r["N_FK"] for r in scaling_rows], y=[r["rel_l2"] for r in scaling_rows], mode="lines+markers", name="MET FK-PINN"))
        fig.update_layout(title="Error-bound scaling audit", xaxis_title="N_FK", yaxis_title="relative L2 error", xaxis_type="log", yaxis_type="log")
        fig.write_html(out / "error_scaling.html", include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.set_default_dtype(DTYPE)
    print(f"device={device} torch={torch.__version__} mode={args.mode}")

    start = time.time()
    n_values = [64, 128, 256] if args.mode == "smoke" else [64, 128, 256, 512, 1024]
    condition_rows = basis_condition_audit(n_values, lam_values=[1.0, 100.0])
    condition_rows += neural_gn_condition([64, 128] if args.mode == "smoke" else [64, 128, 256], seed=0, width=8 if args.mode == "smoke" else 10)
    bias_rows = mc_bias_noise_audit(args.mode)
    scaling_rows = error_bound_scaling(args.mode, device)
    bench_rows, history_rows = benchmark_audit(args.mode, device)
    iter_rows = iteration_proxy(condition_rows)

    write_csv(out / "condition_audit.csv", condition_rows)
    write_csv(out / "mc_bias_noise.csv", bias_rows)
    write_csv(out / "error_scaling.csv", scaling_rows)
    write_csv(out / "benchmark_errors.csv", bench_rows)
    write_csv(out / "training_history.csv", history_rows)
    write_csv(out / "iteration_proxy.csv", iter_rows)
    make_figures(out, condition_rows, bias_rows, bench_rows, scaling_rows)

    table_claim = {
        "paper_table_1_relative_l2": {
            "schrodinger": {"vanilla": 0.624, "fk_pinn": 0.096, "improvement": 0.624 / 0.096},
            "met": {"vanilla": 1.007, "fk_pinn": 0.107, "improvement": 1.007 / 0.107},
            "committor": {"vanilla": 0.839, "fk_pinn": 0.030, "improvement": 0.839 / 0.030},
        }
    }
    summary = {
        "mode": args.mode,
        "device": str(device),
        "elapsed_s": time.time() - start,
        "condition_slope_loglog": {},
        "mc_bias_slope_loglog": {},
        "benchmark_means": {},
        "error_scaling_slope": None,
        "paper_claim_numbers": table_claim,
    }
    for audit in sorted({r["audit"] for r in condition_rows}):
        rs = [r for r in condition_rows if r["audit"] == audit and np.isfinite(r["condition"])]
        if len(rs) >= 3:
            summary["condition_slope_loglog"][audit] = float(stats.linregress(np.log([r["N"] for r in rs]), np.log([r["condition"] for r in rs])).slope)
    for problem in sorted({r["problem"] for r in bias_rows if "dt" in r}):
        rs = [r for r in bias_rows if r.get("problem") == problem and "dt" in r]
        if len(rs) >= 3:
            summary["mc_bias_slope_loglog"][problem] = float(stats.linregress(np.log([r["dt"] for r in rs]), np.log([r["mean_abs_bias"] for r in rs])).slope)
    for p in ["schrodinger", "met", "committor"]:
        summary["benchmark_means"][p] = {}
        for method in ["vanilla_pinn", "fk_pinn"]:
            xs = [r["rel_l2"] for r in bench_rows if r["problem"] == p and r["method"] == method]
            summary["benchmark_means"][p][method] = float(np.mean(xs))
        v = summary["benchmark_means"][p]["vanilla_pinn"]
        f = summary["benchmark_means"][p]["fk_pinn"]
        summary["benchmark_means"][p]["improvement"] = float(v / f) if f > 0 else None
    if len(scaling_rows) >= 3:
        summary["error_scaling_slope"] = float(stats.linregress(np.log([r["N_FK"] for r in scaling_rows]), np.log([r["rel_l2"] for r in scaling_rows])).slope)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY_JSON_START")
    print(json.dumps(summary, indent=2))
    print("SUMMARY_JSON_END")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()

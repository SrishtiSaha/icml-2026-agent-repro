#!/usr/bin/env python3
# /// script
# dependencies = ["numpy", "scipy", "torch", "plotly"]
# ///
"""Stronger FK-PINN audits for the weak reproduction claims.

This is intentionally heavier than the first-pass 1D script.  It uses 2D PDEs,
the paper-style tanh network family, fixed-weight standard PINNs, and
uncertainty-weighted FK-PINNs.  The goal is a faithful numerical audit of the
claim logic, while keeping the wall time small enough for a challenge run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import plotly.graph_objects as go
import torch
from scipy import stats


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(torch.nn.Module):
    def __init__(self, width: int, hidden: int = 4):
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(2, width), torch.nn.Tanh()]
        for _ in range(hidden - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers.append(torch.nn.Linear(width, 1))
        self.net = torch.nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def exact(problem: str, x: torch.Tensor) -> torch.Tensor:
    xx, yy = x[:, :1], x[:, 1:2]
    if problem == "poisson":
        return torch.sin(math.pi * xx) * torch.sin(math.pi * yy) + 0.15 * torch.sin(5 * math.pi * xx) * torch.sin(6 * math.pi * yy)
    if problem == "met":
        r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
        return torch.clamp(0.25 - r2, min=0.0)
    if problem == "schrodinger":
        return torch.sin(5 * math.pi * xx) * torch.sin(5 * math.pi * yy)
    if problem == "committor":
        return xx
    raise ValueError(problem)


def potential(problem: str, x: torch.Tensor) -> torch.Tensor:
    xx, yy = x[:, :1], x[:, 1:2]
    if problem == "schrodinger":
        return -5.0 * (torch.cos(4 * math.pi * xx) + torch.cos(4 * math.pi * yy))
    return torch.zeros_like(xx)


def source(problem: str, x: torch.Tensor) -> torch.Tensor:
    xx, yy = x[:, :1], x[:, 1:2]
    if problem == "poisson":
        return (
            (math.pi**2) * torch.sin(math.pi * xx) * torch.sin(math.pi * yy)
            + 0.15 * 0.5 * ((5 * math.pi) ** 2 + (6 * math.pi) ** 2) * torch.sin(5 * math.pi * xx) * torch.sin(6 * math.pi * yy)
        )
    if problem == "met":
        return torch.ones_like(xx)
    if problem == "schrodinger":
        psi = exact(problem, x)
        return 0.5 * 50 * math.pi**2 * psi + potential(problem, x) * psi
    if problem == "committor":
        return torch.zeros_like(xx)
    raise ValueError(problem)


def laplacian(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    u = model(x)
    grad = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    parts = []
    for j in range(2):
        g = grad[:, j : j + 1]
        h = torch.autograd.grad(g, x, torch.ones_like(g), create_graph=True)[0][:, j : j + 1]
        parts.append(h)
    return parts[0] + parts[1]


def residual(problem: str, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if problem in ("poisson", "met"):
        return -0.5 * laplacian(model, x) - source(problem, x)
    if problem == "schrodinger":
        return -0.5 * laplacian(model, x) + potential(problem, x) * model(x) - source(problem, x)
    if problem == "committor":
        return laplacian(model, x)
    raise ValueError(problem)


def sample_interior(problem: str, n: int, device: torch.device) -> torch.Tensor:
    if problem == "met":
        # Disk centered at (0.5, 0.5) radius 0.5.
        theta = torch.rand(n, 1, device=device) * 2 * math.pi
        r = torch.sqrt(torch.rand(n, 1, device=device)) * 0.5
        return torch.cat([0.5 + r * torch.cos(theta), 0.5 + r * torch.sin(theta)], dim=1)
    return torch.rand(n, 2, device=device)


def sample_boundary(problem: str, n: int, device: torch.device) -> torch.Tensor:
    if problem == "met":
        theta = torch.rand(n, 1, device=device) * 2 * math.pi
        return torch.cat([0.5 + 0.5 * torch.cos(theta), 0.5 + 0.5 * torch.sin(theta)], dim=1)
    m = n // 4
    t = torch.rand(m, 1, device=device)
    pts = [
        torch.cat([torch.zeros_like(t), t], 1),
        torch.cat([torch.ones_like(t), t], 1),
        torch.cat([t, torch.zeros_like(t)], 1),
        torch.cat([t, torch.ones_like(t)], 1),
    ]
    return torch.cat(pts, 0)


@dataclass
class TrainCfg:
    steps: int
    width: int
    hidden: int
    n_int: int
    n_bc: int
    n_fk: int
    batch: int
    lr: float
    fk_noise: float


def loss_terms(problem: str, model: torch.nn.Module, xi: torch.Tensor, xb: torch.Tensor, xfk: torch.Tensor | None, yfk: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lpde = torch.mean(residual(problem, model, xi) ** 2)
    lbc = torch.mean((model(xb) - exact(problem, xb)) ** 2)
    if xfk is None or yfk is None:
        lfk = torch.zeros((), device=xi.device)
    else:
        lfk = torch.mean((model(xfk) - yfk) ** 2)
    return lpde, lbc, lfk


def train_model(problem: str, method: str, cfg: TrainCfg, seed: int, device: torch.device) -> dict:
    seed_all(seed)
    model = MLP(cfg.width, cfg.hidden).to(device)
    xi_all = sample_interior(problem, cfg.n_int, device)
    xb_all = sample_boundary(problem, cfg.n_bc, device)
    if method == "fk":
        xfk = sample_interior(problem, cfg.n_fk, device)
        yfk = exact(problem, xfk) + cfg.fk_noise * torch.randn(cfg.n_fk, 1, device=device)
        logvars = torch.nn.Parameter(torch.zeros(3, device=device))
        opt = torch.optim.Adam(list(model.parameters()) + [logvars], lr=cfg.lr)
    else:
        xfk = yfk = None
        logvars = None
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    hist = []
    for step in range(cfg.steps):
        idx = torch.randint(0, cfg.n_int, (min(cfg.batch, cfg.n_int),), device=device)
        bidx = torch.randint(0, xb_all.shape[0], (min(max(32, cfg.batch // 4), xb_all.shape[0]),), device=device)
        if xfk is not None:
            fidx = torch.randint(0, cfg.n_fk, (min(max(32, cfg.batch // 4), cfg.n_fk),), device=device)
            xfb, yfb = xfk[fidx], yfk[fidx]
        else:
            xfb = yfb = None
        opt.zero_grad(set_to_none=True)
        lpde, lbc, lfk = loss_terms(problem, model, xi_all[idx], xb_all[bidx], xfb, yfb)
        if method == "fk":
            s = torch.clamp(logvars, -8.0, 8.0)
            loss = torch.exp(-s[0]) * lpde + s[0] + torch.exp(-s[1]) * lbc + s[1] + torch.exp(-s[2]) * lfk + s[2]
        else:
            loss = lpde + lbc
        loss.backward()
        opt.step()
        if logvars is not None:
            with torch.no_grad():
                logvars.clamp_(-8.0, 8.0)
        if step in {0, cfg.steps // 4, cfg.steps // 2, cfg.steps - 1}:
            hist.append({"step": step, "loss": float(loss.detach().cpu()), "lpde": float(lpde.detach().cpu()), "lbc": float(lbc.detach().cpu()), "lfk": float(lfk.detach().cpu())})
    return {
        "model": model,
        "xi": xi_all,
        "xb": xb_all,
        "xfk": xfk,
        "yfk": yfk,
        "history": hist,
        "weights": [float(math.exp(-v)) for v in logvars.detach().cpu()] if logvars is not None else [1.0, 1.0],
    }


def rel_l2(problem: str, model: torch.nn.Module, device: torch.device, n: int = 12000) -> float:
    x = sample_interior(problem, n, device)
    with torch.no_grad():
        y = exact(problem, x)
        p = model(x)
        return float((torch.linalg.norm(p - y) / torch.linalg.norm(y)).detach().cpu())


def gd_decay_probe(problem: str, trained: dict, method: str, steps: int, device: torch.device) -> dict:
    # Probe the local loss decay under fixed-step full-batch GD on a copied model.
    src = trained["model"]
    model = MLP(src.net[0].out_features, sum(1 for m in src.net if isinstance(m, torch.nn.Tanh))).to(device)
    model.load_state_dict(src.state_dict())
    xi = trained["xi"][: min(1200, trained["xi"].shape[0])]
    xb = trained["xb"][: min(300, trained["xb"].shape[0])]
    xfk = trained["xfk"][: min(300, trained["xfk"].shape[0])] if trained["xfk"] is not None else None
    yfk = trained["yfk"][: min(300, trained["yfk"].shape[0])] if trained["yfk"] is not None else None
    eta = 2e-5 if problem in ("poisson", "schrodinger") else 1e-4
    vals = []
    for t in range(steps + 1):
        for p in model.parameters():
            p.grad = None
        lpde, lbc, lfk = loss_terms(problem, model, xi, xb, xfk, yfk)
        loss = lpde + lbc + (lfk if method == "fk" else 0.0)
        if t % max(1, steps // 20) == 0:
            vals.append((t, float(loss.detach().cpu())))
        if t == steps:
            break
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= eta * p.grad
    arr = np.array(vals, dtype=float)
    tail = arr[len(arr) // 2 :]
    slope = stats.linregress(tail[:, 0], np.log(np.maximum(tail[:, 1], 1e-30))).slope
    rate = max(1e-12, -float(slope))
    corr = float(abs(np.corrcoef(tail[:, 0], np.log(np.maximum(tail[:, 1], 1e-30)))[0, 1]))
    return {"rate": rate, "corr": corr, "t_eps_0.1": float(math.log(10) / rate), "loss0": vals[0][1], "lossT": vals[-1][1]}


def condition_and_speed(mode: str, out: Path, device: torch.device) -> tuple[list[dict], list[dict]]:
    n_values = [500, 1000, 2000] if mode == "smoke" else [500, 1000, 2000, 4000, 8000]
    steps = 260 if mode == "smoke" else 650
    width = 48 if mode == "smoke" else 96
    seeds = [0] if mode == "smoke" else [0, 1]
    cond_rows, speed_rows = [], []
    for problem in ["poisson", "met"]:
        for n in n_values:
            for seed in seeds:
                for method in ["pinn", "fk"]:
                    cfg = TrainCfg(steps=steps, width=width, hidden=4, n_int=n, n_bc=400, n_fk=max(20, n // 50), batch=384, lr=2e-3, fk_noise=0.02)
                    tr = train_model(problem, method, cfg, 1000 + seed + n, device)
                    probe = gd_decay_probe(problem, tr, method, 80 if mode == "smoke" else 160, device)
                    kappa_proxy = 1.0 / probe["rate"]
                    cond_rows.append({"problem": problem, "N": n, "seed": seed, "method": method, "kappa_pl_proxy": kappa_proxy, "rate": probe["rate"], "corr_log_loss": probe["corr"]})
                    speed_rows.append({"problem": problem, "N": n, "seed": seed, "method": method, **probe})
                    print("COND", problem, n, seed, method, kappa_proxy, flush=True)
    write_csv(out / "paper_scale_condition.csv", cond_rows)
    write_csv(out / "paper_scale_speed.csv", speed_rows)
    return cond_rows, speed_rows


def claim2_bound(mode: str, out: Path, device: torch.device) -> list[dict]:
    nfks = [32, 64, 128] if mode == "smoke" else [32, 64, 128, 256, 512]
    rows = []
    beta = 0.125
    c_by_problem: dict[str, float] = {}
    for problem in ["poisson", "met"]:
        for nfk in nfks:
            width = int(round(42 * (nfk / 32) ** 0.125)) if mode == "smoke" else int(round(64 * (nfk / 32) ** 0.125))
            cfg = TrainCfg(
                steps=450 if mode == "smoke" else 1100,
                width=width,
                hidden=2,
                n_int=10 * nfk,
                n_bc=max(200, nfk),
                n_fk=nfk,
                batch=min(512, 10 * nfk),
                lr=2e-3,
                fk_noise=0.03,
            )
            tr = train_model(problem, "fk", cfg, 3000 + nfk, device)
            err = rel_l2(problem, tr["model"], device, n=6000 if mode == "smoke" else 14000)
            bias_term = cfg.fk_noise + math.sqrt(cfg.fk_noise)
            opt_term = math.exp(-2.5e-4 * cfg.steps)
            raw_bound = nfk ** (-beta) + opt_term + bias_term
            if nfk == nfks[0]:
                c_by_problem[problem] = err / raw_bound
            bound = c_by_problem[problem] * raw_bound
            rows.append({"problem": problem, "N_FK": nfk, "width": width, "rel_l2": err, "raw_bound_terms": raw_bound, "calibrated_bound": bound, "measured_over_bound": err / bound, "beta": beta})
            print("C2", problem, nfk, width, err, bound, flush=True)
    write_csv(out / "paper_scale_claim2.csv", rows)
    return rows


def claim4_table(mode: str, out: Path, device: torch.device) -> list[dict]:
    problems = ["schrodinger", "met", "committor"]
    seeds = [0] if mode == "smoke" else [0, 1, 2]
    rows = []
    for problem in problems:
        for seed in seeds:
            for method in ["pinn", "fk"]:
                cfg = TrainCfg(
                    steps=500 if mode == "smoke" else 1600,
                    width=48 if mode == "smoke" else 96,
                    hidden=4,
                    n_int=2500 if mode == "smoke" else 9000,
                    n_bc=400,
                    n_fk=50 if mode == "smoke" else 180,
                    batch=512,
                    lr=1.5e-3,
                    fk_noise=0.02 if problem != "met" else 0.10,
                )
                tr = train_model(problem, method, cfg, 5000 + seed, device)
                err = rel_l2(problem, tr["model"], device, n=8000 if mode == "smoke" else 16000)
                rows.append({"problem": problem, "method": method, "seed": seed, "rel_l2": err, "weights": json.dumps(tr["weights"])})
                print("C4", problem, method, seed, err, flush=True)
    write_csv(out / "paper_scale_claim4.csv", rows)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def grouped_mean(rows: list[dict], key_fields: list[str], value: str) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        k = tuple(r[f] for f in key_fields)
        groups.setdefault(k, []).append(float(r[value]))
    out = []
    for k, vals in sorted(groups.items()):
        rec = {f: v for f, v in zip(key_fields, k)}
        rec[value + "_mean"] = float(np.mean(vals))
        rec[value + "_std"] = float(np.std(vals))
        out.append(rec)
    return out


def make_figures(out: Path, cond: list[dict], c2: list[dict], c4: list[dict], speed: list[dict]) -> None:
    fig = go.Figure()
    for problem in sorted({r["problem"] for r in cond}):
        for method in ["pinn", "fk"]:
            rs = [r for r in grouped_mean([x for x in cond if x["problem"] == problem and x["method"] == method], ["N"], "kappa_pl_proxy")]
            fig.add_trace(go.Scatter(x=[r["N"] for r in rs], y=[r["kappa_pl_proxy_mean"] for r in rs], mode="lines+markers", name=f"{problem}-{method}"))
    fig.update_layout(title="Paper-scale PL proxy conditioning", xaxis_type="log", yaxis_type="log", xaxis_title="collocation N", yaxis_title="1 / fitted GD decay rate")
    fig.write_html(out / "paper_scale_condition.html", include_plotlyjs="cdn")

    fig = go.Figure()
    for problem in sorted({r["problem"] for r in c2}):
        rs = [r for r in c2 if r["problem"] == problem]
        fig.add_trace(go.Scatter(x=[r["N_FK"] for r in rs], y=[r["rel_l2"] for r in rs], mode="lines+markers", name=f"{problem} measured"))
        fig.add_trace(go.Scatter(x=[r["N_FK"] for r in rs], y=[r["calibrated_bound"] for r in rs], mode="lines+markers", name=f"{problem} bound"))
    fig.update_layout(title="Claim 2 out-of-sample bound", xaxis_type="log", yaxis_type="log", xaxis_title="N_FK")
    fig.write_html(out / "paper_scale_claim2.html", include_plotlyjs="cdn")

    fig = go.Figure()
    means = grouped_mean(c4, ["problem", "method"], "rel_l2")
    for method in ["pinn", "fk"]:
        rs = [r for r in means if r["method"] == method]
        fig.add_trace(go.Bar(x=[r["problem"] for r in rs], y=[r["rel_l2_mean"] for r in rs], error_y={"type": "data", "array": [r["rel_l2_std"] for r in rs]}, name=method))
    fig.update_layout(title="Paper-protocol benchmark errors", yaxis_title="relative L2", barmode="group")
    fig.write_html(out / "paper_scale_claim4.html", include_plotlyjs="cdn")

    # Speed-up from actual rates.
    speed_means = grouped_mean(speed, ["problem", "N", "method"], "rate")
    fig = go.Figure()
    for problem in sorted({r["problem"] for r in speed_means}):
        ns = sorted({r["N"] for r in speed_means if r["problem"] == problem})
        vals = []
        for n in ns:
            rp = next(r["rate_mean"] for r in speed_means if r["problem"] == problem and r["N"] == n and r["method"] == "pinn")
            rf = next(r["rate_mean"] for r in speed_means if r["problem"] == problem and r["N"] == n and r["method"] == "fk")
            vals.append(rf / max(rp, 1e-12))
        fig.add_trace(go.Scatter(x=ns, y=vals, mode="lines+markers", name=problem))
    fig.update_layout(title="Measured GD speed-up T_PINN/T_FK", xaxis_type="log", yaxis_type="log", xaxis_title="collocation N")
    fig.write_html(out / "paper_scale_speed.html", include_plotlyjs="cdn")


def summarize(out: Path, cond: list[dict], c2: list[dict], c4: list[dict], speed: list[dict], elapsed: float) -> dict:
    summary: dict = {"elapsed_s": elapsed, "condition_slopes": {}, "claim2": {}, "claim4_means": {}, "speed_slopes": {}}
    for problem in sorted({r["problem"] for r in cond}):
        for method in ["pinn", "fk"]:
            means = grouped_mean([r for r in cond if r["problem"] == problem and r["method"] == method], ["N"], "kappa_pl_proxy")
            if len(means) >= 3:
                summary["condition_slopes"][f"{problem}_{method}"] = float(stats.linregress(np.log([r["N"] for r in means]), np.log([r["kappa_pl_proxy_mean"] for r in means])).slope)
    for problem in sorted({r["problem"] for r in c2}):
        rs = [r for r in c2 if r["problem"] == problem]
        ratios = [r["measured_over_bound"] for r in rs[1:]]
        summary["claim2"][problem] = {"max_unseen_ratio": float(max(ratios)) if ratios else None, "last_ratio": float(ratios[-1]) if ratios else None}
    for r in grouped_mean(c4, ["problem", "method"], "rel_l2"):
        summary["claim4_means"].setdefault(r["problem"], {})[r["method"]] = r["rel_l2_mean"]
    for p, d in summary["claim4_means"].items():
        if "pinn" in d and "fk" in d:
            d["improvement"] = d["pinn"] / max(d["fk"], 1e-12)
    speed_means = grouped_mean(speed, ["problem", "N", "method"], "rate")
    for problem in sorted({r["problem"] for r in speed_means}):
        ns = sorted({r["N"] for r in speed_means if r["problem"] == problem})
        vals = []
        for n in ns:
            rp = next(r["rate_mean"] for r in speed_means if r["problem"] == problem and r["N"] == n and r["method"] == "pinn")
            rf = next(r["rate_mean"] for r in speed_means if r["problem"] == problem and r["N"] == n and r["method"] == "fk")
            vals.append(rf / max(rp, 1e-12))
        if len(vals) >= 3:
            summary["speed_slopes"][problem] = float(stats.linregress(np.log(ns), np.log(vals)).slope)
    (out / "paper_scale_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--outdir", default="results/paper_scale")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sections", default="all", help="comma list: c1,c2,c4,c5 or all")
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = device_auto() if args.device == "auto" else torch.device(args.device)
    print(f"device={device} torch={torch.__version__} mode={args.mode}", flush=True)
    start = time.time()
    sections = {"c1", "c2", "c4", "c5"} if args.sections == "all" else set(args.sections.split(","))
    cond: list[dict] = []
    speed: list[dict] = []
    c2: list[dict] = []
    c4: list[dict] = []
    if "c1" in sections or "c5" in sections:
        cond, speed = condition_and_speed(args.mode, out, device)
    if "c2" in sections:
        c2 = claim2_bound(args.mode, out, device)
    if "c4" in sections:
        c4 = claim4_table(args.mode, out, device)
    make_figures(out, cond, c2, c4, speed)
    summary = summarize(out, cond, c2, c4, speed, time.time() - start)
    print("SUMMARY_JSON_START")
    print(json.dumps(summary, indent=2))
    print("SUMMARY_JSON_END")


if __name__ == "__main__":
    main()

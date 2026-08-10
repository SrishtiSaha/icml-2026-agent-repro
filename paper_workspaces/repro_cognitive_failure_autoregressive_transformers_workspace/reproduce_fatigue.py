#!/usr/bin/env python3
# /// script
# dependencies = [
#   "accelerate>=0.33.0",
#   "datasets>=2.20.0",
#   "huggingface-hub>=0.24.0",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "plotly>=5.22.0",
#   "scikit-learn>=1.5.0",
#   "scipy>=1.13.0",
#   "sentencepiece>=0.2.0",
#   "torch>=2.3.0",
#   "transformers>=4.44.0,<4.47.0",
# ]
# ///
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


K_PROMPT = 64
H_LOW = 3.8
H_HIGH = 5.0
BETA = 5.0
KAPPA = 50.0
SMOOTH_WINDOW = 5
THETA = 0.50
THETA_LOW = 0.40
WEIGHTS = {"attention": 0.40, "entropy": 0.35, "drift": 0.25}


@dataclass
class SequenceResult:
    dataset: str
    sample_id: int
    prompt_len: int
    generated_len: int
    text: str
    rep_ratio: float
    mean_fi: float
    first20_fi: float
    mean_phi_a: float
    mean_phi_e: float
    mean_phi_d: float
    mean_attention: float
    mean_entropy: float
    mean_drift: float
    static_flips: int
    hysteresis_flips: int


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def phi_a(attention_mass: float) -> float:
    return 1.0 - clip01(attention_mass)


def phi_e(entropy: float) -> float:
    if entropy < H_LOW:
        penalty = (H_LOW - entropy) / H_LOW
    elif entropy <= H_HIGH:
        penalty = 0.0
    else:
        penalty = (entropy - H_HIGH) / BETA
    return clip01(penalty)


def phi_d(drift: float) -> float:
    return clip01(drift / KAPPA)


def fatigue_index(attention_mass: float, entropy: float, drift: float) -> float:
    return (
        WEIGHTS["attention"] * phi_a(attention_mass)
        + WEIGHTS["entropy"] * phi_e(entropy)
        + WEIGHTS["drift"] * phi_d(drift)
    )


def moving_average(xs: list[float], window: int = SMOOTH_WINDOW) -> list[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(xs[lo : i + 1])))
    return out


def count_static_flips(xs: list[float], theta: float = THETA) -> int:
    if not xs:
        return 0
    states = [x >= theta for x in xs]
    return sum(int(a != b) for a, b in zip(states, states[1:]))


def count_hysteresis_flips(xs: list[float], theta: float = THETA, low: float = THETA_LOW) -> int:
    active = False
    flips = 0
    for x in xs:
        if not active and x >= theta:
            active = True
            flips += 1
        elif active and x <= low:
            active = False
            flips += 1
    return flips


def repetition_ratio(token_ids: list[int], n: int = 4) -> float:
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    for gram in grams:
        if gram in seen:
            repeated += 1
        seen.add(gram)
    return repeated / max(1, len(grams))


def entropy_from_logits(logits: torch.Tensor) -> float:
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(-(probs * log_probs).sum(dim=-1).item())


def top_p_sample(logits: torch.Tensor, top_p: float = 0.95, temperature: float = 1.0) -> torch.Tensor:
    logits = logits.float() / max(temperature, 1e-6)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    next_sorted = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, next_sorted)


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_lm(model_id: str, need_attention: bool = False, quant4: bool = False):
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None
    if need_attention:
        kwargs["attn_implementation"] = "eager"
    if quant4:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs.pop("torch_dtype", None)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not torch.cuda.is_available():
        model.to(device_name())
    model.eval()
    return tok, model


def cleanup_model(tok=None, model=None) -> None:
    del tok
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def manual_probe_generate(
    model,
    tok,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    need_attention: bool = True,
    need_hidden: bool = True,
    attention_span: tuple[int, int] | None = None,
    top_p: float = 0.95,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])
    past = None
    cur = input_ids
    h0 = None
    generated: list[int] = []
    rows: list[dict[str, float]] = []
    use_cache = True
    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(
                input_ids=cur,
                past_key_values=past,
                use_cache=use_cache,
                output_attentions=need_attention,
                output_hidden_states=need_hidden,
            )
            logits = out.logits[:, -1, :]
            ent = entropy_from_logits(logits)
            att_mass = 0.0
            if need_attention and out.attentions is not None:
                att = out.attentions[-1][0, :, -1, :]
                if attention_span is None:
                    stop = min(K_PROMPT, prompt_len, att.shape[-1])
                    att_mass = float(att[:, :stop].sum(dim=-1).mean().item())
                else:
                    lo, hi = attention_span
                    lo = max(0, min(lo, att.shape[-1]))
                    hi = max(lo, min(hi, att.shape[-1]))
                    att_mass = float(att[:, lo:hi].sum(dim=-1).mean().item()) if hi > lo else 0.0
            drift = 0.0
            if need_hidden and out.hidden_states is not None:
                hidden = out.hidden_states[-1][0]
                if h0 is None:
                    h0 = hidden[min(prompt_len - 1, hidden.shape[0] - 1)].detach().float()
                ht = hidden[-1].detach().float()
                drift = float(torch.linalg.vector_norm(ht - h0).item())
            fi = fatigue_index(att_mass, ent, drift)
            rows.append(
                {
                    "step": step,
                    "attention": att_mass,
                    "entropy": ent,
                    "drift": drift,
                    "phi_a": phi_a(att_mass),
                    "phi_e": phi_e(ent),
                    "phi_d": phi_d(drift),
                    "fi": fi,
                }
            )
            next_token = top_p_sample(logits, top_p=top_p, temperature=1.0)
            token = int(next_token.item())
            generated.append(token)
            cur = next_token.to(device)
            past = out.past_key_values
            if tok.eos_token_id is not None and token == tok.eos_token_id:
                break
    text = tok.decode(generated, skip_special_tokens=True)
    return {
        "prompt_len": prompt_len,
        "generated_ids": generated,
        "text": text,
        "trajectory": rows,
    }


def formula_audit(outdir: Path) -> dict[str, Any]:
    rows = []
    for a in np.linspace(0, 1, 5):
        for e in [0.0, 2.0, 3.8, 4.4, 5.0, 7.5]:
            for d in [0.0, 10.0, 25.0, 50.0, 75.0]:
                rows.append({"A": a, "E": e, "D": d, "FI": fatigue_index(a, e, d)})
    df = pd.DataFrame(rows)
    path = outdir / "claim1_formula_grid.csv"
    df.to_csv(path, index=False)
    checks = {
        "n_points": int(len(df)),
        "bounded": bool(((df["FI"] >= -1e-12) & (df["FI"] <= 1 + 1e-12)).all()),
        "weights_sum": float(sum(WEIGHTS.values())),
        "weights": WEIGHTS,
        "constants": {
            "K": K_PROMPT,
            "H_low": H_LOW,
            "H_high": H_HIGH,
            "beta": BETA,
            "kappa": KAPPA,
            "smoothing_window": SMOOTH_WINDOW,
            "theta": THETA,
            "theta_low": THETA_LOW,
        },
        "attention_monotone_example": {
            "FI_A_0.2": fatigue_index(0.2, 4.4, 0.0),
            "FI_A_0.8": fatigue_index(0.8, 4.4, 0.0),
        },
        "entropy_low_penalty_example": {
            "FI_E_1.0": fatigue_index(0.5, 1.0, 0.0),
            "FI_E_4.4": fatigue_index(0.5, 4.4, 0.0),
        },
        "drift_monotone_example": {
            "FI_D_10": fatigue_index(0.5, 4.4, 10.0),
            "FI_D_80": fatigue_index(0.5, 4.4, 80.0),
        },
        "worked_example": fatigue_index(0.5, 4.4, 20.0),
        "csv": str(path),
    }
    (outdir / "claim1_formula_audit.json").write_text(json.dumps(checks, indent=2))
    return checks


def hotpot_samples(n: int) -> list[dict[str, str]]:
    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa"
    except Exception:
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        url = "https://huggingface.co/datasets/hotpot_qa"
    samples = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        ctx = ex.get("context", {})
        parts: list[str] = []
        if isinstance(ctx, dict):
            titles = ctx.get("title", [])
            sentences = ctx.get("sentences", [])
            for title, sents in zip(titles[:4], sentences[:4]):
                joined = " ".join(sents[:3] if isinstance(sents, list) else [str(sents)])
                parts.append(f"{title}: {joined}")
        prompt = (
            "Answer the question using the evidence. Keep the answer concise.\n\n"
            f"Evidence:\n{' '.join(parts)}\n\nQuestion: {ex['question']}\nAnswer:"
        )
        samples.append({"dataset": "HotpotQA", "prompt": prompt, "answer": str(ex.get("answer", "")), "source_url": url})
    return samples


def squad_samples(n: int) -> list[dict[str, str]]:
    ds = load_dataset("rajpurkar/squad", split="validation")
    samples = []
    for ex in ds.select(range(min(n, len(ds)))):
        answer = ex.get("answers", {}).get("text", [""])[0]
        prompt = (
            "Answer the question using the context. Keep the answer concise.\n\n"
            f"Context: {ex['context']}\n\nQuestion: {ex['question']}\nAnswer:"
        )
        samples.append(
            {
                "dataset": "SQuAD",
                "prompt": prompt,
                "answer": str(answer),
                "source_url": "https://huggingface.co/datasets/rajpurkar/squad",
            }
        )
    return samples


def trivia_samples(n: int) -> list[dict[str, str]]:
    try:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        url = "https://huggingface.co/datasets/mandarjoshi/trivia_qa"
    except Exception:
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        url = "https://huggingface.co/datasets/trivia_qa"
    samples = []
    for ex in ds.select(range(min(n, len(ds)))):
        ans = ex.get("answer", {})
        value = ans.get("value", "") if isinstance(ans, dict) else str(ans)
        prompt = f"Answer the trivia question concisely.\n\nQuestion: {ex['question']}\nAnswer:"
        samples.append({"dataset": "TriviaQA", "prompt": prompt, "answer": str(value), "source_url": url})
    return samples


def summarize_sequence(dataset: str, i: int, probed: dict[str, Any]) -> SequenceResult:
    traj = probed["trajectory"]
    fis = [r["fi"] for r in traj]
    smoothed = moving_average(fis)
    mean = lambda key: float(np.mean([r[key] for r in traj])) if traj else float("nan")
    return SequenceResult(
        dataset=dataset,
        sample_id=i,
        prompt_len=int(probed["prompt_len"]),
        generated_len=len(probed["generated_ids"]),
        text=probed["text"][:500],
        rep_ratio=repetition_ratio(probed["generated_ids"]),
        mean_fi=float(np.mean(fis)) if fis else float("nan"),
        first20_fi=float(np.mean(fis[:20])) if fis else float("nan"),
        mean_phi_a=mean("phi_a"),
        mean_phi_e=mean("phi_e"),
        mean_phi_d=mean("phi_d"),
        mean_attention=mean("attention"),
        mean_entropy=mean("entropy"),
        mean_drift=mean("drift"),
        static_flips=count_static_flips(smoothed),
        hysteresis_flips=count_hysteresis_flips(smoothed),
    )


def qa_eval(
    outdir: Path,
    model_id: str,
    n_hotpot: int,
    n_trivia: int,
    n_squad: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    tok, model = load_lm(model_id, need_attention=True)
    all_samples = hotpot_samples(n_hotpot) + trivia_samples(n_trivia) + squad_samples(n_squad)
    rows: list[dict[str, Any]] = []
    trajectories = []
    t0 = time.time()
    for i, sample in enumerate(all_samples):
        probed = manual_probe_generate(
            model,
            tok,
            sample["prompt"],
            max_new_tokens=max_new_tokens,
            seed=seed + i,
            need_attention=True,
            need_hidden=True,
        )
        seq = summarize_sequence(sample["dataset"], i, probed)
        row = asdict(seq)
        row["model"] = model_id
        row["dataset_url"] = sample["source_url"]
        rows.append(row)
        for tr in probed["trajectory"]:
            trr = {"dataset": sample["dataset"], "sample_id": i, "model": model_id, **tr}
            trajectories.append(trr)
        if (i + 1) % 10 == 0:
            print(f"qa progress {i+1}/{len(all_samples)} elapsed={time.time()-t0:.1f}s", flush=True)
    cleanup_model(tok, model)
    df = pd.DataFrame(rows)
    trdf = pd.DataFrame(trajectories)
    df.to_csv(outdir / "qa_sequence_metrics.csv", index=False)
    trdf.to_csv(outdir / "qa_token_trajectories.csv", index=False)
    summary: dict[str, Any] = {"model": model_id, "max_new_tokens": max_new_tokens, "datasets": {}, "n_total": int(len(df))}
    for ds_name, sub in df.groupby("dataset"):
        rho, p = spearmanr(sub["mean_fi"], sub["rep_ratio"])
        rho20, p20 = spearmanr(sub["first20_fi"], sub["rep_ratio"])
        static = float(sub["static_flips"].mean())
        hyst = float(sub["hysteresis_flips"].mean())
        reduction = 100.0 * (static - hyst) / static if static > 0 else float("nan")
        summary["datasets"][ds_name] = {
            "n": int(len(sub)),
            "mean_fi": float(sub["mean_fi"].mean()),
            "mean_repetition_ratio": float(sub["rep_ratio"].mean()),
            "spearman_full": float(rho),
            "spearman_full_p": float(p),
            "spearman_first20": float(rho20),
            "spearman_first20_p": float(p20),
            "static_flips_per_gen": static,
            "hysteresis_flips_per_gen": hyst,
            "hysteresis_reduction_pct": reduction,
        }
    hot = df[df["dataset"] == "HotpotQA"].copy()
    severe = (hot["rep_ratio"] >= 0.6).astype(int)
    if severe.nunique() < 2:
        cutoff = hot["rep_ratio"].quantile(0.75)
        severe = (hot["rep_ratio"] >= cutoff).astype(int)
    auroc: dict[str, float] = {}
    if severe.nunique() >= 2:
        for col, label in [
            ("mean_fi", "fatigue_index"),
            ("mean_phi_e", "entropy_only"),
            ("mean_phi_d", "drift_only"),
            ("mean_phi_a", "attention_inverse"),
        ]:
            auroc[label] = float(roc_auc_score(severe, hot[col]))
    else:
        for label in ["fatigue_index", "entropy_only", "drift_only", "attention_inverse"]:
            auroc[label] = float("nan")
    summary["hotpot_auroc"] = {
        "n": int(len(hot)),
        "severe_positive_rate": float(severe.mean()) if len(severe) else float("nan"),
        "threshold_note": "rep_ratio>=0.6; fell back to top quartile if one-class",
        **auroc,
    }
    (outdir / "qa_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def entropy_generation_metrics(model, tok, prompt: str, max_new_tokens: int, seed: int) -> dict[str, Any]:
    probed = manual_probe_generate(
        model,
        tok,
        prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
        need_attention=False,
        need_hidden=False,
    )
    ent = [r["entropy"] for r in probed["trajectory"]]
    return {
        "prompt_len": probed["prompt_len"],
        "generated_len": len(probed["generated_ids"]),
        "mean_entropy": float(np.mean(ent)) if ent else float("nan"),
        "tail_entropy": float(np.mean(ent[-max(1, len(ent) // 4) :])) if ent else float("nan"),
        "rep_ratio": repetition_ratio(probed["generated_ids"]),
        "text": probed["text"][:300],
    }


def scaling_eval(outdir: Path, n_samples: int, max_new_tokens: int, seed: int) -> dict[str, Any]:
    samples = hotpot_samples(n_samples)
    models = [
        ("tiiuae/falcon-rw-1b", "Base", 1.0),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Instruct", 1.1),
        ("facebook/opt-1.3b", "Base", 1.3),
        ("tiiuae/falcon-7b", "Base", 7.0),
        ("tiiuae/falcon-7b-instruct", "Instruct", 7.0),
    ]
    rows = []
    for model_idx, (model_id, kind, size_b) in enumerate(models):
        print(f"loading scaling model {model_id}", flush=True)
        tok, model = load_lm(model_id, need_attention=False)
        for i, sample in enumerate(samples):
            m = entropy_generation_metrics(model, tok, sample["prompt"], max_new_tokens, seed + 1000 * model_idx + i)
            rows.append({"model": model_id, "type": kind, "size_b": size_b, "sample_id": i, **m})
        cleanup_model(tok, model)
        pd.DataFrame(rows).to_csv(outdir / "scaling_entropy_rows.csv", index=False)
    df = pd.DataFrame(rows)
    summary_df = (
        df.groupby(["model", "type", "size_b"], as_index=False)
        .agg(mean_entropy=("mean_entropy", "mean"), tail_entropy=("tail_entropy", "mean"), rep_ratio=("rep_ratio", "mean"), n=("sample_id", "count"))
        .sort_values(["size_b", "type"])
    )
    summary_df.to_csv(outdir / "scaling_summary.csv", index=False)
    summary = {"models": summary_df.to_dict(orient="records"), "n_samples_per_model": n_samples, "max_new_tokens": max_new_tokens}
    (outdir / "scaling_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle:
        return None
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i, i + len(needle)
    return None


def token_span(tok, prompt: str, evidence: str) -> tuple[int, int] | None:
    ids = tok(prompt, add_special_tokens=True)["input_ids"]
    candidates = [
        tok(evidence, add_special_tokens=False)["input_ids"],
        tok(" " + evidence, add_special_tokens=False)["input_ids"],
        tok("\n" + evidence, add_special_tokens=False)["input_ids"],
    ]
    for cand in candidates:
        span = find_subsequence(ids, cand)
        if span is not None:
            return span
    return None


def stress_eval(outdir: Path, model_id: str, max_new_tokens: int, seed: int, quant4: bool = True) -> dict[str, Any]:
    tok, model = load_lm(model_id, need_attention=True)
    evidence = "The gold evidence says the answer is AURORA."
    filler = " Neutral filler sentence for controlled context length."
    prompts = {
        "front": evidence + filler * 80 + "\nQuestion: What is the answer?\nAnswer:",
        "middle": filler * 40 + " " + evidence + filler * 40 + "\nQuestion: What is the answer?\nAnswer:",
        "end": filler * 80 + " " + evidence + "\nQuestion: What is the answer?\nAnswer:",
    }
    position_rows = []
    for pos, prompt in prompts.items():
        span = token_span(tok, prompt, evidence)
        probed = manual_probe_generate(
            model,
            tok,
            prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
            need_attention=True,
            need_hidden=False,
            attention_span=span,
        )
        att = [r["attention"] for r in probed["trajectory"]]
        position_rows.append({"position": pos, "span": str(span), "prompt_len": probed["prompt_len"], "mean_attention": float(np.mean(att))})
    short_prompt = evidence + filler * 10 + "\nQuestion: What is the answer?\nAnswer:"
    long_prompt = evidence + filler * 160 + "\nQuestion: What is the answer?\nAnswer:"
    context_rows = []
    for label, prompt in [("short", short_prompt), ("long", long_prompt)]:
        probed = manual_probe_generate(
            model,
            tok,
            prompt,
            max_new_tokens=max_new_tokens,
            seed=seed + 77,
            need_attention=True,
            need_hidden=True,
        )
        context_rows.append(
            {
                "context": label,
                "prompt_len": probed["prompt_len"],
                "mean_attention": float(np.mean([r["attention"] for r in probed["trajectory"]])),
                "mean_entropy": float(np.mean([r["entropy"] for r in probed["trajectory"]])),
                "mean_drift": float(np.mean([r["drift"] for r in probed["trajectory"]])),
            }
        )
    fp16_entropy_rows = []
    q_prompt = hotpot_samples(1)[0]["prompt"]
    fp16_probe = manual_probe_generate(model, tok, q_prompt, max_new_tokens=max_new_tokens, seed=seed + 88, need_attention=False, need_hidden=False)
    fp16_entropy_rows = [r["entropy"] for r in fp16_probe["trajectory"]]
    cleanup_model(tok, model)
    quant_summary: dict[str, Any] = {"available": False}
    if quant4:
        try:
            tok4, model4 = load_lm(model_id, need_attention=False, quant4=True)
            q4_probe = manual_probe_generate(model4, tok4, q_prompt, max_new_tokens=max_new_tokens, seed=seed + 88, need_attention=False, need_hidden=False)
            q4_entropy = [r["entropy"] for r in q4_probe["trajectory"]]
            quant_summary = {
                "available": True,
                "fp16_mean_entropy": float(np.mean(fp16_entropy_rows)),
                "nf4_mean_entropy": float(np.mean(q4_entropy)),
                "entropy_delta_nf4_minus_fp16": float(np.mean(q4_entropy) - np.mean(fp16_entropy_rows)),
                "model": model_id,
            }
            cleanup_model(tok4, model4)
        except Exception as exc:
            quant_summary = {"available": False, "error": repr(exc), "fp16_mean_entropy": float(np.mean(fp16_entropy_rows))}
    pos_df = pd.DataFrame(position_rows)
    ctx_df = pd.DataFrame(context_rows)
    pos_df.to_csv(outdir / "stress_position_attention.csv", index=False)
    ctx_df.to_csv(outdir / "stress_context_length.csv", index=False)
    front = float(pos_df[pos_df.position == "front"]["mean_attention"].iloc[0])
    mid = float(pos_df[pos_df.position == "middle"]["mean_attention"].iloc[0])
    end = float(pos_df[pos_df.position == "end"]["mean_attention"].iloc[0])
    summary = {
        "model": model_id,
        "position_attention": position_rows,
        "front_middle_ratio": front / max(mid, 1e-9),
        "front_end_ratio": front / max(end, 1e-9),
        "context_length": context_rows,
        "quantization": quant_summary,
    }
    (outdir / "stress_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def smoke_eval(outdir: Path, max_new_tokens: int, seed: int) -> dict[str, Any]:
    tok, model = load_lm("sshleifer/tiny-gpt2", need_attention=True)
    prompt = "Context: The answer is blue.\nQuestion: What color is the answer?\nAnswer:"
    probed = manual_probe_generate(model, tok, prompt, max_new_tokens=max_new_tokens, seed=seed, need_attention=True, need_hidden=True)
    seq = summarize_sequence("smoke", 0, probed)
    pd.DataFrame([asdict(seq)]).to_csv(outdir / "smoke_sequence_metrics.csv", index=False)
    pd.DataFrame(probed["trajectory"]).to_csv(outdir / "smoke_token_trajectory.csv", index=False)
    cleanup_model(tok, model)
    return {"model": "sshleifer/tiny-gpt2", "sequence": asdict(seq)}


def make_figures(outdir: Path) -> None:
    try:
        import plotly.express as px

        if (outdir / "qa_token_trajectories.csv").exists():
            tr = pd.read_csv(outdir / "qa_token_trajectories.csv")
            fig = px.line(tr.groupby(["dataset", "step"], as_index=False)["fi"].mean(), x="step", y="fi", color="dataset", title="Mean FI trajectories by dataset")
            fig.write_html(outdir / "qa_fi_trajectories.html", include_plotlyjs="cdn")
        if (outdir / "scaling_summary.csv").exists():
            sc = pd.read_csv(outdir / "scaling_summary.csv")
            fig = px.scatter(sc, x="size_b", y="mean_entropy", color="type", text="model", title="Scaling sweep: mean entropy")
            fig.update_traces(textposition="top center")
            fig.write_html(outdir / "scaling_entropy.html", include_plotlyjs="cdn")
        if (outdir / "stress_position_attention.csv").exists():
            st = pd.read_csv(outdir / "stress_position_attention.csv")
            fig = px.bar(st, x="position", y="mean_attention", title="Evidence-position attention")
            fig.write_html(outdir / "stress_position_attention.html", include_plotlyjs="cdn")
    except Exception as exc:
        print(f"figure generation skipped: {exc}", flush=True)


def upload_outputs(outdir: Path, repo_id: str) -> dict[str, str]:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(outdir), path_in_repo="outputs")
    return {"dataset_repo": f"https://huggingface.co/datasets/{repo_id}", "path": "outputs"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "gpu", "formula"], default="smoke")
    parser.add_argument("--outdir", default="outputs/fatigue_repro")
    parser.add_argument("--model", default="facebook/opt-2.7b")
    parser.add_argument("--hotpot", type=int, default=120)
    parser.add_argument("--trivia", type=int, default=80)
    parser.add_argument("--squad", type=int, default=80)
    parser.add_argument("--scaling-samples", type=int, default=24)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--stress-new", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--upload-repo", default="")
    parser.add_argument("--skip-quant4", action="store_true")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "mode": args.mode,
        "argv": vars(args),
        "device": device_name(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "paper": "https://arxiv.org/abs/2605.30981",
        "openreview": "https://openreview.net/forum?id=dE3Z3bfEzk",
        "model_urls": {
            "main": f"https://huggingface.co/{args.model}",
            "scaling": [
                "https://huggingface.co/tiiuae/falcon-rw-1b",
                "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "https://huggingface.co/facebook/opt-1.3b",
                "https://huggingface.co/tiiuae/falcon-7b",
                "https://huggingface.co/tiiuae/falcon-7b-instruct",
            ],
        },
        "dataset_urls": [
            "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
            "https://huggingface.co/datasets/mandarjoshi/trivia_qa",
            "https://huggingface.co/datasets/rajpurkar/squad",
        ],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    summary: dict[str, Any] = {"metadata": meta, "claim1": formula_audit(outdir)}
    if args.mode == "smoke":
        summary["smoke"] = smoke_eval(outdir, max_new_tokens=min(args.max_new, 8), seed=args.seed)
    elif args.mode == "gpu":
        summary["qa"] = qa_eval(outdir, args.model, args.hotpot, args.trivia, args.squad, args.max_new, args.seed)
        summary["scaling"] = scaling_eval(outdir, args.scaling_samples, max_new_tokens=min(args.max_new, 48), seed=args.seed)
        summary["stress"] = stress_eval(outdir, args.model, max_new_tokens=args.stress_new, seed=args.seed, quant4=not args.skip_quant4)
    make_figures(outdir)
    if args.upload_repo:
        summary["upload"] = upload_outputs(outdir, args.upload_repo)
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("FINAL_SUMMARY_JSON_START")
    print(json.dumps(summary, indent=2))
    print("FINAL_SUMMARY_JSON_END")


if __name__ == "__main__":
    main()

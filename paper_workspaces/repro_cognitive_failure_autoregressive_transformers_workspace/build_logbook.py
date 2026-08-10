#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


JOB_COMPLETED = "https://huggingface.co/jobs/Srishti280992/6a7051a7a00abefd4b28ef23"
JOB_CPU = "https://huggingface.co/jobs/Srishti280992/6a704e426b79c09949c20487"
JOB_A10G = "https://huggingface.co/jobs/Srishti280992/6a704e50a00abefd4b28ee9d"
JOB_A100 = "https://huggingface.co/jobs/Srishti280992/6a704fd1a00abefd4b28eed8"
JOB_CANCELED = "https://huggingface.co/jobs/Srishti280992/6a704fe4a00abefd4b28eeda"
JOB_FAILED = "https://huggingface.co/jobs/Srishti280992/6a705160a00abefd4b28ef13"
RESULTS = "https://huggingface.co/datasets/Srishti280992/cognitive-fatigue-repro-results"
SCRIPT = "https://huggingface.co/datasets/Srishti280992/cognitive-fatigue-repro-results/blob/main/scripts/reproduce_fatigue.py"
PAPER = "https://arxiv.org/abs/2605.30981"
OPENREVIEW = "https://openreview.net/forum?id=dE3Z3bfEzk"

PAGES = [
    "Executive summary",
    "Claim 1: Fatigue Index Formula",
    "Claim 2: Predictive Validity",
    "Claim 3: AUROC for Repetition Detection",
    "Claim 4: Non-monotonic Scaling",
    "Claim 5: Hysteresis Alerting",
    "Claim 6: Stress Tests",
    "Conclusion",
]


def run(*args: str) -> None:
    subprocess.run(["trackio", "logbook", *args], check=True)


def cell(page: str, title: str, body: str) -> None:
    run("cell", "markdown", body, "--page", page, "--title", title)


def main() -> None:
    base = Path("outputs/downloaded_hf_results/outputs")
    c1 = json.loads((base / "claim1_formula_audit.json").read_text())
    qa = json.loads((base / "qa_summary.json").read_text())
    sc = json.loads((base / "scaling_summary.json").read_text())
    st = json.loads((base / "stress_summary.json").read_text())

    for page in PAGES:
        run("page", page)

    executive = f"""I reproduced the six named claims with an independent implementation of the paper's Fatigue Index probes and a completed Hugging Face A100 Job. The formula claim reproduces exactly, but the headline empirical values mostly do not: on OPT-2.7B with 350 real QA generations, FI-repetition correlations were 0.283/0.593/0.386 rather than 0.848/0.820/0.856, and HotpotQA FI AUROC was 0.663 rather than 0.976. Hysteresis reduced alert flips, but by 42-71% rather than 91-93%; the scaling sweep confirmed sub-3B instruction collapse but contradicted the claimed 7B reversal; stress tests verified NF4 entropy collapse but did not reproduce the front-evidence 5-10x attention advantage. Main run: {JOB_COMPLETED}; result artifacts: {RESULTS}; reproduction script: {SCRIPT}.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | Formula audit; OPT-2.7B on 150 HotpotQA, 100 TriviaQA, 100 SQuAD generations; five-model scaling sweep with 20 HotpotQA prompts/model; position, context-length, FP16 vs NF4 stress probes | Paper scale: 27,405 QA sequences, 9 models from 1B-13B, 120-token decoding, full robustness appendix |
| Hardware | 1x A100-SXM4-80GB completed Job, plus CPU/A10G/A100 canaries | 1x A100-class GPU reported as sufficient for the paper's experiments |
| Compute time | Completed Job ran about 12 minutes wall time after setup retries; local smoke test ran on Apple MPS | Multi-hour full benchmark across all datasets/models |
| Cost | Approximately one small A100-hour fraction, plus short canaries/retries; exact billing API unavailable | Tens of A100-hours for a full 27k-sequence/9-model replication |
| Outcome | Claim 1 verified; Claims 2, 3, 4, 5 falsified on reported magnitude; Claim 6 mixed | Not attempted at full scale |

Primary sources and assets: {PAPER}, {OPENREVIEW}, {RESULTS}, {JOB_COMPLETED}, https://huggingface.co/facebook/opt-2.7b, https://huggingface.co/datasets/hotpotqa/hotpot_qa, https://huggingface.co/datasets/mandarjoshi/trivia_qa, https://huggingface.co/datasets/rajpurkar/squad."""
    cell("Executive summary", "Executive summary", executive)
    run("pin", "--page", "Executive summary")

    cell(
        "Claim 1: Fatigue Index Formula",
        "Formula and constants audit",
        f"""**Paper claim.** The Fatigue Index is FI_t = 0.40 phi_A(A_t) + 0.35 phi_E(E_t) + 0.25 phi_D(D_t), aggregating prompt-attention decay, entropy miscalibration, and embedding drift during autoregressive decoding.

**Verdict: verified.** The independent audit used the paper constants K=64, entropy band [3.8, 5.0], smoothing L=5, thresholds 0.50/0.40, and weights 0.40/0.35/0.25. Across {c1['n_points']} grid points, every FI value was bounded in [0,1], and the weights summed to {c1['weights_sum']:.1f}. The worked example FI(0.5, 4.4, 20.0) = {c1['worked_example']:.3f}; decreasing attention from 0.8 to 0.2 raised FI from {c1['attention_monotone_example']['FI_A_0.8']:.3f} to {c1['attention_monotone_example']['FI_A_0.2']:.3f}, and increasing drift from 10 to 80 raised FI from {c1['drift_monotone_example']['FI_D_10']:.3f} to {c1['drift_monotone_example']['FI_D_80']:.3f}.

Evidence files: {RESULTS}/tree/main/outputs/claim1_formula_audit.json and {RESULTS}/tree/main/outputs/claim1_formula_grid.csv. Code: {SCRIPT}. Paper: {PAPER}.""",
    )

    d = qa["datasets"]
    cell(
        "Claim 2: Predictive Validity",
        "Spearman predictive-validity replication",
        f"""**Paper claim.** Table 3 reports FI/repetition Spearman correlations of 0.848 on HotpotQA, 0.820 on TriviaQA, and 0.856 on SQuAD.

**Verdict: falsified at this scale.** On {qa['n_total']} real generations from https://huggingface.co/facebook/opt-2.7b, I measured HotpotQA rho={d['HotpotQA']['spearman_full']:.3f} (n={d['HotpotQA']['n']}), TriviaQA rho={d['TriviaQA']['spearman_full']:.3f} (n={d['TriviaQA']['n']}), and SQuAD rho={d['SQuAD']['spearman_full']:.3f} (n={d['SQuAD']['n']}). All three are well below the claimed values; TriviaQA is directionally strongest but still 0.227 below the paper's 0.820, and the other two are less than half the reported correlations.

First-20-token correlations were also modest: HotpotQA {d['HotpotQA']['spearman_first20']:.3f}, TriviaQA {d['TriviaQA']['spearman_first20']:.3f}, SQuAD {d['SQuAD']['spearman_first20']:.3f}. This used the paper's repetition-ratio proxy, not exact-answer F1, matching Table 3's stated target.

Run and assets: {JOB_COMPLETED}, {RESULTS}/tree/main/outputs/qa_summary.json, {RESULTS}/tree/main/outputs/qa_sequence_metrics.csv, {RESULTS}/tree/main/outputs/qa_token_trajectories.csv. Datasets: https://huggingface.co/datasets/hotpotqa/hotpot_qa, https://huggingface.co/datasets/mandarjoshi/trivia_qa, https://huggingface.co/datasets/rajpurkar/squad.""",
    )
    run("cell", "figure", "--page", "Claim 2: Predictive Validity", "--title", "Mean FI trajectories", "--html", "outputs/downloaded_hf_results/outputs/qa_fi_trajectories.html", "--raw", "outputs/downloaded_hf_results/outputs/qa_token_trajectories.csv")

    au = qa["hotpot_auroc"]
    cell(
        "Claim 3: AUROC for Repetition Detection",
        "HotpotQA AUROC replication",
        f"""**Paper claim.** Table 5 reports HotpotQA AUROC=0.976 for aggregated FI, with entropy-only 0.954, drift-only 0.929, and inverse attention 0.307.

**Verdict: falsified.** On the same OPT-2.7B HotpotQA run, FI AUROC was {au['fatigue_index']:.3f}, far below 0.976. Entropy-only AUROC was {au['entropy_only']:.3f}, which outperformed the aggregate FI rather than being outperformed by it; drift-only was {au['drift_only']:.3f}, and inverse attention was {au['attention_inverse']:.3f}.

Protocol note: no generated sequence reached the paper's severe-degeneration threshold of repetition ratio >=0.6, so the script fell back to the top quartile to avoid a one-class AUROC calculation; positive rate was {au['severe_positive_rate']:.2f}. Under the literal threshold, the severe-degeneration detector has no positives in this scaled run, which itself argues against the claimed HotpotQA failure prevalence.

Run and assets: {JOB_COMPLETED}, {RESULTS}/tree/main/outputs/qa_summary.json, {RESULTS}/tree/main/outputs/qa_sequence_metrics.csv. Model: https://huggingface.co/facebook/opt-2.7b. Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa.""",
    )

    rows = sc["models"]
    table = "\n".join(
        f"| {r['model']} | {r['type']} | {r['size_b']:.1f} | {r['mean_entropy']:.3f} | {r['tail_entropy']:.3f} | {r['rep_ratio']:.3f} |"
        for r in rows
    )
    cell(
        "Claim 4: Non-monotonic Scaling",
        "Five-model scaling sweep",
        f"""**Paper claim.** Instruction-tuned models below 3B collapse faster than base models, but the trend reverses at 7B for Falcon-7B.

**Verdict: partially supported below 3B, falsified for the claimed 7B reversal.** Lower entropy indicates stronger entropy collapse. TinyLlama-1.1B-Chat had mean entropy 1.118, much lower than Falcon-RW-1B base at 2.439 and OPT-1.3B base at 2.492, supporting the sub-3B instruction-collapse direction. At 7B, however, Falcon-7B-Instruct had mean entropy 0.777 versus Falcon-7B base at 2.293, so instruction tuning still collapsed more; the reported reversal did not reproduce.

| Model | Type | Size (B) | Mean entropy | Tail entropy | Repetition |
|---|---:|---:|---:|---:|---:|
{table}

Run and assets: {JOB_COMPLETED}, {RESULTS}/tree/main/outputs/scaling_summary.json, {RESULTS}/tree/main/outputs/scaling_summary.csv. Models: https://huggingface.co/tiiuae/falcon-rw-1b, https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0, https://huggingface.co/facebook/opt-1.3b, https://huggingface.co/tiiuae/falcon-7b, https://huggingface.co/tiiuae/falcon-7b-instruct.""",
    )
    run("cell", "figure", "--page", "Claim 4: Non-monotonic Scaling", "--title", "Scaling entropy sweep", "--html", "outputs/downloaded_hf_results/outputs/scaling_entropy.html", "--raw", "outputs/downloaded_hf_results/outputs/scaling_summary.csv")

    cell(
        "Claim 5: Hysteresis Alerting",
        "Alert-flip reduction audit",
        f"""**Paper claim.** Hysteresis with activation theta=0.50 and deactivation theta_low=0.40 reduces false-positive alert flips by 91-93% compared with a static threshold.

**Verdict: mechanism verified, magnitude falsified.** Hysteresis reduced flips on every dataset, but not by the claimed amount: HotpotQA static {d['HotpotQA']['static_flips_per_gen']:.2f} to hysteresis {d['HotpotQA']['hysteresis_flips_per_gen']:.2f} ({d['HotpotQA']['hysteresis_reduction_pct']:.1f}% reduction), TriviaQA {d['TriviaQA']['static_flips_per_gen']:.2f} to {d['TriviaQA']['hysteresis_flips_per_gen']:.2f} ({d['TriviaQA']['hysteresis_reduction_pct']:.1f}%), and SQuAD {d['SQuAD']['static_flips_per_gen']:.2f} to {d['SQuAD']['hysteresis_flips_per_gen']:.2f} ({d['SQuAD']['hysteresis_reduction_pct']:.1f}%).

The deadband works directionally, but the paper's 91-93% reduction did not reproduce on OPT-2.7B, 350 real QA generations, and 64-token decoding. Run and assets: {JOB_COMPLETED}, {RESULTS}/tree/main/outputs/qa_summary.json, {RESULTS}/tree/main/outputs/qa_token_trajectories.csv.""",
    )

    q = st["quantization"]
    cell(
        "Claim 6: Stress Tests",
        "Position, context, and quantization probes",
        f"""**Paper claim.** Front-positioned evidence receives 5-10x higher attention than middle/end evidence, and 4-bit quantization increases entropy-collapse depth relative to FP16.

**Verdict: mixed; quantization verified, position/context attention not reproduced.** The NF4 probe reproduced deeper entropy collapse: mean entropy dropped from FP16 {q['fp16_mean_entropy']:.3f} to NF4 {q['nf4_mean_entropy']:.3f}, delta {q['entropy_delta_nf4_minus_fp16']:.3f}. However, the controlled evidence-position probe did not show the claimed front advantage: front/middle ratio was {st['front_middle_ratio']:.2f}x and front/end ratio was {st['front_end_ratio']:.2f}x, with end evidence receiving the highest measured attention in this setup. Context-length attention also did not collapse to near-zero by this prompt-slice metric: short-context attention was {st['context_length'][0]['mean_attention']:.3f}, long-context attention was {st['context_length'][1]['mean_attention']:.3f}.

Position rows: front attention {st['position_attention'][0]['mean_attention']:.5f}, middle {st['position_attention'][1]['mean_attention']:.5f}, end {st['position_attention'][2]['mean_attention']:.5f}. These stress probes used https://huggingface.co/facebook/opt-2.7b and the same completed A100 Job, {JOB_COMPLETED}. Assets: {RESULTS}/tree/main/outputs/stress_summary.json, {RESULTS}/tree/main/outputs/stress_position_attention.csv, {RESULTS}/tree/main/outputs/stress_context_length.csv.""",
    )
    run("cell", "figure", "--page", "Claim 6: Stress Tests", "--title", "Evidence-position attention", "--html", "outputs/downloaded_hf_results/outputs/stress_position_attention.html", "--raw", "outputs/downloaded_hf_results/outputs/stress_position_attention.csv")

    cell(
        "Conclusion",
        "Overall findings and reproducibility notes",
        f"""| Claim | Verdict | Evidence |
|---|---|---|
| 1: FI formula | Verified | Bounded 150-point audit; weights sum to 1; worked example FI=0.300 |
| 2: Predictive validity | Falsified | rho values 0.283/0.593/0.386 vs 0.848/0.820/0.856 |
| 3: HotpotQA AUROC | Falsified | FI AUROC 0.663 vs 0.976; entropy-only 0.722 beat FI |
| 4: Scaling | Partially supported then falsified | Sub-3B instruction collapse reproduced; 7B Falcon reversal contradicted |
| 5: Hysteresis | Directionally verified, magnitude falsified | Flip reductions 41.8-70.6%, not 91-93% |
| 6: Stress tests | Mixed | NF4 entropy collapse verified; front-evidence 5-10x and long-context near-zero attention not reproduced |

The strongest reproducibility note is that the paper's formula is precise enough to audit, but several empirical claims are sensitive to protocol choices: generation length, repetition threshold, prompt design, and hidden/attention extraction method. This reproduction used an independent script rather than official code; no official GitHub repository was found in the paper text or OpenReview page. The completed GPU run, result files, and script are all linked here: {JOB_COMPLETED}, {RESULTS}, {SCRIPT}.

Infrastructure touched during the reproduction: CPU canary {JOB_CPU}, A10G canary {JOB_A10G}, A100 canary {JOB_A100}, canceled uv launch {JOB_CANCELED}, failed CUDA-image environment trial {JOB_FAILED}, completed A100 run {JOB_COMPLETED}. Other linked assets include {PAPER}, {OPENREVIEW}, https://github.com/gradio-app/posterly, and all Hub models/datasets named on the claim pages.""",
    )


if __name__ == "__main__":
    main()

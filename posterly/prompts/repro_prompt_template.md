# Agent Prompt Template

First read the challenge instructions:

```bash
curl -sL https://huggingface.co/datasets/ICML-2026-agent-repro/challenge/resolve/main/README.md
```

Your job is to reproduce the ICML 2026 paper:

- Submission number: `<submission-number>`
- Title: `<paper-title>`
- OpenReview id: `<openreview-id>`
- arXiv id, if any: `<arxiv-id>`

Major claims to verify:

- Work only inside `paper_workspaces/repro_<paper>_workspace`. Keep routes, model ladders, and paper-specific launchers in that workspace; never add paper identities or special cases to core agent code.
- Pin the paper, code, dataset, and checkpoint revisions in `.repro/source_manifest.json` before extracting claims.
- Complete `.repro/claims.json` using the generic claim contract. Every claim needs an exact source anchor, evidence modality, metric, comparator, paper-scale requirement, fidelity requirement, data-integrity requirement, controls, statistical adequacy, and symmetric verification/falsification conditions.
- Identify each major empirical, theoretical, or benchmark claim.
- Convert each major claim into a claim ledger row before running experiments: required evidence, available artifacts, reproduction route, pass condition, output file, and expected blocker if any.
- Exhaust released artifacts first. Prefer exact aggregate recomputation over expensive reruns when public labels, scores, logs, metadata, or result tables already identify the claim.
- Build one deterministic verifier script that emits `summary.json`, per-claim CSV/JSON outputs, raw tables for figures, and executable pass/fail checks.
- Create one Trackio logbook page per claim when possible.
- Run a local smoke test first.
- Treat local runs as dependency/data smoke tests. If local execution is much slower than expected, numerically unstable, or blocked by host-specific solver/model issues, stop local escalation and move to a checkpointed HF Job.
- For each substantive empirical claim where feasible, run at least one scaled experiment using Hugging Face Jobs.
- For agent/LLM systems papers, distinguish the research contribution from the proprietary backend. If the backbone model, search API, or hosted service is not the paper's contribution, a documented similar-class substitution via Hugging Face Inference Providers or a self-hosted backend such as vLLM/llama.cpp can still be a faithful full reproduction, not a toy reproduction.
- For theoretical claims, do not default to a toy simulation. First reproduce the proof route directly: source theorem anchors, assumptions, proof obligations, algebraic/symbolic checks, boundary cases, and any executable verifier for exponents, inequalities, reductions, or recurrences. Use simulations only as sanity checks unless the theorem itself is empirical.
- If a small HF Job can independently validate the aggregate verifier, run it and record the job URL even when no GPU-heavy rerun is required.
- Before launching any long HF Job or sweep, make it resumable: split by the natural experimental unit such as dataset, seed, model, fold, target size, scenario, or claim; write per-unit raw and summary outputs; and upload/sync after each completed unit. A canceled job should still leave usable artifacts for completed units.
- Run both confirmation checks and counterexample searches. Do not stop at the authors' happy path: sweep seeds, dimensions, edge cases, perturbations, dataset splits, model settings, or aggregation formulas when those are relevant to the claim.
- Reject tautological checks. The prediction and observation must come from independently implemented routes; theoretical audits must include a non-vacuity test, assumption activation, boundary/adversarial search, or a construction that could fail.
- Treat row identity, denominators, split hashes, dataset/checkpoint revisions, preprocessing parity, cohort selection, and train/evaluation leakage as claim-bearing evidence. Use cross-fitting or held-out evaluation whenever learned nuisance components could see evaluation rows.
- Match the paper's algorithm, architecture/checkpoint class, metric, comparator, and scale. A smaller run needs a predeclared adequacy argument and cannot silently satisfy a full-scope empirical claim.
- Use paired randomness and same-condition comparisons where possible. Include null, negative, destructive, or no-intervention controls appropriate to the claim, and evaluate every required cell rather than averaging away a failure.
- Clearly label toy or proxy reproductions as `toy` and state the blocker. Only use `toy` when there is no credible way to validate the claim at the stated paper scope.
- If an external public reproduction or prior high-scoring logbook exists, use it as guidance for coverage, settings, and presentation. Do not present it as your evidence unless you independently rerun the relevant checks or clearly label it as an external reference.
- Publish scripts, logs, outputs, generated data, checkpoints, and plots as Trackio artifacts.
- Add and pin an executive summary, then add and pin the poster.
- Publish the Trackio logbook and print the link.

Claim-to-test protocol:
- For each claim, write the claim as a falsifiable statement.
- Create a claim ledger with columns: claim, required evidence, released artifacts, method, pass condition, outputs, verdict, and scope.
- Define the expected result numerically or mechanically. Examples: theorem predicts slope approximately 0.5; algorithm should be invariant to X; bound should hold over random instances; reported Table 1 aggregation should equal 22.1%; method should beat baseline under the stated synthetic setup.
- Inventory released artifacts before launching compute: CSVs, JSONL, dashboards, evaluation logs, model outputs, official tables, seeds, configs, metadata, and checkpoints.
- When artifacts are sufficient, reproduce exact denominators, seeds, confidence intervals, AUROC/accuracy cells, aggregate tables, behavior/category breakdowns, or other paper-level summaries directly from them.
- Name the confirmation test: the paper-faithful run, recomputation, theorem check, benchmark subset, or static code/config check that should support the claim.
- Name the counterexample search: seeds, dimensions, parameters, edge cases, alternative aggregations, ablations, null baselines, or adversarial inputs that could break the claim.
- Route the claim before spending compute: deterministic recomputation for arithmetic, local/HF CPU jobs for released-code behavior, HF GPU jobs only for real training/eval evidence, and multi-model LLM matrices only for LLM-agent performance claims.
- For theorem/proof claims, route before compute as a `proof audit`: cite theorem and lemma source locations, list assumptions, identify each proof obligation, and build deterministic checks for algebraic steps where possible. Label empirical runs as `sanity check` when they do not instantiate the theorem's full scope.
- Use the shared claim router for new papers when possible: `local_command` for deterministic checks, `hf_job_command` for one remote job, and `matrix_hf_job` for seed/model/dataset sweeps. Treat paper-specific harnesses as configured commands, not as special agent logic.
- For matrix or sweep jobs, design partial success into the runner: persist each completed unit before starting the next one, make final aggregation optional, and include enough metadata to rebuild aggregate tables from per-unit outputs if the job stops early.
- For LLM-agent claims, run a model/mode/scenario matrix rather than one model on one happy-path setting. Escalate models only when the cheaper/current route leaves the claim unresolved.
- When substituting a model/backend, record the original backend, replacement backend, model size/class/capability rationale, provider or self-hosting route, expected effect on results, and exact model/revision/API endpoint. Do not label the run `toy` solely because of a documented comparable backend swap; reserve `toy` for reduced data, reduced task scope, proxy tasks, or models far below the original class.
- Record exact commands, configs, seeds, scope, and artifact paths.
- If the evidence breaks the claim, call it `falsified` or `contradicted`; a clean falsification is valuable evidence.
- Build `.repro/plan.json` in cheapest-decisive-first order. Run `repro-agent precheck --workspace <workspace> --stage plan` before paid compute.
- Record each natural execution unit with `repro-agent record-run`; a successful unit is valid only when its expected artifacts exist and their hashes are recorded.
- Compile the final per-claim result into `.repro/evidence.json`, then require `repro-agent precheck --workspace <workspace> --stage evidence` to pass before rendering or publishing.

Verdict-first logbook pages:
- Start every claim page with one of `VERIFIED`, `FALSIFIED`, `TOY`, or `INCONCLUSIVE`.
- Put the exact metric or pass/fail result in the first paragraph.
- Lead with the strongest completed evidence. Put smoke runs, failed attempts, canceled jobs, and setup notes after the verdict as scope, caveats, or artifacts so they do not drown out successful scaled evidence.
- Include denominators, seeds, model/dataset/attack splits, confidence intervals, and all relevant cells when those determine the score.
- Include the exact command that generated the evidence.
- State the exact scope: full paper scope, released-code scope, CPU-only recomputation, HF GPU job, static inspection, or toy/proxy.
- If a backend/model substitution was used, state it in the first claim page paragraph or scope table and link the replacement Hub model/provider/self-hosted deployment evidence.
- Link or list the artifact bundle: JSON/CSV outputs, logs, plots, job ids, checkpoints, generated data, and scripts.
- Avoid ambiguous phrasing such as "looks good" or "partially works" without a metric and blocker.

Before publishing:
- For each claim, state one of: verified, falsified/contradicted, inconclusive, toy/proxy only.
- Prefer at least one real execution artifact per non-documentation claim.
- If a claim is only static inspection or arithmetic recomputation, say so explicitly.
- Add raw figure/table cells for the key outputs so readers and future agents can fetch the numbers without rerunning.
- After any judge feedback, append a new revision cell instead of replacing old evidence.
- When responding to judge feedback, translate each scoring objection into a concrete missing-evidence gap, map it to exact knobs such as seeds, replicates, models, datasets, target sizes, or compute route, and make the new revision cell state what changed from the previous run.
- If the judge calls evidence toy/proxy, first check whether the evidence modality is wrong, not only whether the scale is too small. For example, a continuous-domain theorem may need a proof-algebra audit rather than a denser grid simulation.
- If a judge or extracted claim mentions a theorem component that is absent from the paper, audit the source text and label it as a claim-extraction error instead of silently treating it as an unverified paper result.
- Pin the newest executive summary.
- Record a report manifest only after all deterministic evidence gates pass. Report artifacts are content-addressed, so changing claims, plans, evidence, pages, or poster content invalidates downstream stages.
- Verify the public static logbook URL contains the newest job id/results, newest verdict text, and no stale toy-only summaries when stronger evidence has replaced them. It must not contain `trackio-local-path://`, `trackio-artifact://`, stale placeholders, or old corrupted text.
- A failed precheck, self-check, publish, or public read-back is blocking. Never convert missing judge output or unavailable infrastructure into a successful gate.
- Ingest judge output as structured fix tickets. A ticket asks for decisive evidence—verified or falsified—and names the minimal experiment delta. Do not republish unchanged evidence.
- Preserve every previously decisive claim during repair. Per-claim non-regression is required; a higher aggregate score cannot hide a degraded claim.
- If `trackio logbook publish` fails after the static logbook is generated, try a direct upload to the already-created Space and then run the public read-back check again.

Do not modify benchmark prompts, hidden configs, oracle designs, or evaluation thresholds unless the logbook explicitly labels the run as an intervention/ablation rather than a faithful reproduction.

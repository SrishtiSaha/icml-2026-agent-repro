The reproduction verifies the three theoretical claims by independent numerical audits of the finite-precision and mean-field counting formulas, and falsifies the three empirical claims as literally stated. The paper's own source figures show cramming linear-fit `R^2` values below `0.995` for Gemma-3-270M and Llama-3.2-1B, Table 1 contains ratios outside `5-20x`, and the copying-task figure contains weak sigmoid fits (`R^2=0.46` and `0.70`). I used the arXiv PDF/source bundle, local audit scripts, rendered source figures, and one small T4 GPU Job that smoke-tested soft-prompt cramming on https://huggingface.co/sshleifer/tiny-gpt2. Total wall-clock compute was minutes locally plus an 8-minute-capped T4-small job, approximately `$0.05` at the timeout cap using the published `$0.40/hour` T4-small rate.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | Numerical audits for theorems; primary-source audit of Figure 2c, Table 1, and copying Figure; one tiny cramming smoke job | Re-run all paper cramming/copying experiments across Pythia, Qwen-2.5, Llama-3.2, and Gemma-3 model families |
| Hardware | Local CPU plus one HF T4-small Job | Paper reports 2 NVIDIA H100 GPUs for cramming experiments |
| Compute time | Minutes locally; HF Job capped at 8 minutes | Multi-model prompt optimization and fine-tuning, likely many GPU-hours |
| Cost | About `$0.05` estimated at the T4-small timeout cap | Not attempted; H100 multi-model replication would be much higher |
| Outcome | 3 theoretical claims verified; 3 empirical claims falsified literally | Full-scale rerun not attempted |

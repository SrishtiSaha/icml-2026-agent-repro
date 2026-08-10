Verdict: verified under the paper's finite Wasserstein precision / mean-field assumption.

The Corollary 4.9 threshold does not contain the prompt length `m`: the audited threshold was a function of `d`, `r`, `epsilon`, `q`, and `|V|` only. Across four settings, the first integer beyond the mean-field threshold had accessible-fraction cap below one, and every additional token again multiplied the cap by `1/|V|`.

This supports the prompt-independent accessibility-threshold arithmetic. It does not independently validate the modeling assumption that a concrete transformer admits the exact finite Wasserstein precision used by the corollary.

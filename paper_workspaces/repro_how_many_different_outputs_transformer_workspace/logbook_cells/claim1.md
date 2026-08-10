Verdict: verified as a numerical audit of the counting formula, not as a proof replacement.

The arXiv source bundle for https://arxiv.org/abs/2605.22223 states Theorem 4.4 as a finite-precision packing bound: a transformer with precision `epsilon`, embedding radius `r`, dimension `d`, and prompt length `m` can access at most `(1+2r/epsilon)^(d*m)` distinct output sequences. The audit in `audit_theory_claims.py` swept six `(d,m,r,epsilon)` settings and explicitly enumerated small finite grids; the maximum relative error between enumeration and the formula was `8.66e-16`.

Control: halving `epsilon` increased the finite grid capacity in every setting, showing the bound depends on the finite precision condition as expected. Source materials used: https://arxiv.org/e-print/2605.22223 and https://arxiv.org/pdf/2605.22223.

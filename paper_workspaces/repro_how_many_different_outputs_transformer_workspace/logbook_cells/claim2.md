Verdict: verified for the finite-prompt threshold arithmetic.

Corollary 4.5 follows by comparing the finite prompt capacity `(1+2r/epsilon)^(d*m)` to the full sequence space `|V|^n`. In five independent settings, the first integer `n` above `(d*ln(1+2r/epsilon)/ln(|V|))*m` always had an accessible-fraction cap below one, and each additional generated token multiplied that cap by exactly `1/|V|`.

The figure cell below plots representative finite-prompt decay caps on a log scale. This audit checks the stated counting implication and the exponential rate, not empirical cramming optimization.

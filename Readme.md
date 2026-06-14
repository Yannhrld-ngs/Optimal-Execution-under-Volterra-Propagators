# Optimal Portfolio Selection and Cross-Asset Market Impact under Volterra Propagators

A discrete-time solver and numerical study of optimal portfolio selection under
**transient cross-impact** (Volterra propagators), **temporary impact**, and an
**alpha signal**. The project reformulates the continuous-time problem of
Abi Jaber, Neuman & Tuschmann (2026) as a finite-dimensional quadratic program
and reproduces its main qualitative findings.

## Overview

Under a general (nonnegative-definite Volterra) propagator the optimal portfolio
problem is non-Markovian and time-inconsistent, and its adapted solution requires
stochastic Fredholm equations. This project takes a simpler route:

- discretize the time horizon on a grid `t_k = k Δt`, `Δt = T/n`;
- assume a **deterministic** alpha signal (open-loop control);
- the risk–revenue functional then reduces to a concave quadratic program

  ```
  J(u) = -uᵀ Dₙ u + gₙᵀ u + const
  ```

  with closed-form maximizer

  ```
  u* = ½ Dₙ⁻¹ gₙ
  ```

When the signal is deterministic this open-loop control coincides with the
optimal adapted control, so no optimality is lost relative to the continuous-time
solution.

## Method

The block matrix `Dₙ` collects the four effects of the objective:

```
Dₙ = (Δt/2)·Iₙ⊗Λ̄                temporary impact
   + (Δt²/2)·(Gₙ + Gₙᵀ)         transient (cross-)impact
   + (γΔt³/2)·(LᵀL)⊗Σ           Markowitz risk
   + (ϱΔt²/2)·Jₙ⊗Π              terminal-inventory penalty
```

with `(Gₙ)_{kj} = G(t_k,t_j)·1_{k>j}`, `(LᵀL)_{ij} = n - max(k,j)`,
`Jₙ = 1ₙ1ₙᵀ`, and `Λ̄ = ½(Λ + Λᵀ)`.

The transient impact uses the factorized propagator `G(t,s) = C·φ(t−s)·1_{t>s}`
with three scalar kernels:

| Kernel       | φ(t)       | Note                                  |
|--------------|------------|---------------------------------------|
| No transient | 0          | frictionless benchmark                |
| Exponential  | e^(−ρt)    | Markovian, ρ = 0.5                    |
| Fractional   | t^(−α)     | power-law / long-memory, α = 0.25     |

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

```bash
pip install numpy matplotlib
```

## Usage

```python
import numpy as np
from lib import Portfolio   # adjust to your module name

# --- problem setup (Experiment 2: two assets, cross-impact, no signal) ---
N, T, n = 2, 10, 200
X0     = np.array([10, 0])
varrho = 4
gamma  = 0
Lambda = np.diag([0.03, 0.03])
Pi     = np.eye(N)
Sigma  = np.eye(N)                 # irrelevant when gamma = 0
C      = np.array([[0.06, 0.05],
                   [0.05, 0.06]])
E_dA   = np.zeros((n, N))          # no signal

ptf = Portfolio(N=N, T=T, n=n)
# u_star, X_star = ptf.build_optimal(...)  # adjust to your solver's API
```

> Method names above are illustrative — replace them with your actual
> `Portfolio` API.

## Experiments

| # | Theme                                   | Setup                                                |
|---|-----------------------------------------|------------------------------------------------------|
| 1 | Baseline (no signal, no cross-impact)   | diagonal `C`                                          |
| 2 | Cross-impact, no signal                 | off-diagonal `C` → transaction-triggered round trip  |
| 3 | Signal, fast-decaying Asset 1           | β = diag(0.9, 0.3)                                    |
| 4 | Signal, roles swapped                   | β = diag(0.3, 0.9)                                    |
| 5 | Three assets, chain cross-impact        | `C` couples 1–2 and 2–3 only                          |
| 6 | Correlation on liquidation              | γ = 0.3, uncorrelated vs. fully correlated Σ          |

## Reference

E. Abi Jaber, E. Neuman, S. Tuschmann.
*Optimal Portfolio Choice with Cross-Impact Propagators*, 2026.


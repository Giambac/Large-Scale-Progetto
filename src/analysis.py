"""
analysis.py — Bootstrap confidence interval pure function (ALAB-03, D-13).

No I/O. No SQL. No global state. Pure function only.

Used by examples/compute_ci.py (CLI table) and notebooks/analysis.ipynb
(interactive exploration). Deterministic resampling via a seeded
numpy.random.Generator — mirrors PairBag.sample()'s RNG-isolation pattern
(Phase 4 D-19, CLAUDE.md "never mutate the global random module").
"""
from __future__ import annotations

import numpy as np

from src.logging_setup import deviation


DEFAULT_N_BOOTSTRAP: int = 10_000
DEFAULT_CI: float = 0.95


def compute_bootstrap_ci(
    values: list[float],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Compute a percentile-based bootstrap confidence interval for the mean.

    Args:
        values:      Observed values (e.g. turns-to-convergence for one strategy).
                     MUST be non-empty (fail loudly per CLAUDE.md).
        n_bootstrap: Number of bootstrap resamples (default 10,000 — standard).
                     Values below 1000 fire deviation() (low statistical power).
        ci:          Confidence level in (0, 1). 0.95 = 95% CI (default).
        seed:        RNG seed for reproducibility. Default 0 — pass a different
                     value for sensitivity analysis.

    Returns:
        (lower_bound, upper_bound) percentile bracket of resampled means.

    Raises:
        AssertionError: if values is empty or ci is not in (0, 1).
    """
    assert len(values) > 0, "compute_bootstrap_ci called with empty values"
    assert 0 < ci < 1, f"ci must be in (0, 1), got {ci}"

    if n_bootstrap < 1000:
        deviation(
            "compute_bootstrap_ci called with low n_bootstrap — wide CI / low power",
            n_bootstrap=n_bootstrap,
            n_values=len(values),
        )

    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = arr.shape[0]

    # Vectorised resampling: shape (n_bootstrap, n) of integer indices.
    idx = rng.integers(low=0, high=n, size=(n_bootstrap, n))
    means = arr[idx].mean(axis=1)

    alpha = 1.0 - ci
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)

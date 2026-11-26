"""Deterministic bootstrap helpers used by the raw judge metrics."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence


def discrete_interval(values: Sequence[float]) -> tuple[float, float]:
    """Return the paper's discrete 95% bootstrap interval."""

    if not values:
        raise ValueError("cannot form an interval from no bootstrap values")
    ordered = sorted(values)
    count = len(ordered)
    return ordered[int(0.025 * count)], ordered[min(count - 1, int(0.975 * count))]


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    """Return the linearly interpolated percentile used for edit-level rates."""

    if not values:
        raise ValueError("cannot calculate a percentile from no values")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    index = quantile * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def clustered_counter_bootstrap(
    clusters: Sequence[Counter[str]],
    *,
    levels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Bootstrap categorical percentages by resampling complete sentences."""

    if not clusters:
        raise ValueError("bootstrap requires at least one cluster")
    if samples <= 0:
        raise ValueError("samples must be positive")
    encoded = [tuple(cluster[level] for level in levels) for cluster in clusters]
    distributions = {level: [] for level in levels}
    count = len(encoded)
    rng = random.Random(seed)
    for _ in range(samples):
        totals = [0] * len(levels)
        for _ in range(count):
            values = encoded[rng.randrange(count)]
            for index, value in enumerate(values):
                totals[index] += value
        denominator = sum(totals)
        if denominator == 0:
            raise ValueError("a bootstrap sample contains no categorical observations")
        for index, level in enumerate(levels):
            distributions[level].append(totals[index] / denominator * 100)
    return distributions


def mean_bootstrap(values: Sequence[float], *, samples: int, seed: int) -> list[float]:
    """Bootstrap a mean with the paper's seeded Python RNG."""

    if not values:
        raise ValueError("mean bootstrap requires at least one value")
    if samples <= 0:
        raise ValueError("samples must be positive")
    count = len(values)
    rng = random.Random(seed)
    return [sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples)]

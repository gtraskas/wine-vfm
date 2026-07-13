"""Value-for-money (VFM) target: log(points/price) on fixed analytic bounds, 0-99.

The raw ratio points/price is hyperbolically skewed, so we take
log(points/price) and min-max scale it against the ANALYTIC bounds implied by
the dataset curation constants — worst possible ratio (MIN_POINTS at
MAX_PRICE) to best possible ratio (MAX_POINTS at MIN_PRICE). Stateless and
deterministic: no fitting, no frozen artifact, fully reproducible from the
constants alone.

The frontier agent estimates points and price (quantities that exist in the
world), then passes them through this SAME transform — keeping its estimates
directly comparable with models that predict VFM natively.
"""

import numpy as np

VFM_MAX_SCORE: int = 99

# Curation bounds of the curated dataset (points 80-100, price $4-250)
MIN_POINTS: float = 80.0
MAX_POINTS: float = 100.0
MIN_PRICE: float = 4.0
MAX_PRICE: float = 250.0

# Analytic bounds of log(points/price) under the curation constants:
# worst = cheapest quality at the highest price; best = the inverse.
VFM_LOG_MIN: float = float(np.log(MIN_POINTS / MAX_PRICE))  # log(80/250) = -1.139
VFM_LOG_MAX: float = float(np.log(MAX_POINTS / MIN_PRICE))  # log(100/4) =  3.219

# Deterministic verdict bands on the 0-99 scale
BARGAIN_FLOOR: int = 70
OVERPRICED_CEILING: int = 35


def compute_vfm(points: float, price: float) -> int:
    """Map (points, price) to the 0-99 VFM scale.

    Args:
        points: Critic score (80-100 after curation).
        price: Bottle price in USD (4-250 after curation).

    Returns:
        Integer VFM score, clipped to [0, 99].
    """
    raw = float(np.log(points / price))
    scaled = (raw - VFM_LOG_MIN) / (VFM_LOG_MAX - VFM_LOG_MIN) * VFM_MAX_SCORE
    return int(np.clip(scaled, 0, VFM_MAX_SCORE))


def verdict(vfm: int) -> tuple[str, str]:
    """Deterministic verdict from a VFM score.

    Args:
        vfm: Value-for-money score on the 0-99 scale.

    Returns:
        (label, reason) — bargain / fair / overpriced with an explanation.
    """
    if vfm >= BARGAIN_FLOOR:
        return "bargain", f"VFM {vfm}/99 — strong value for the quality."
    if vfm <= OVERPRICED_CEILING:
        return "overpriced", f"VFM {vfm}/99 — paying for label, not quality."
    return "fair", f"VFM {vfm}/99 — priced about right for what it delivers."

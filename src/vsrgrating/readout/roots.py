"""Root-finding for functions known to increase."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

EXPANSIONS = 64
BISECTIONS = 100


def increasing_root(
	fn: Callable[[np.ndarray], ArrayLike], target: float, guess: np.ndarray
) -> np.ndarray:
	"""Where each coordinate of an elementwise increasing `fn` reaches `target`."""
	center = np.asarray(guess, dtype=np.float64)
	below = np.ones_like(center)
	above = np.ones_like(center)

	for _ in range(EXPANSIONS):
		short_low = _finite(fn(center - below)) >= target
		short_high = _finite(fn(center + above)) <= target
		if not (short_low.any() or short_high.any()):
			break

		below = np.where(short_low, 2.0 * below, below)
		above = np.where(short_high, 2.0 * above, above)
	else:
		raise ValueError(f"no bracket around the guess after {EXPANSIONS} doublings")

	low = center - below
	high = center + above
	for _ in range(BISECTIONS):
		mid = 0.5 * (low + high)
		rising = _finite(fn(mid)) < target
		low = np.where(rising, mid, low)
		high = np.where(rising, high, mid)

	return 0.5 * (low + high)


def _finite(values: ArrayLike) -> np.ndarray:
	"""A comparison against NaN is silently false and would close the wrong bracket."""
	out = np.asarray(values)
	if not np.isfinite(out).all():
		raise FloatingPointError(f"{int((~np.isfinite(out)).sum())} non-finite function values")

	return out

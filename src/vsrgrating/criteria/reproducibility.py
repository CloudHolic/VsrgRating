"""Q1: does the readout ordering survive refitting on a different half of the people?"""

import numpy as np
import polars as pl

from ..evaluation import Fitted
from . import readouts


def halves(n_persons: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
	"""Two disjoint person masks; with an odd count the second holds the extra person."""
	order = np.random.default_rng(seed).permutation(n_persons)
	first = np.zeros(n_persons, dtype=bool)
	first[order[: n_persons // 2]] = True

	return first, ~first


def table(
	seed: int, left: Fitted, right: Fitted, seen: np.ndarray, grid: tuple[float, ...]
) -> pl.DataFrame:
	"""One split's rank correlation per candidate, on items both halves saw and can read."""
	left_values, left_domain = readouts.of(left, grid)
	right_values, right_domain = readouts.of(right, grid)
	keep = left_domain & right_domain & seen
	rho = readouts.agreement(left_values, right_values, keep)

	return pl.DataFrame(
		{
			"seed": seed,
			"candidate": list(rho),
			"spearman": list(rho.values()),
			"items": int(keep.sum()),
		}
	)

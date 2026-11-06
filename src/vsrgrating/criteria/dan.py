"""Q4: do the readouts follow the dan course ladders?"""

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from .mapset import nm_index


def table(
	courses: dict[str, list[dict]],
	items: pl.DataFrame,
	values: dict[str, np.ndarray],
	domain: np.ndarray,
) -> pl.DataFrame:
	"""Rank correlation between each course's ladder and each readout."""
	lookup = nm_index(items)
	rows = []

	for course, steps in courses.items():
		order = np.array([lookup[s["beatmap_id"]] for s in steps if s["beatmap_id"] in lookup])
		ladder = np.flatnonzero(domain[order])
		kept = order[ladder]

		for name, value in values.items():
			rows.append(
				{
					"course": course,
					"candidate": name,
					"steps": kept.size,
					"spearman": float(spearmanr(value[kept], ladder).statistic),
				}
			)

	return pl.DataFrame(rows)

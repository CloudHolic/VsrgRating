"""Q3: do the readouts keep the order a mapper gave the difficulties of one mapset?"""

from itertools import combinations

import numpy as np
import polars as pl


def nm_index(items: pl.DataFrame) -> dict[int, int]:
	"""beatmap_id to item_idx for the NM items: a rate edit is not the mapper's order."""
	nm = items.filter(pl.col("rate_group") == "NM")
	return dict(zip(nm["beatmap_id"].to_list(), nm["item_idx"].to_list(), strict=True))


def pairs(units: list[dict], items: pl.DataFrame) -> np.ndarray:
	"""Every ordered pair inside a unit, easier first, shaped (n, 2)."""
	lookup = nm_index(items)
	return np.array(
		[pair for unit in units for pair in combinations([lookup[b] for b in unit["order"]], 2)]
	)


def table(values: dict[str, np.ndarray], domain: np.ndarray, ordered: np.ndarray) -> pl.DataFrame:
	"""Share of pairs whose readout puts the mapper's easier difficulty lower."""
	usable = ordered[domain[ordered[:, 0]] & domain[ordered[:, 1]]]
	first, second = usable[:, 0], usable[:, 1]

	return pl.DataFrame(
		{
			"candidate": list(values),
			"share": [float(np.mean(v[first] < v[second])) for v in values.values()],
			"pairs": usable.shape[0],
		}
	)

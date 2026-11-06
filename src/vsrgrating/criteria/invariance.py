"""Q2: do two groups of players put the items in the same order?"""

from dataclasses import replace

import numpy as np
import polars as pl

from ..data import Split
from ..evaluation import Fitted
from . import readouts
from .reproducibility import halves

SPLITS = ("random", "theta", "activity")


def groups(split: str, data: Split, theta: np.ndarray, seed: int) -> np.ndarray:
	"""True for group A: a random half, or above the median ability or activity."""
	if split == "random":
		group = halves(data.n_persons, seed)[0]
	else:
		activity = np.bincount(data.person, minlength=data.n_persons).astype(np.float64)
		scores = theta if split == "theta" else activity
		group = scores > np.median(scores)

	if not group.any() or group.all():
		raise ValueError(f"the {split} split puts all {group.size} people in one group")

	return group


def paired(data: Split, group: np.ndarray) -> Split:
	"""Every response, with item i split into 2i (below) and 2i + 1 (group A)."""
	item = data.item * 2 + group[data.person].astype(np.int64)
	return replace(data, item=item, n_items=2 * data.n_items)


def compare(split: str, fitted: Fitted, data: Split, grid: tuple[float, ...]) -> pl.DataFrame:
	"""The two copies' agreement per candidate, and how far it tracks difficulty."""
	values, domain = readouts.of(fitted, grid)
	usable = domain & readouts.seen(data)
	keep = usable[1::2] & usable[0::2]

	above = {name: value[1::2] for name, value in values.items()}
	below = {name: value[0::2] for name, value in values.items()}
	rho = readouts.agreement(above, below, keep)

	rows = []
	for name in rho:
		delta = above[name][keep] - below[name][keep]
		middle = (above[name][keep] + below[name][keep]) / 2.0
		rows.append(
			{
				"split": split,
				"candidate": name,
				"spearman": rho[name],
				"slope": float(np.polyfit(middle, delta, 1)[0]),
				"correlation": float(np.corrcoef(middle, delta)[0, 1]),
				"items": int(keep.sum()),
			}
		)

	return pl.DataFrame(rows)

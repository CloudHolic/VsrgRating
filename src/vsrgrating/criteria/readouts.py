"""Every readout of one fit on a y grid, and how 2 fits' ordering agree."""

import numpy as np
from scipy.stats import spearmanr

from ..data import Split
from ..evaluation import Fitted
from ..irt.cnrm.model import ItemParams as CnrmParams
from ..readout import cnrm, zoi_beta

# Each ZOI-Beta readout divides by a, and where the likelihood is flat in a
# the quotient carries no information.
MIN_A = 0.2


def of(fitted: Fitted, grid: tuple[float, ...]) -> tuple[dict[str, np.ndarray], np.ndarray]:
	"""Each readout's value per item, named "readout(y)", and the items it can be read on."""
	item = fitted.item
	if isinstance(item, CnrmParams):
		assert fitted.model.transform is not None and fitted.raw is not None

		spec, raw = fitted.model.transform, fitted.raw
		values = {_name("b½", y): cnrm.difficulty(item, spec, raw, y) for y in grid}
		return values, np.ones(item.beta.shape, dtype=bool)

	values = {}
	for name, readout in (("b*", zoi_beta.b_star), ("b~", zoi_beta.b_tilde), ("b50", zoi_beta.b50)):
		values |= {_name(name, y): readout(item, y) for y in grid}

	return values, np.exp(np.asarray(item.log_a)) >= MIN_A


def seen(data: Split) -> np.ndarray:
	"""The item this fit has responses for; the rest are set by the prior alone."""
	return np.bincount(data.item, minlength=data.n_items) > 0


def agreement(
	left: dict[str, np.ndarray], right: dict[str, np.ndarray], keep: np.ndarray
) -> dict[str, float]:
	"""Rank correlation peer candidate on the items in `keep`."""
	return {name: float(spearmanr(left[name][keep], right[name][keep]).statistic) for name in left}


def _name(readout: str, y: float) -> str:
	return f"{readout}({y:.2f})"

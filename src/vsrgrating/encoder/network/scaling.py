"""Standardization of the continuous model inputs over a training set."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..inputs import Known, LayoutInput, token_known


@dataclass(frozen=True)
class Scaler:
	"""Column means and standard deviations of the continuous features over a training set."""

	mean: dict[str, np.ndarray]
	std: dict[str, np.ndarray]

	@classmethod
	def fit(cls, inputs: Sequence[LayoutInput]) -> Scaler:
		"""Statistics over `inputs`, every value counted once."""
		# Judgment statistics exist even when no training chart has a judgment.
		sums: dict[str, list[np.ndarray]] = {"judgment": [np.zeros(2) for _ in range(3)]}
		for x in inputs:
			for name, (values, known) in _columns(x).items():
				flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
				use = known.reshape(-1, known.shape[-1]) == Known.KNOWN

				acc = sums.setdefault(name, [np.zeros(flat.shape[1]) for _ in range(3)])
				acc[0] += use.sum(axis=0)
				acc[1] += np.where(use, flat, 0.0).sum(axis=0)
				acc[2] += np.where(use, flat**2, 0.0).sum(axis=0)

		mean, std = {}, {}
		for name, (n, total, squares) in sums.items():
			mean[name] = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
			var = np.divide(squares, n, out=np.zeros_like(total), where=n > 0) - mean[name] ** 2
			std[name] = np.sqrt(np.maximum(var, 0.0))

		return cls(mean, std)

	def apply(self, name: str, values: np.ndarray, known: np.ndarray) -> np.ndarray:
		"""Standardized values."""
		std = self.std[name]
		scaled = np.divide(values - self.mean[name], std, out=np.zeros(values.shape), where=std > 0)
		return np.where(known == Known.KNOWN, scaled, 0.0).astype(np.float32)


@dataclass(frozen=True)
class VectorScaler:
	"""Column means and standard deviations of plain feature vectors."""

	mean: np.ndarray
	std: np.ndarray

	@classmethod
	def fit(cls, rows: np.ndarray) -> VectorScaler:
		"""Statistics over the rows of `rows`."""
		return cls(rows.mean(axis=0), rows.std(axis=0))

	def apply(self, rows: np.ndarray) -> np.ndarray:
		"""Standardized rows."""
		return np.divide(rows - self.mean, self.std, out=np.zeros(rows.shape), where=self.std > 0)


def _columns(x: LayoutInput) -> dict[str, tuple[np.ndarray, np.ndarray]]:
	"""Each continuous block of `x` with the Known array of its values."""
	columns = {
		"token": (x.token, token_known(x)),
		"lane": (x.lane, x.lane_known),
		"finger": (x.finger, x.finger_known),
		"pair": (x.pair, x.pair_known),
		"meta": (x.meta[None], np.zeros((1, x.meta.size), np.int8)),
	}
	if x.judgment is not None:
		columns["judgment"] = (x.judgment, np.zeros(x.judgment.shape, np.int8))

	return columns

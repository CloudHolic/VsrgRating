"""Standardization of the continuous model inputs over a training set."""

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..inputs import (
	FINGER_STREAMS,
	JUDGMENT_STREAMS,
	OCCUPANCY_STREAMS,
	PAIR_STREAMS,
	VISUAL_STREAMS,
	ChartLoad,
	Known,
	LayoutInput,
	LayoutLoad,
)

# Each load family and its stream names, in the order of the load arrays' stream axis.
LOAD_FAMILIES = {
	"finger": FINGER_STREAMS,
	"pair": PAIR_STREAMS,
	"judgment": JUDGMENT_STREAMS,
	"visual": VISUAL_STREAMS,
}


@dataclass(frozen=True)
class Scaler:
	"""Column means and standard deviations of the continuous features over a training set."""

	mean: dict[str, np.ndarray]
	std: dict[str, np.ndarray]

	@classmethod
	def fit(cls, inputs: Iterable[LayoutInput]) -> Scaler:
		"""Statistics over `inputs`, every known value counted once."""
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

	def to_dict(self) -> dict[str, dict[str, Tensor]]:
		"""The statistics as tensor, loadable with `torch.load(..., weights_only=True)`."""
		return {
			part: {name: torch.from_numpy(values) for name, values in stats.items()}
			for part, stats in (("mean", self.mean), ("std", self.std))
		}

	@classmethod
	def from_dict(cls, record: dict[str, dict[str, Tensor]]) -> Scaler:
		"""The scaler `to_dict` wrote."""
		mean, std = (
			{name: value.numpy().astype(np.float64) for name, value in record[part].items()}
			for part in ("mean", "std")
		)
		return cls(mean, std)

	def digest(self) -> str:
		"""A hash of the statistics, equal for equal scalers."""
		h = hashlib.blake2b(digest_size=8)
		for name in sorted(self.mean):
			h.update(name.encode())
			h.update(self.mean[name].astype(np.float64).tobytes())
			h.update(self.std[name].astype(np.float64).tobytes())

		return h.hexdigest()


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


@dataclass(frozen=True)
class LoadScaler:
	"""Mean and standard deviation of every load family's (stream, tau) over a training set."""

	mean: dict[str, np.ndarray]
	std: dict[str, np.ndarray]

	@classmethod
	def fit(cls, loads: Iterable[tuple[Sequence[LayoutLoad], ChartLoad]]) -> LoadScaler:
		"""Statistics over each chart's loads under all its layouts."""
		sums: dict[str, list[np.ndarray]] = {}

		for layouts_, chart in loads:
			blocks = [("judgment", chart.judgment), ("visual", chart.visual)]

			for load in layouts_:
				blocks += [("finger", load.finger), ("pair", load.pair)]

			for family, values in blocks:
				flat = _transform(family, values).reshape(-1, *values.shape[-2:])
				acc = sums.setdefault(family, [np.zeros(values.shape[-2:]) for _ in range(3)])
				acc[0] += flat.shape[0]
				acc[1] += flat.sum(axis=0)
				acc[2] += (flat**2).sum(axis=0)

		mean, std = {}, {}
		for family, (n, total, squares) in sums.items():
			mean[family] = total / np.maximum(n, 1)
			std[family] = np.sqrt(np.maximum(squares / np.maximum(n, 1) - mean[family] ** 2, 0.0))

		return cls(mean, std)

	@property
	def valid(self) -> dict[str, np.ndarray]:
		"""(streams, taus) bool per family."""
		return {family: std > 0 for family, std in self.std.items()}

	def apply(self, family: str, values: np.ndarray) -> np.ndarray:
		"""Standardized float32 values of one family."""
		std = self.std[family]
		scaled = np.divide(
			_transform(family, values) - self.mean[family],
			std,
			out=np.zeros(values.shape),
			where=std > 0,
		)

		return scaled.astype(np.float32)

	def to_dict(self) -> dict[str, dict[str, Tensor]]:
		"""The statistics as tensors."""
		return {
			part: {name: torch.from_numpy(values) for name, values in stats.items()}
			for part, stats in (("mean", self.mean), ("std", self.std))
		}

	@classmethod
	def from_dict(cls, record: dict[str, dict[str, Tensor]]) -> LoadScaler:
		"""The scaler `to_dict` wrote."""
		mean, std = (
			{name: value.numpy().astype(np.float64) for name, value in record[part].items()}
			for part in ("mean", "std")
		)
		return cls(mean, std)


def _columns(x: LayoutInput) -> dict[str, tuple[np.ndarray, np.ndarray]]:
	"""Each continuous block of `x` with the Known array of its values."""
	columns = {
		"token": (x.token, x.token_known),
		"lane": (x.lane, x.lane_known),
		"finger": (x.finger, x.finger_known),
		"pair": (x.pair, x.pair_known),
		"meta": (x.meta[None], x.meta_known[None]),
	}
	if x.judgment is not None:
		columns["judgment"] = (x.judgment, np.zeros(x.judgment.shape, np.int8))

	return columns


def _transform(family: str, values: np.ndarray) -> np.ndarray:
	"""log1p on every stream but the occupancies, along the stream axis (second to last)."""
	occupancy = np.array([name in OCCUPANCY_STREAMS for name in LOAD_FAMILIES[family]])
	return np.where(occupancy[:, None], values, np.log1p(values))

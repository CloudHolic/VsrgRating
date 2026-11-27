"""Standardization of the load targets over a training set."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .load import (
	FINGER_STREAMS,
	JUDGMENT_STREAMS,
	OCCUPANCY_STREAMS,
	PAIR_STREAMS,
	VISUAL_STREAMS,
	ChartLoad,
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


def _transform(family: str, values: np.ndarray) -> np.ndarray:
	"""log1p on every stream but the occupancies, along the stream axis (second to last)."""
	occupancy = np.array([name in OCCUPANCY_STREAMS for name in LOAD_FAMILIES[family]])
	return np.where(occupancy[:, None], values, np.log1p(values))

"""Heads from a chart vector, or from summary statistics, to CNRM item parameters."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .batching import LayoutBatch
from .model import ChartEncoder


@dataclass(frozen=True)
class ItemInit:
	"""The head's starting output."""

	alpha: tuple[float, float, float]
	beta: float
	log_sigma: float


@dataclass(frozen=True)
class ItemOutput:
	"""CNRM item parameters of C charts: alpha (C, 3), beta (C,), sigma(C,)."""

	alpha: Tensor
	beta: Tensor
	sigma: Tensor


class ItemHead(nn.Module):
	"""Linear map to (alpha_1..3, beta, log sigma), starting at `init` for every input."""

	def __init__(self, in_dim: int, init: ItemInit) -> None:
		super().__init__()

		self.linear = nn.Linear(in_dim, 5)
		nn.init.zeros_(self.linear.weight)

		with torch.no_grad():
			self.linear.bias.copy_(torch.tensor([*init.alpha, init.beta, init.log_sigma]))

	def forward(self, z: Tensor) -> ItemOutput:
		"""Item parameters of the rows of `z`."""
		out = self.linear(z)
		# The clamp only keeps exp finite.
		return ItemOutput(out[:, :3], out[:, 3], torch.exp(out[:, 4].clamp(-20.0, 10.0)))


class ChartItemModel(nn.Module):
	"""The encoder and its head: a batch of charts to their item parameters."""

	def __init__(self, encoder: ChartEncoder, head: ItemHead) -> None:
		super().__init__()
		self.encoder = encoder
		self.head = head

	def forward(self, batch: LayoutBatch) -> ItemOutput:
		"""Item parameters of the batch's charts."""
		return self.head(self.encoder(batch))


class StatsModel(nn.Module):
	"""The baseline: the head straight on standardized summary statistics."""

	def __init__(self, n_stats: int, init: ItemInit) -> None:
		super().__init__()
		self.head = ItemHead(n_stats, init)

	def forward(self, stats: Tensor) -> ItemOutput:
		"""Item parameters of the rows of `stats`."""
		return self.head(stats)

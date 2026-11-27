"""Pretraining heads and losses."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..inputs import (
	FINGER_STREAMS,
	JUDGMENT_STREAMS,
	LOAD_TAUS,
	PAIR_STREAMS,
	VISUAL_STREAMS,
	Finger,
	Hand,
)

MLM_STATES = 6
_FAMILIES = ("finger", "pair", "judgment", "visual")


@dataclass(frozen=True)
class LoadOutput:
	"""Predicted standardized load."""

	finger: Tensor  # (B, T, K, 3, 5)
	pair: Tensor  # (B, T, P, 5, 5)
	judgment: Tensor  # (B, T, 4, 5)
	visual: Tensor  # (B, T, 7, 5)


@dataclass(frozen=True)
class LoadTargets:
	"""Standardized load targets shaped as `LoadOutput`."""

	finger: Tensor
	pair: Tensor
	judgment: Tensor
	visual: Tensor
	fingers: Tensor
	finger_mask: Tensor
	pair_mask: Tensor

	def to(self, device: torch.device | str) -> LoadTargets:
		"""The same targets on `device`."""
		return LoadTargets(**{name: value.to(device) for name, value in vars(self).items()})


class MLMHead(nn.Module):
	"""Each lane's state from its token and lane representations."""

	def __init__(self, d_model: int, d_lane: int) -> None:
		super().__init__()

		self.token = nn.Linear(d_model, d_lane)
		self.lane = nn.Linear(d_lane, d_lane)
		self.out = nn.Linear(d_lane, MLM_STATES)

	def forward(self, token_repr: Tensor, lane_repr: Tensor) -> Tensor:
		"""(B, T, L, G) logits."""
		return self.out(functional.gelu(self.token(token_repr)[:, :, None] + self.lane(lane_repr)))


class LoadHead(nn.Module):
	"""The load to come from a token representation."""

	def __init__(self, d_model: int, width: int) -> None:
		super().__init__()
		d, taus = d_model, len(LOAD_TAUS)

		self.finger_token = nn.Linear(d, d)
		self.finger_hand = nn.Embedding(len(Hand), d)
		self.finger_kind = nn.Embedding(len(Finger), d)
		self.finger_out = nn.Linear(d, len(FINGER_STREAMS) * taus)

		self.pair_token = nn.Linear(d, d)
		self.pair_hand = nn.Embedding(len(Hand), d)
		self.pair_first = nn.Embedding(len(Finger), d)
		self.pair_second = nn.Embedding(len(Finger), d)
		self.pair_out = nn.Linear(d, len(PAIR_STREAMS) * taus)

		self.judgment = nn.Sequential(
			nn.Linear(d, d),
			nn.GELU(),
			nn.Linear(d, len(JUDGMENT_STREAMS) * taus),
		)
		self.visual = nn.Sequential(
			nn.Linear(d, d),
			nn.GELU(),
			nn.Linear(d, len(VISUAL_STREAMS) * taus),
		)
		self.pool = nn.Linear(width, d)

	def forward(self, token_repr: Tensor, fingers: Tensor, pair_index: Tensor) -> LoadOutput:
		"""`token_repr` (B, T, d), `fingers`(B, K, 2), `pair_index` (B, P, 3)."""
		b, t, _ = token_repr.shape
		taus = len(LOAD_TAUS)

		finger_entity = self.finger_hand(fingers[..., 0]) + self.finger_kind(fingers[..., 1])
		finger = self.finger_out(
			functional.gelu(self.finger_token(token_repr)[:, :, None] + finger_entity[:, None])
		)
		pair_entity = (
			self.pair_hand(pair_index[..., 0])
			+ self.pair_first(pair_index[..., 1])
			+ self.pair_second(pair_index[..., 2])
		)
		pair = self.pair_out(
			functional.gelu(self.pair_token(token_repr)[:, :, None] + pair_entity[:, None])
		)

		return LoadOutput(
			finger=finger.reshape(b, t, -1, len(FINGER_STREAMS), taus),
			pair=pair.reshape(b, t, -1, len(PAIR_STREAMS), taus),
			judgment=self.judgment(token_repr).reshape(b, t, len(JUDGMENT_STREAMS), taus),
			visual=self.visual(token_repr).reshape(b, t, len(VISUAL_STREAMS), taus),
		)

	def pooled(self, vector: Tensor, fingers: Tensor, pair_index: Tensor) -> LoadOutput:
		"""The same head on pooled chart vectors (B, width), without the T axis."""
		out = self.forward(self.pool(vector)[:, None], fingers, pair_index)
		return LoadOutput(*(getattr(out, name)[:, 0] for name in _FAMILIES))


def mlm_loss(logits: Tensor, labels: Tensor, chart_of_layout: Tensor, n_charts: int) -> Tensor:
	"""Cross entropy of the masked cells."""
	valid = labels >= 0
	cells = functional.cross_entropy(
		logits.reshape(-1, MLM_STATES), labels.clamp_min(0).reshape(-1).long(), reduction="none"
	).reshape(labels.shape)
	lanes = valid.sum(dim=2)
	per_token = (cells * valid).sum(dim=2) / lanes.clamp_min(1)
	masked = lanes > 0

	per_layout = per_token.sum(dim=1)
	tokens = masked.sum(dim=1).to(per_layout.dtype)
	charts = _chart_mean(per_layout, chart_of_layout, n_charts)
	counts = _chart_mean(tokens, chart_of_layout, n_charts)

	return charts.sum() / counts.sum().clamp_min(1)


def load_token_loss(
	output: LoadOutput,
	targets: LoadTargets,
	valid: Mapping[str, Tensor],
	weights: Tensor,
	n_tokens: Tensor,
	chart_of_layout: Tensor,
	n_charts: int,
) -> Tensor:
	"""Huber loss of the load to come at every supervised token."""
	per_token = _family_mean(output, targets, valid, token_axis=True)
	per_layout = (weights * per_token).sum(dim=1) / n_tokens.to(per_token.dtype)

	return _chart_mean(per_layout, chart_of_layout, n_charts).mean()


def load_pool_loss(
	output: LoadOutput,
	targets: LoadTargets,
	prefix_end0: Tensor,
	valid: Mapping[str, Tensor],
	chart_of_layout: Tensor,
	n_charts: int,
) -> Tensor:
	"""Huber loss of the pooled prefix's load to come, at the end of the first segment's prefix."""
	rows = torch.arange(prefix_end0.shape[0], device=prefix_end0.device)
	at_end = LoadTargets(
		finger=targets.finger[rows, prefix_end0],
		pair=targets.pair[rows, prefix_end0],
		judgment=targets.judgment[rows, prefix_end0],
		visual=targets.visual[rows, prefix_end0],
		fingers=targets.fingers,
		finger_mask=targets.finger_mask,
		pair_mask=targets.pair_mask,
	)
	per_layout = _family_mean(output, at_end, valid, token_axis=False)

	return _chart_mean(per_layout, chart_of_layout, n_charts).mean()


def _family_mean(
	output: LoadOutput, targets: LoadTargets, valid: Mapping[str, Tensor], token_axis: bool
) -> Tensor:
	"""Within each family the mean over real entities, streams with spread and taus."""
	values, present = [], []
	extra = (slice(None), None) if token_axis else (slice(None),)
	for family in _FAMILIES:
		pred, target = getattr(output, family), getattr(targets, family)
		loss = functional.huber_loss(pred, target, reduction="none", delta=1.0)
		mask = valid[family].to(loss.dtype)
		if family in ("finger", "pair"):
			entity = getattr(targets, f"{family}_mask").to(loss.dtype)
			summed = (loss * entity[(*extra, slice(None), None, None)] * mask).sum(dim=(-3, -2, -1))
			count = entity.sum(dim=1)[extra] * mask.sum()
		else:
			summed = (loss * mask).sum(dim=(-2, -1))
			count = torch.full_like(summed, float(mask.sum()))
		count = count.expand_as(summed)
		values.append(torch.where(count > 0, summed / count.clamp_min(1), torch.zeros_like(summed)))
		present.append((count > 0).to(loss.dtype))
	return torch.stack(values).sum(dim=0) / torch.stack(present).sum(dim=0).clamp_min(1)


def _chart_mean(values: Tensor, chart_of_layout: Tensor, n_charts: int) -> Tensor:
	"""Each chart's mean over its layouts."""
	sums = values.new_zeros(n_charts).index_add(0, chart_of_layout, values)
	counts = torch.bincount(chart_of_layout, minlength=n_charts).to(values.dtype)

	return sums / counts

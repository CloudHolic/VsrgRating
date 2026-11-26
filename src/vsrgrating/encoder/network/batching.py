"""Layout inputs as padded torch batches."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..inputs import Known, LayoutInput, token_known
from .scaling import Scaler


@dataclass
class LayoutBatch:
	"""B layout passes padded to T tokens, L lanes, P pairs and J judgment windows."""

	state: Tensor
	token: Tensor
	token_dt_known: Tensor
	lane: Tensor
	lane_known: Tensor
	finger: Tensor
	finger_known: Tensor
	pair: Tensor
	pair_known: Tensor
	pair_index: Tensor
	hands: Tensor
	lane_hand: Tensor
	lane_finger: Tensor
	lane_position: Tensor
	meta: Tensor
	judgment: Tensor
	judgment_mask: Tensor
	judgment_unknown: Tensor
	token_mask: Tensor
	lane_mask: Tensor
	pair_mask: Tensor
	chart_of_layout: Tensor
	n_charts: int

	def to(self, device: torch.device | str) -> LayoutBatch:
		"""The same batch on `device`."""
		moved = {
			name: value.to(device) if isinstance(value, Tensor) else value
			for name, value in vars(self).items()
		}

		return LayoutBatch(**moved)


def collate(
	charts: Sequence[Sequence[LayoutInput]],
	scaler: Scaler,
	judgment_drop: float,
	generator: torch.Generator | None,
) -> LayoutBatch:
	"""One batch from each chart's layout inputs."""
	drop = torch.rand(len(charts), generator=generator) < judgment_drop
	inputs = [x for layouts_ in charts for x in layouts_]
	chart_of_layout = [c for c, layouts_ in enumerate(charts) for _ in layouts_]
	unknown = [bool(drop[c]) or inputs[i].judgment is None for i, c in enumerate(chart_of_layout)]

	b = len(inputs)
	t = max(x.state.shape[0] for x in inputs)
	lanes = max(x.state.shape[1] for x in inputs)
	p = max(x.pair_index.shape[0] for x in inputs)
	j = max((x.judgment.shape[0] for x in inputs if x.judgment is not None), default=0)

	out = {
		"state": np.zeros((b, t, lanes), np.int64),
		"token": np.zeros((b, t, inputs[0].token.shape[1]), np.float32),
		"token_dt_known": np.full((b, t), Known.NONE, np.int64),
		"lane": np.zeros((b, t, lanes, inputs[0].lane.shape[2]), np.float32),
		"lane_known": np.full((b, t, lanes, inputs[0].lane.shape[2]), Known.NONE, np.int64),
		"finger": np.zeros((b, t, lanes, inputs[0].finger.shape[2]), np.float32),
		"finger_known": np.full((b, t, lanes, inputs[0].finger.shape[2]), Known.NONE, np.int64),
		"pair": np.zeros((b, t, p, inputs[0].pair.shape[2]), np.float32),
		"pair_known": np.full((b, t, p, inputs[0].pair.shape[2]), Known.NONE, np.int64),
		"pair_index": np.zeros((b, p, 3), np.int64),
		"hands": np.zeros((b, t, 2), np.int64),
		"lane_hand": np.zeros((b, lanes), np.int64),
		"lane_finger": np.zeros((b, lanes), np.int64),
		"lane_position": np.zeros((b, lanes, 3), np.float32),
		"meta": np.zeros((b, inputs[0].meta.shape[0]), np.float32),
		"judgment": np.zeros((b, j, 2), np.float32),
		"judgment_mask": np.zeros((b, j), bool),
		"token_mask": np.zeros((b, t), bool),
		"lane_mask": np.zeros((b, lanes), bool),
		"pair_mask": np.zeros((b, p), bool),
	}

	for i, x in enumerate(inputs):
		n, k = x.state.shape
		q = x.pair_index.shape[0]

		out["state"][i, :n, :k] = x.state
		out["token"][i, :n] = scaler.apply("token", x.token, token_known(x))
		out["token_dt_known"][i, :n] = x.token_dt_known

		for name in ("lane", "finger", "pair"):
			known = getattr(x, f"{name}_known")
			width = k if name != "pair" else q

			out[name][i, :n, :width] = scaler.apply(name, getattr(x, name), known)
			out[f"{name}_known"][i, :n, :width] = known

		out["pair_index"][i, :q] = x.pair_index
		out["hands"][i, :n] = x.hands
		out["lane_hand"][i, :k] = x.lane_hand
		out["lane_finger"][i, :k] = x.lane_finger
		out["lane_position"][i, :k] = x.lane_position
		out["meta"][i] = scaler.apply("meta", x.meta, np.zeros(x.meta.shape, np.int8))

		if x.judgment is not None and not unknown[i]:
			out["judgment"][i, : len(x.judgment)] = scaler.apply(
				"judgment", x.judgment, np.zeros(x.judgment.shape, np.int8)
			)
			out["judgment_mask"][i, : len(x.judgment)] = True

		out["token_mask"][i, :n] = True
		out["lane_mask"][i, :k] = True
		out["pair_mask"][i, :q] = True

	tensors = {name: torch.from_numpy(value) for name, value in out.items()}

	return LayoutBatch(
		**tensors,
		judgment_unknown=torch.tensor(unknown),
		chart_of_layout=torch.tensor(chart_of_layout),
		n_charts=len(charts),
	)


def encoder_tokens(n_tokens: int, segment: int) -> int:
	"""Tokens a layout of `n_tokens` note tokens sends through the encoder."""
	return n_tokens + math.ceil(n_tokens / (segment - 1))


def token_batches(
	n_tokens: np.ndarray,
	n_layouts: np.ndarray,
	budget: int,
	segment: int,
	rng: np.random.Generator,
) -> list[np.ndarray]:
	"""Chart indices grouped so that a batch's layouts, padded to its longest, stay within
	`budget` encoder tokens."""
	jitter = rng.uniform(0.9, 1.1, n_tokens.size)
	order = np.argsort(n_tokens * jitter, kind="stable")

	batches: list[list[int]] = []
	current: list[int] = []
	layouts_, longest = 0, 0
	for chart in order.tolist():
		cost = encoder_tokens(int(n_tokens[chart]), segment)
		grown = (layouts_ + int(n_layouts[chart])) * max(longest, cost)
		if current and grown > budget:
			batches.append(current)
			current, layouts_, longest = [], 0, 0
		current.append(chart)
		layouts_ += int(n_layouts[chart])
		longest = max(longest, cost)
	if current:
		batches.append(current)

	return [np.array(batches[i]) for i in rng.permutation(len(batches))]

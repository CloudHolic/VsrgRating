"""The encoder."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..inputs import (
	FINGER_FEATURES,
	LANE_FEATURES,
	META_FEATURES,
	PAIR_FEATURES,
	TOKEN_FEATURES,
	Finger,
	Flag,
	Hand,
	Known,
	LaneState,
)
from .batching import LayoutBatch
from .blocks import Block, Valued, masked_mean, mlp, rotary


@dataclass(frozen=True)
class EncoderConfig:
	"""Widths and depths of the encoder."""

	d_lane: int = 128
	lane_layers: int = 2
	lane_heads: int = 4
	d_model: int = 256
	layers: int = 6
	heads: int = 8
	ffn: int = 1024
	dropout: float = 0.1
	segment: int = 4096
	queries: int = 4


@dataclass(frozen=True)
class EncodedLayout:
	"""One layout pass encoded."""

	token_repr: Tensor
	meta_repr: Tensor
	lane_repr: Tensor
	mask: Tensor


class ChartEncoder(nn.Module):
	"""Lanes to tokens by self-attention over the lanes."""

	def __init__(self, config: EncoderConfig) -> None:
		super().__init__()

		self.config = config
		d, dl = config.d_model, config.d_lane

		self.state = nn.Embedding(len(LaneState), dl)
		self.position = mlp(3, dl)
		self.finger = nn.Embedding(len(Finger), dl)
		self.hand = nn.Embedding(len(Hand), dl)
		self.lane_values = Valued(len(LANE_FEATURES) + len(FINGER_FEATURES), dl)

		lane_layer = nn.TransformerEncoderLayer(
			dl, config.lane_heads, 4 * dl, config.dropout, batch_first=True, norm_first=True
		)
		self.lanes = nn.TransformerEncoder(
			lane_layer, config.lane_layers, enable_nested_tensor=False
		)
		self.lane_out = nn.Linear(dl, d)

		self.token_values = Valued(len(TOKEN_FEATURES), d)
		self.pair_values = Valued(len(PAIR_FEATURES), d)

		self.pair_first = nn.Embedding(len(Finger), d)
		self.pair_second = nn.Embedding(len(Finger), d)
		self.mirror = nn.Embedding(len(Flag), d)
		self.same_fingers = nn.Embedding(len(Flag), d)

		self.meta = mlp(len(META_FEATURES), d)
		self.judgment = mlp(2, d)
		self.judgment_unknown = nn.Parameter(torch.randn(d) * 0.02)

		self.blocks = nn.ModuleList(
			Block(d, config.heads, config.ffn, config.dropout) for _ in range(config.layers)
		)
		self.norm = nn.LayerNorm(d)
		self.queries = nn.Parameter(torch.randn(config.queries, d) * 0.02)

		self.pool_key = nn.Linear(d, d)
		self.pool_value = nn.Linear(d, d)

	@property
	def width(self) -> int:
		"""Length of the chart vector."""
		return self.config.queries * self.config.d_model

	def encode_layout(self, batch: LayoutBatch) -> EncodedLayout:
		"""Every layout pass of the batch, before pooling."""
		b, t, lanes = batch.state.shape
		dl = self.config.d_lane

		lane = (
			self.state(batch.state)
			+ (
				self.position(batch.lane_position)
				+ self.finger(batch.lane_finger)
				+ self.hand(batch.lane_hand)
			)[:, None]
			+ self.lane_values(
				torch.cat([batch.lane, batch.finger], dim=-1),
				torch.cat([batch.lane_known, batch.finger_known], dim=-1),
			)
		)

		padded = (~batch.lane_mask)[:, None].expand(b, t, lanes).reshape(b * t, lanes)
		lane_repr = self.lanes(lane.reshape(b * t, lanes, dl), src_key_padding_mask=padded)
		lane_repr = lane_repr.reshape(b, t, lanes, dl)
		lane_mean = masked_mean(lane_repr, batch.lane_mask[:, None, :, None], dim=2)

		token_known = torch.full_like(batch.token, Known.KNOWN, dtype=torch.long)
		token_known[..., 0] = batch.token_dt_known

		pair = (
			self.pair_values(batch.pair, batch.pair_known)
			+ (
				self.pair_first(batch.pair_index[..., 1])
				+ self.pair_second(batch.pair_index[..., 2])
			)[:, None]
		)
		tokens = (
			self.lane_out(lane_mean)
			+ self.token_values(batch.token, token_known)
			+ masked_mean(pair, batch.pair_mask[:, None, :, None], dim=2)
			+ self.mirror(batch.hands[..., 0])
			+ self.same_fingers(batch.hands[..., 1])
		)

		judgment = masked_mean(self.judgment(batch.judgment), batch.judgment_mask[..., None], dim=1)
		unknown = batch.judgment_unknown[:, None]
		meta = self.meta(batch.meta) + torch.where(unknown, self.judgment_unknown, judgment)

		token_repr, meta_repr = self._sequence(tokens, meta, batch.token_mask)

		return EncodedLayout(token_repr, meta_repr, lane_repr, batch.token_mask)

	def pool(self, encoded: EncodedLayout) -> Tensor:
		"""(B, queries * d_model): attention pooling over the note tokens and the meta vector."""
		keys = torch.cat([encoded.token_repr, encoded.meta_repr[:, None]], dim=1)
		mask = torch.cat([encoded.mask, encoded.mask.new_ones(encoded.mask.shape[0], 1)], dim=1)
		scores = self.queries @ self.pool_key(keys).transpose(1, 2) / math.sqrt(keys.shape[-1])
		weights = scores.masked_fill(~mask[:, None], float("-inf")).softmax(dim=-1)

		return (weights @ self.pool_value(keys)).flatten(1)

	def forward(self, batch: LayoutBatch) -> Tensor:
		"""(C, width): each chart's vector, the mean over its layout passes."""
		vectors = self.pool(self.encode_layout(batch))
		sums = vectors.new_zeros(batch.n_charts, vectors.shape[1])
		sums.index_add_(0, batch.chart_of_layout, vectors)
		counts = torch.bincount(batch.chart_of_layout, minlength=batch.n_charts)

		return sums / counts[:, None]

	def _sequence(self, tokens: Tensor, meta: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
		"""The Transformer over segments of note tokens, each led by the meta token."""
		b, t, d = tokens.shape
		chunk = min(self.config.segment - 1, t)
		segments = math.ceil(t / chunk)
		pad = segments * chunk - t

		notes = functional.pad(tokens, (0, 0, 0, pad)).reshape(b, segments, chunk, d)
		note_mask = functional.pad(mask, (0, pad)).reshape(b, segments, chunk)

		x = torch.cat([meta[:, None, None].expand(b, segments, 1, d), notes], dim=2)
		x = x.reshape(b * segments, chunk + 1, d)

		keep = torch.cat([note_mask.new_ones(b, segments, 1), note_mask], dim=2)
		keep = keep.reshape(b * segments, 1, 1, chunk + 1)

		cos, sin = rotary(chunk + 1, d // self.config.heads, x.device)

		for block in self.blocks:
			x = block(x, keep, cos, sin)
		x = self.norm(x).reshape(b, segments, chunk + 1, d)

		token_repr = x[:, :, 1:].reshape(b, segments * chunk, d)[:, :t]
		used = note_mask.any(dim=2)
		meta_repr = masked_mean(x[:, :, 0], used[..., None], dim=1)

		return token_repr, meta_repr

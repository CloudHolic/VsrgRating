"""Building block of the encoder."""

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..inputs import Known


class Valued(nn.Module):
	"""Continuous columns with their Known State."""

	def __init__(self, columns: int, width: int) -> None:
		super().__init__()
		self.net = mlp(3 * columns, width)

	def forward(self, values: Tensor, known: Tensor) -> Tensor:
		"""(..., width) from values and Known states of the same shape."""
		flags = [(known == Known.NONE).to(values.dtype), (known == Known.MASKED).to(values.dtype)]
		return self.net(torch.cat([values, *flags], dim=-1))


class Block(nn.Module):
	"""Pre-LN Transformer block with rotary positions."""

	def __init__(self, d: int, heads: int, ffn: int, dropout: float) -> None:
		super().__init__()

		self.heads = heads
		self.dropout = dropout
		self.norm1 = nn.LayerNorm(d)
		self.qkv = nn.Linear(d, 3 * d)
		self.out = nn.Linear(d, d)
		self.norm2 = nn.LayerNorm(d)
		self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
		self.drop = nn.Dropout(dropout)

	def forward(self, x: Tensor, keep: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
		"""`x` (N, T, d) attending where `keep` (N, 1, 1, T) is set."""
		n, t, d = x.shape

		q, k, v = self.qkv(self.norm1(x)).reshape(n, t, 3, self.heads, d // self.heads).unbind(2)
		q, k, v = (part.transpose(1, 2) for part in (q, k, v))
		q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)

		attn = functional.scaled_dot_product_attention(
			q, k, v, attn_mask=keep, dropout_p=self.dropout if self.training else 0.0
		)
		x = x + self.drop(self.out(attn.transpose(1, 2).reshape(n, t, d)))

		return x + self.drop(self.ffn(self.norm2(x)))


def mlp(inputs: int, width: int) -> nn.Sequential:
	"""2 linear layers with a GELU between."""
	return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, width))


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
	"""Mean of `x` over `dim` where `mask` is set; 0 where nothing is."""
	weight = mask.to(x.dtype)
	return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def rotary(length: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
	"""Cosines and sines of the rotary angles for positions 0..length-1."""
	half = width // 2
	inverse = 10000.0 ** (-torch.arange(half, device=device, dtype=torch.float32) / half)
	angle = torch.arange(length, device=device, dtype=torch.float32)[:, None] * inverse

	return angle.cos(), angle.sin()


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
	half = x.shape[-1] // 2
	first, second = x[..., :half], x[..., half:]

	return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)

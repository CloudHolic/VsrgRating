"""The encoder's training objective: CNRM cell log likelihood under a fixed normal person posterior
q = N(mu, Sigma)."""

import math

import numpy as np
import torch
from torch import Tensor

BOUNDARY_NODES = 48

_SPAN = 10.0
_BEND = 6.0

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(BOUNDARY_NODES)


def expected_log_k(
	x: Tensor,
	g: Tensor,
	log_dg: Tensor,
	alpha: Tensor,
	beta: Tensor,
	sigma: Tensor,
	mu: Tensor,
	cov: Tensor,
) -> Tensor:
	"""E_q[log k] per cell: closed form inside (0, 1), piecewise Gauss-Legendre over eta at the
	endpoints."""
	interior, one, m, v = _prepare(x, g, log_dg, alpha, beta, sigma, mu, cov)
	g, log_dg = _interior_only(interior, g, log_dg)
	inside = -_HALF_LOG_2PI - torch.log(sigma) + log_dg - ((g - m) ** 2 + v) / (2.0 * sigma**2)

	edge = ~interior
	means = _endpoint_mean(one[edge], m[edge], v[edge], sigma[edge])

	return torch.where(interior, inside, torch.zeros_like(m).masked_scatter(edge, means))


def log_expected_k(
	x: Tensor,
	g: Tensor,
	log_dg: Tensor,
	alpha: Tensor,
	beta: Tensor,
	sigma: Tensor,
	mu: Tensor,
	cov: Tensor,
) -> Tensor:
	"""log E_q[k] per cell, in closed form."""
	interior, one, m, v = _prepare(x, g, log_dg, alpha, beta, sigma, mu, cov)
	g, log_dg = _interior_only(interior, g, log_dg)
	scale = torch.sqrt(sigma**2 + v)
	inside = -_HALF_LOG_2PI - torch.log(scale) + log_dg - (g - m) ** 2 / (2.0 * scale**2)
	edge = torch.special.log_ndtr(_endpoint_z(one, m, scale))

	return torch.where(interior, inside, edge)


def _prepare(
	x: Tensor,
	g: Tensor,
	log_dg: Tensor,
	alpha: Tensor,
	beta: Tensor,
	sigma: Tensor,
	mu: Tensor,
	cov: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
	_check(x, g, log_dg, alpha, beta, sigma, mu, cov)

	m = (alpha * mu).sum(dim=-1) + beta
	v = torch.einsum("ni,nij,nj->n", alpha, cov, alpha).clamp_min(0.0)

	return (x != 0.0) & (x != 1.0), x == 1.0, m, v


def _check(
	x: Tensor,
	g: Tensor,
	log_dg: Tensor,
	alpha: Tensor,
	beta: Tensor,
	sigma: Tensor,
	mu: Tensor,
	cov: Tensor,
) -> None:
	given = {
		"x": x,
		"g": g,
		"log_dg": log_dg,
		"alpha": alpha,
		"beta": beta,
		"sigma": sigma,
		"mu": mu,
		"cov": cov,
	}

	if wrong := [name for name, t in given.items() if t.dtype != torch.float64]:
		raise ValueError(f"{', '.join(wrong)} must be float64")
	if alpha.ndim != 2:
		raise ValueError(f"alpha must be (cells, D), got {tuple(alpha.shape)}")

	n, dims = alpha.shape
	expected: dict[str, tuple[int, ...]] = dict.fromkeys(
		("x", "g", "log_dg", "beta", "sigma"), (n,)
	)
	expected |= {"mu": (n, dims), "cov": (n, dims, dims)}

	for name, shape in expected.items():
		if tuple(given[name].shape) != shape:
			raise ValueError(f"{name} must be {shape}, got {tuple(given[name].shape)}")


def _interior_only(interior: Tensor, g: Tensor, log_dg: Tensor) -> tuple[Tensor, Tensor]:
	# The endpoint cells' G values are never used, but aNaN there would still posion
	# the gradient of the discarded branch.
	zero = torch.zeros_like(g)
	return torch.where(interior, g, zero), torch.where(interior, log_dg, zero)


def _endpoint_mean(one: Tensor, m: Tensor, v: Tensor, sigma: Tensor) -> Tensor:
	"""E[log Phi(z)] over eta ~ N(m, v) at endpoint cells; log Phi(z(m)) where v = 0."""
	positive = v > 0.0
	sd = torch.sqrt(torch.where(positive, v, torch.ones_like(v)))
	threshold = one.to(m.dtype)
	lo, hi = m - _SPAN * sd, m + _SPAN * sd

	below = torch.clamp(threshold - _BEND * sigma, lo, hi)
	above = torch.clamp(threshold + _BEND * sigma, lo, hi)
	cuts = torch.stack([lo, below, above, hi], dim=1)

	nodes = torch.as_tensor(_NODES, dtype=m.dtype, device=m.device)
	weights = torch.as_tensor(_WEIGHTS, dtype=m.dtype, device=m.device)

	half = (cuts[:, 1:] - cuts[:, :-1])[..., None] / 2.0
	eta = (cuts[:, 1:] + cuts[:, :-1])[..., None] / 2.0 + half * nodes
	u = (eta - m[:, None, None]) / sd[:, None, None]
	density = torch.exp(-0.5 * u**2 - _HALF_LOG_2PI) / sd[:, None, None]
	z = _endpoint_z(one[:, None, None], eta, sigma[:, None, None])
	pieces = (half * weights * density * torch.special.log_ndtr(z)).sum(dim=(1, 2))

	return torch.where(positive, pieces, torch.special.log_ndtr(_endpoint_z(one, m, sigma)))


def _endpoint_z(one: Tensor, eta: Tensor, scale: Tensor) -> Tensor:
	"""log k = log Phi(z): (eta - 1) / scale at 1, -eta / scale at 0."""
	return torch.where(one, (eta - 1.0) / scale, -eta / scale)

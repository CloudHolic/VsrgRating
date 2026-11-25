"""The encoder's training objective: CNRM cell log likelihood under a fixed normal person posterior
q = N(mu, Sigma)."""

import math

import numpy as np
import torch
from torch import Tensor

BOUNDARY_NODES = 21

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_NODES, _WEIGHTS = np.polynomial.hermite.hermgauss(BOUNDARY_NODES)


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
	"""E_q[log k] per cell: closed form inside (0, 1), Gauss-Hermite over eta at the endpoints."""
	interior, one, m, v = _prepare(x, g, log_dg, alpha, beta, sigma, mu, cov)
	g, log_dg = _interior_only(interior, g, log_dg)
	inside = -_HALF_LOG_2PI - torch.log(sigma) + log_dg - ((g - m) ** 2 + v) / (2.0 * sigma**2)

	nodes = torch.as_tensor(_NODES, dtype=x.dtype, device=x.device)
	weights = torch.as_tensor(_WEIGHTS / math.sqrt(math.pi), dtype=x.dtype, device=x.device)
	eta = m[:, None] + math.sqrt(2.0) * _sqrt(v)[:, None] * nodes
	z = _endpoint_z(one[:, None], eta, sigma[:, None])
	edge = (torch.special.log_ndtr(z) * weights).sum(dim=1)

	return torch.where(interior, inside, edge)


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


def _sqrt(v: Tensor) -> Tensor:
	# sqrt has an infinite slope at 0; v = 0 happens for a zero psoterior or alpha.
	positive = v > 0.0
	root = torch.sqrt(torch.where(positive, v, torch.ones_like(v)))
	return torch.where(positive, root, torch.zeros_like(v))


def _endpoint_z(one: Tensor, eta: Tensor, scale: Tensor) -> Tensor:
	"""log k = log Phi(z): (eta - 1) / scale at 1, -eta / scale at 0."""
	return torch.where(one, (eta - 1.0) / scale, -eta / scale)

"""Censored Normal Response Model (Minamimoto, Wakai & Okada, 2026)."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from ..partition import partition


class ItemParams(NamedTuple):
	"""Item parameters."""

	alpha: Array
	beta: Array
	sigma: Array


def log_k(
	x: Array,
	theta: Array,
	item: ItemParams,
	scale: tuple[Array, Array] | None = None,
) -> Array:
	"""Log probability at the endpoints and log density inside.

	`scale` is (G(x), log G'(x)) from a response-scale transform (TransformSpec.apply), evaluated
	at the interior responses; None is the paper's linear v, G(x) = x.
	"""
	if item.alpha.ndim > item.beta.ndim:
		mu = (item.alpha * theta).sum(axis=-1) + item.beta
	else:
		mu = item.alpha * theta + item.beta

	part = partition(x)
	g, log_dg = (x, 0.0) if scale is None else scale
	interior = norm.logpdf((g - mu) / item.sigma) - jnp.log(item.sigma) + log_dg
	lower = norm.logcdf(-mu / item.sigma)
	upper = norm.logcdf((mu - 1.0) / item.sigma)

	return jnp.where(part.zero, lower, jnp.where(part.one, upper, interior))

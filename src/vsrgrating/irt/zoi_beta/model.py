"""Zero and one inflated Beta IRT model (Molenaar, Curi & Bazan, 2022)."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.nn import log_sigmoid
from jax.scipy.special import gammaln
from jax.scipy.stats import norm

from ..partition import partition


class ItemParams(NamedTuple):
	"""Item parameters."""

	log_a: Array
	b: Array
	omega: Array
	gamma_0: Array
	gamma_1: Array


COORDS = ItemParams._fields


def log_k(x: Array, theta: Array, item: ItemParams) -> Array:
	"""Log probability at 0 and 1 and log density inside."""
	a_theta = jnp.exp(item.log_a) * theta
	u0 = item.gamma_0 - a_theta
	u1 = item.gamma_1 - a_theta
	part = partition(x)

	x_in = jnp.where(part.interior, x, 0.5)
	inside = (
		log_sigmoid(u1)
		+ log_sigmoid(-u0)
		+ jnp.log(-jnp.expm1(-(item.gamma_1 - item.gamma_0)))
		+ _log_beta(x_in, a_theta + item.b, item.omega)
	)

	return jnp.where(part.zero, log_sigmoid(u0), jnp.where(part.one, log_sigmoid(-u1), inside))


def log_prior(item: ItemParams, scale: float) -> Array:
	"""log h(tau): every coordinate N(0, scale^2), gamma_1 truncated to (gamma_0, inf)."""
	free = (
		norm.logpdf(item.log_a, scale=scale)
		+ norm.logpdf(item.b, scale=scale)
		+ norm.logpdf(item.omega, scale=scale)
		+ norm.logpdf(item.gamma_0, scale=scale)
	)

	truncated = norm.logpdf(item.gamma_1, scale=scale) - norm.logcdf(-item.gamma_0 / scale)
	truncated = jnp.where(item.gamma_1 > item.gamma_0, truncated, -jnp.inf)

	return (free + truncated).sum()


def _log_beta(x: Array, eta: Array, omega: Array) -> Array:
	a = jnp.exp((eta + omega) / 2.0)
	b = jnp.exp((omega - eta) / 2.0)
	log_norm = gammaln(a) + gammaln(b) - gammaln(a + b)

	return (a - 1.0) * jnp.log(x) + (b - 1.0) * jnp.log1p(-x) - log_norm

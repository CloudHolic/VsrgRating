"""Adaptive Gauss-Hermite quadrature for integrals against the N(0, 1) ability prior."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.stats import norm

_GRID = np.linspace(-8.0, 8.0, 321)
_FLAT_SCALE = 0.3  # node scale where the log posterior is not concave at the mode


def hermite(n: int) -> tuple[Array, Array]:
	"""Gauss-Hermite nodes and log weights for the kernel exp(-x^2)."""
	x, w = np.polynomial.hermite.hermgauss(n)
	with np.errstate(divide="ignore"):
		log_w = np.log(w)

	return jnp.asarray(x), jnp.asarray(log_w)


def adaptive_nodes(mode: Array, scale: Array, x_h: Array, log_w_h: Array) -> tuple[Array, Array]:
	"""Per-person nodes theta_ph = mode_p + sqrt(2) scale_p x_h, shape (persons, nodes), and
	log weights such that sum_h w_ph g(theta_ph) approximates the integral of g(theta) phi(theta).
	"""
	theta = mode[:, None] + jnp.sqrt(2.0) * scale[:, None] * x_h[None, :]
	log_w = (
		log_w_h[None, :]
		+ x_h[None, :] ** 2
		+ jnp.log(jnp.sqrt(2.0) * scale)[:, None]
		+ norm.logpdf(theta)
	)

	return theta, log_w


def posterior_modes(
	log_post: Callable[[Array], Array], start: Array, *, steps: int, clip: float, scan: bool
) -> tuple[Array, Array]:
	"""Each person's posterior mode of theta and the node scale (-f'')^(-1/2) there, by Newton."""
	grad = jax.grad(lambda theta: log_post(theta).sum())

	def curvature(theta: Array) -> Array:
		# The log posterior is a sum of one-person terms: its Hessian is diagonal, and one
		# Hessian-vector product with ones returns that diagonal.
		return jax.jvp(grad, (theta,), (jnp.ones_like(theta),))[1]

	mode = start

	if scan:
		# A non-concave posterior can hold Newton in a poor local mode.
		grid = jnp.asarray(_GRID)
		on_grid = jax.lax.map(lambda t: log_post(jnp.full_like(start, t)), grid)  # (G, persons)
		best = grid[jnp.argmax(on_grid, axis=0)]
		mode = jnp.where(log_post(start) >= on_grid.max(axis=0), start, best)

	def step(_: int, theta: Array) -> Array:
		h = curvature(theta)
		move = jnp.where(h < 0.0, -grad(theta) / h, 0.0)
		return theta + jnp.clip(move, -clip, clip)

	mode = jax.lax.fori_loop(0, steps, step, mode)
	h = curvature(mode)

	return mode, jnp.where(h < -1e-8, 1.0 / jnp.sqrt(-h), _FLAT_SCALE)

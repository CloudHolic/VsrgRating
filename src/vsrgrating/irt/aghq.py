"""Adaptive Gauss-Hermite quadrature for integrals against the N(0, 1) ability prior."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.stats import norm

_GRID = np.linspace(-8.0, 8.0, 321)
_FLAT_SCALE = 0.3  # node scale where the log posterior is not concave at the mode


def hermite(n: int, dims: int = 1) -> tuple[Array, Array]:
	"""Gauss-Hermite nodes and log weights for the kernel exp(-|x|^2)."""
	x, w = np.polynomial.hermite.hermgauss(n)
	with np.errstate(divide="ignore"):
		log_w = np.log(w)

	if dims == 1:
		return jnp.asarray(x), jnp.asarray(log_w)

	nodes = jnp.stack(jnp.meshgrid(*([x] * dims), indexing="ij"), axis=-1).reshape(-1, dims)
	weights = jnp.stack(jnp.meshgrid(*([log_w] * dims), indexing="ij"), axis=-1).reshape(-1, dims)

	return nodes, weights.sum(axis=-1)


def adaptive_nodes(mode: Array, scale: Array, x_h: Array, log_w_h: Array) -> tuple[Array, Array]:
	"""Per-person nodes theta_ph = mode_p + sqrt(2) scale_p x_h and log weights such that
	sum_h w_ph g(theta_ph) approximates the integral of g(theta) phi(theta).
	"""
	if mode.ndim == 1:
		theta = mode[:, None] + jnp.sqrt(2.0) * scale[:, None] * x_h[None, :]
		log_w = (
			log_w_h[None, :]
			+ x_h[None, :] ** 2
			+ jnp.log(jnp.sqrt(2.0) * scale)[:, None]
			+ norm.logpdf(theta)
		)
	else:
		dims = x_h.shape[1]
		theta = mode[:, None, :] + jnp.sqrt(2.0) * jnp.einsum("pij,hj->phi", scale, x_h)
		log_det = jnp.log(jnp.diagonal(scale, axis1=1, axis2=2)).sum(axis=-1)
		log_w = (
			log_w_h[None, :]
			+ (x_h**2).sum(axis=-1)[None, :]
			+ 0.5 * dims * jnp.log(2.0)
			+ log_det[:, None]
			+ norm.logpdf(theta).sum(axis=-1)
		)

	return theta, log_w


def posterior_modes(
	log_post: Callable[[Array], Array], start: Array, *, steps: int, clip: float, scan: bool
) -> tuple[Array, Array]:
	"""Each person's posterior mode of theta and the node scale there, by Newton."""
	if start.ndim == 2 and scan:
		raise ValueError("the grid scan is one-dimensional; pass scan=False with D >= 2")

	grad = jax.grad(lambda theta: log_post(theta).sum())
	newton, spread = _scalar(grad) if start.ndim == 1 else _blocks(grad, start)

	mode = start

	if scan:
		# A non-concave posterior can hold NEwton in a poor local mode.
		grid = jnp.asarray(_GRID)
		on_grid = jax.lax.map(lambda t: log_post(jnp.full_like(start, t)), grid)  # (G, persons)
		best = grid[jnp.argmax(on_grid, axis=0)]
		mode = jnp.where(log_post(start) >= on_grid.max(axis=0), start, best)

	def step(_: int, theta: Array) -> Array:
		move, concave = newton(theta)
		return theta + jnp.clip(jnp.where(concave, move, 0.0), -clip, clip)

	mode = jax.lax.fori_loop(0, steps, step, mode)

	return mode, spread(mode)


type Newton = Callable[[Array], tuple[Array, Array]]
type Spread = Callable[[Array], Array]


def _scalar(grad: Callable[[Array], Array]) -> tuple[Newton, Spread]:
	def curvature(theta: Array) -> Array:
		# The log posterior is a sum of one-person terms: its Hessian is diagonal,
		# and one Hessian-vector product with ones returns that diagonal.
		return jax.jvp(grad, (theta,), (jnp.ones_like(theta),))[1]

	def newton(theta: Array) -> tuple[Array, Array]:
		h = curvature(theta)
		return -grad(theta) / h, h < 0.0

	def spread(mode: Array) -> Array:
		h = curvature(mode)
		return jnp.where(h < -1e-8, 1.0 / jnp.sqrt(-h), _FLAT_SCALE)

	return newton, spread


def _blocks(grad: Callable[[Array], Array], start: Array) -> tuple[Newton, Spread]:
	dims = start.shape[1]
	basis = [jnp.zeros_like(start).at[:, k].set(1.0) for k in range(dims)]

	def factor(theta: Array) -> tuple[Array, Array, Array]:
		# The log posterior is a sum of one-person terms, so H is block diagonal:
		# one linearization of the gradient, then on tangent per coordinate, gives every block.
		g, hvp = jax.linearize(grad, theta)
		chol = jnp.linalg.cholesky(-jnp.stack([hvp(e) for e in basis], axis=-1))
		diagonal = jnp.diagonal(chol, axis1=1, axis2=2)
		concave = jnp.isfinite(chol).all(axis=(1, 2)) & (diagonal > 1e-4).all(axis=1)

		return g, chol, concave

	def newton(theta: Array) -> tuple[Array, Array]:
		g, chol, concave = factor(theta)
		# (-H)^(-1) g from -H = L L^T: solve L y = g, then L^T move = y.
		y = jax.scipy.linalg.solve_triangular(chol, g[..., None], lower=True)
		move = jax.scipy.linalg.solve_triangular(chol, y, lower=True, trans=1)[..., 0]

		return move, concave[:, None]

	def spread(mode: Array) -> Array:
		_, chol, concave = factor(mode)
		# The posterior covariance is L^-T L^-1, so L^-T is a square root of it.
		root = jnp.swapaxes(jnp.linalg.inv(chol), 1, 2)
		return jnp.where(concave[:, None, None], root, _FLAT_SCALE * jnp.eye(dims))

	return newton, spread

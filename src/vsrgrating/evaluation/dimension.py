"""Whether 2 parts of the items measure one ability: the latent correlation of their thetas."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.stats import norm
from scipy.optimize import minimize

# On the grid a bivariate normal with rho near 1 is a ridge narrower than the spacing.
RHO_MAX = 0.999
_BOUNDS = [(-8.0, 8.0), (-8.0, 8.0), (-5.0, 3.0), (-5.0, 3.0), (0.0, RHO_MAX)]
_TINY = 1e-300


@dataclass(frozen=True)
class Bivariate:
	"""The fitted distribution of (theta_1, theta_2) and its maximized log likelihood."""

	mean: tuple[float, float]
	sd: tuple[float, float]
	rho: float
	loglik: float


def latent_correlation(first: np.ndarray, second: np.ndarray, grid: np.ndarray) -> Bivariate:
	"""Maximum likelihood mean, sd and correlation from the same persons' grid log likelihoods
	in 2 parts, each shaped (persons, grid)."""
	args = _prepare(first, second, grid)
	weights = jnp.ones(first.shape[0])
	offset = float(first.max(axis=1).sum() + second.max(axis=1).sum())

	free, value = _minimize(_FREE, _start(first, second, grid), _BOUNDS, (*args, weights))

	return Bivariate(
		mean=(float(free[0]), float(free[1])),
		sd=(float(jnp.exp(free[2])), float(jnp.exp(free[3]))),
		rho=float(free[4]),
		loglik=offset - value,
	)


def rho_se(
	first: np.ndarray, second: np.ndarray, grid: np.ndarray, n_boot: int = 200, seed: int = 0
) -> float:
	"""The standard error of rho by resampling persons, each refit started at the estimate."""
	args = _prepare(first, second, grid)
	persons = first.shape[0]
	start, _ = _minimize(_FREE, _start(first, second, grid), _BOUNDS, (*args, jnp.ones(persons)))

	rng = np.random.default_rng(seed)
	draws = []

	for _ in range(n_boot):
		weights = jnp.asarray(rng.multinomial(persons, np.full(persons, 1.0 / persons)))
		estimate, _ = _minimize(_FREE, start, _BOUNDS, (*args, weights.astype(jnp.float64)))
		draws.append(estimate[4])

	return float(np.std(draws))


def _prepare(first: np.ndarray, second: np.ndarray, grid: np.ndarray) -> tuple[Array, ...]:
	"""Likelihoods scaled by each row's maximum, so sums over the grid stay in range."""
	if first.shape != second.shape or first.shape[1] != grid.size:
		raise ValueError(f"shapes disagree: {first.shape}, {second.shape}, grid {grid.size}")

	a = np.exp(first - first.max(axis=1, keepdims=True))
	log_b = second - second.max(axis=1, keepdims=True)

	return jnp.asarray(a), jnp.asarray(np.exp(log_b)), jnp.asarray(log_b), jnp.asarray(grid)


def _start(first: np.ndarray, second: np.ndarray, grid: np.ndarray) -> np.ndarray:
	"""Moments of each person's most likely theta in each part."""
	t1 = np.clip(grid[first.argmax(axis=1)], -4.0, 4.0)
	t2 = np.clip(grid[second.argmax(axis=1)], -4.0, 4.0)
	rho = float(np.clip(np.corrcoef(t1, t2)[0, 1], 0.1, 0.95))

	return np.array([t1.mean(), t2.mean(), np.log(t1.std()), np.log(t2.std()), rho])


def _minimize(
	objective, start: np.ndarray, bounds: list[tuple[float, float]], args: tuple[Array, ...]
) -> tuple[np.ndarray, float]:
	def value_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
		value, grad = objective(jnp.asarray(params), *args)
		return float(value), np.asarray(grad, dtype=np.float64)

	result = minimize(value_and_grad, start, jac=True, method="L-BFGS-B", bounds=bounds)
	if not np.isfinite(result.fun):
		raise FloatingPointError(f"non-finite likelihood at {result.x}")

	return result.x, float(result.fun)


def _free(params: Array, a: Array, b: Array, log_b: Array, grid: Array, weights: Array) -> Array:
	"""Minus the weighted log likelihood, up to the rows' maxima."""
	mu1, mu2, log_s1, log_s2, rho = params
	s1, s2 = jnp.exp(log_s1), jnp.exp(log_s2)
	step = grid[1] - grid[0]

	mean = mu2 + rho * s2 * (grid - mu1) / s1
	spread = s2 * jnp.sqrt(1.0 - rho**2)
	joint = norm.logpdf(grid, mu1, s1)[:, None] + norm.logpdf(grid[None, :], mean[:, None], spread)
	marginal = ((a @ jnp.exp(joint)) * b).sum(axis=1) * step**2

	return -(weights * jnp.log(marginal + _TINY)).sum()


_FREE = jax.jit(jax.value_and_grad(_free))

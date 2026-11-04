"""EM-based marginal maximum likelihood for the CNRM."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from ..aghq import adaptive_nodes, hermite, posterior_modes
from ..partition import partition
from .model import ItemParams, log_k

_NEWTON_STEPS = 25
_NEWTON_CLIP = 2.0


@dataclass(frozen=True)
class EMConfig:
	"""Quadrature and stopping rule."""

	n_nodes: int
	tol: float
	max_iter: int


@dataclass(frozen=True)
class EMResult:
	"""Item estimates, the marginal log-likelihood per iteration, and train-person EAPs."""

	item: ItemParams
	loglik: np.ndarray
	n_iter: int
	converged: bool
	degenerate_items: np.ndarray
	eap: np.ndarray
	psd: np.ndarray

	@property
	def degenerate(self) -> bool:
		"""True when EM stopped because an update turned non-finite."""
		return self.degenerate_items.size > 0


def init_person_mean_ols(
	person: np.ndarray, item: np.ndarray, x: np.ndarray, n_persons: int, n_items: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Item-wise OLS of responses on standardized person means (slope, intercept, residual SE)."""
	mean_p = np.bincount(person, x, n_persons) / np.bincount(person, minlength=n_persons)
	t = ((mean_p - mean_p.mean()) / mean_p.std())[person]

	n = np.bincount(item, minlength=n_items).astype(float)
	t_bar = np.bincount(item, t, n_items) / n
	x_bar = np.bincount(item, x, n_items) / n

	dt = t - t_bar[item]
	sxx = np.bincount(item, dt**2, n_items)
	sxy = np.bincount(item, dt * (x - x_bar[item]), n_items)

	with np.errstate(divide="ignore", invalid="ignore"):
		slope = sxy / sxx
		intercept = x_bar - slope * t_bar
		sse = np.bincount(item, (x - intercept[item] - slope[item] * t) ** 2, n_items)
		rse = np.sqrt(sse / (n - 2.0))

	dt_all = t - t.mean()
	slope_all = (dt_all * (x - x.mean())).sum() / (dt_all**2).sum()
	intercept_all = x.mean() - slope_all * t.mean()
	rse_all = np.sqrt(((x - intercept_all - slope_all * t) ** 2).sum() / (len(x) - 2.0))

	valid = (sxx > 0.0) & (n > 2.0) & (rse > 0.0)

	return (
		np.where(valid, slope, slope_all),
		np.where(valid, intercept, intercept_all),
		np.where(valid, rse, rse_all),
	)


def fit_em(
	person: np.ndarray,
	item: np.ndarray,
	x: np.ndarray,
	n_persons: int,
	n_items: int,
	config: EMConfig,
) -> EMResult:
	"""Runs EM until every parameter moves less than `tol` or `max_iter` is reached."""
	x_h, log_w_h = hermite(config.n_nodes)
	p, i, xs = jnp.asarray(person), jnp.asarray(item), jnp.asarray(x)
	params = ItemParams(
		*(jnp.asarray(v) for v in init_person_mean_ols(person, item, x, n_persons, n_items))
	)
	mode = jnp.zeros(n_persons)

	loglik = []
	converged = False
	degenerate_items = np.array([], dtype=np.int64)

	for _ in range(config.max_iter):
		mode, scale = _modes(p, i, xs, n_persons, params, mode)
		theta, log_w = adaptive_nodes(mode, scale, x_h, log_w_h)
		updated, ll = _em_step(p, i, xs, n_persons, n_items, params, theta, log_w)
		loglik.append(float(ll))

		finite = (
			jnp.isfinite(updated.alpha) & jnp.isfinite(updated.beta) & jnp.isfinite(updated.sigma)
		)
		if not bool(finite.all()):
			# Nothing in MML bounds sigma away from 0: an item with too few responses drives it
			# there until the posterior over the nodes collapses and the update is 0/0.
			degenerate_items = np.flatnonzero(~np.asarray(finite))
			break

		change = max(float(jnp.abs(u - v).max()) for u, v in zip(updated, params, strict=True))
		params = updated

		if change < config.tol:
			converged = True
			break

	mode, scale = _modes(p, i, xs, n_persons, params, mode)
	theta, log_w = adaptive_nodes(mode, scale, x_h, log_w_h)
	post, _ = _posterior(p, i, xs, n_persons, params, theta, log_w)
	eap = (post * theta).sum(axis=1)
	psd = jnp.sqrt((post * (theta - eap[:, None]) ** 2).sum(axis=1))

	return EMResult(
		item=params,
		loglik=np.asarray(loglik),
		n_iter=len(loglik),
		converged=converged,
		degenerate_items=degenerate_items,
		eap=np.asarray(eap),
		psd=np.asarray(psd),
	)


def person_loglik(
	person: np.ndarray,
	item: np.ndarray,
	x: np.ndarray,
	n_persons: int,
	params: ItemParams,
	n_nodes: int,
) -> np.ndarray:
	"""Each person's log marginal likelihood by Adaptive Gauss-Hermite with `n_nodes` nodes."""
	p, i, xs = jnp.asarray(person), jnp.asarray(item), jnp.asarray(x)
	mode, scale = _modes(p, i, xs, n_persons, params, jnp.zeros(n_persons))
	theta, log_w = adaptive_nodes(mode, scale, *hermite(n_nodes))
	_, log_marginal = _posterior(p, i, xs, n_persons, params, theta, log_w)

	return np.asarray(log_marginal)


@partial(jax.jit, static_argnames=("n_persons",))
def _modes(
	person: Array, item: Array, x: Array, n_persons: int, params: ItemParams, start: Array
) -> tuple[Array, Array]:
	def log_post(theta: Array) -> Array:
		lk = log_k(x, theta[person], ItemParams(*(v[item] for v in params)))
		return jax.ops.segment_sum(lk, person, num_segments=n_persons) + norm.logpdf(theta)

	return posterior_modes(log_post, start, steps=_NEWTON_STEPS, clip=_NEWTON_CLIP, scan=False)


@partial(jax.jit, static_argnames=("n_persons",))
def _posterior(
	person: Array,
	item: Array,
	x: Array,
	n_persons: int,
	params: ItemParams,
	theta: Array,
	log_w: Array,
) -> tuple[Array, Array]:
	"""Posterior weights over each person's nodes and each log marginal likelihood."""
	lk = log_k(x[:, None], theta[person], ItemParams(*(v[item, None] for v in params)))
	joint = jax.ops.segment_sum(lk, person, num_segments=n_persons) + log_w
	log_marginal = logsumexp(joint, axis=1)

	return jnp.exp(joint - log_marginal[:, None]), log_marginal


@partial(jax.jit, static_argnames=("n_persons", "n_items"))
def _em_step(
	person: Array,
	item: Array,
	x: Array,
	n_persons: int,
	n_items: int,
	params: ItemParams,
	theta: Array,
	log_w: Array,
) -> tuple[ItemParams, Array]:
	"""One E-step and M-step."""
	post, log_marginal = _posterior(person, item, x, n_persons, params, theta, log_w)

	t = theta[person]
	a, b, s = (v[item, None] for v in params)
	mu = a * t + b

	z0 = -mu / s
	z1 = (1.0 - mu) / s

	mills0 = jnp.exp(norm.logpdf(z0) - norm.logcdf(z0))
	mills1 = jnp.exp(norm.logpdf(z1) - norm.logcdf(-z1))

	part = partition(x)
	zero, one, xc = part.zero[:, None], part.one[:, None], x[:, None]

	ey = jnp.where(zero, mu - s * mills0, jnp.where(one, mu + s * mills1, xc))
	ey2 = jnp.where(
		zero,
		mu**2 - s * mu * mills0 + s**2,
		jnp.where(one, mu**2 + s * (1.0 + mu) * mills1 + s**2, xc**2),
	)

	w = post[person]
	n_i = jax.ops.segment_sum(jnp.ones_like(x), item, num_segments=n_items)

	def mean(v: Array) -> Array:
		return jax.ops.segment_sum(v.sum(axis=1), item, num_segments=n_items) / n_i

	m_t, m_t2 = mean(w * t), mean(w * t**2)
	m_y, m_y2, m_ty = mean(w * ey), mean(w * ey2), mean(w * t * ey)

	alpha_new = (m_ty - m_t * m_y) / (m_t2 - m_t**2)
	beta_new = m_y - alpha_new * m_t
	sigma_new = jnp.sqrt(
		m_y2
		- 2.0 * alpha_new * m_ty
		- 2.0 * beta_new * m_y
		+ alpha_new**2 * m_t2
		+ 2.0 * alpha_new * beta_new * m_t
		+ beta_new**2
	)

	return ItemParams(alpha_new, beta_new, sigma_new), log_marginal.sum()

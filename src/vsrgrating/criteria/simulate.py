"""Responses drawn from a fit: what Q5 would see if the model were exactly right."""

import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from ..evaluation import Fitted
from ..irt.cnrm.model import ItemParams as CnrmParams
from ..irt.cnrm.transform import TransformSpec
from ..irt.zoi_beta.model import ItemParams as ZoiParams

_S_GRID = np.linspace(-40.0, 0.0, 400_001)


def draw(
	rng: np.random.Generator, fitted: Fitted, item: np.ndarray, theta: np.ndarray
) -> np.ndarray:
	"""One response per (item, theta) pair."""
	params = fitted.item
	if isinstance(params, CnrmParams):
		assert fitted.model.transform is not None and fitted.raw is not None
		return _cnrm(rng, params, fitted.model.transform, fitted.raw, item, theta)

	return _zoi_beta(rng, params, item, theta)


def _zoi_beta(
	rng: np.random.Generator, params: ZoiParams, item: np.ndarray, theta: np.ndarray
) -> np.ndarray:
	log_a, b, omega, gamma_0, gamma_1 = (np.asarray(c)[item] for c in params)
	a_theta = np.exp(log_a) * theta
	p_zero = expit(gamma_0 - a_theta)
	p_one = expit(a_theta - gamma_1)

	eta = a_theta + b
	inside = rng.beta(np.exp((eta + omega) / 2.0), np.exp((omega - eta) / 2.0))
	u = rng.random(theta.shape)

	return np.where(u < p_zero, 0.0, np.where(u < p_zero + p_one, 1.0, inside))


def _cnrm(
	rng: np.random.Generator,
	params: CnrmParams,
	spec: TransformSpec,
	raw: jnp.ndarray,
	item: np.ndarray,
	theta: np.ndarray,
) -> np.ndarray:
	alpha, beta, sigma = (np.asarray(c)[item] for c in params)
	load = (alpha * theta).sum(axis=-1) if alpha.ndim == 2 else alpha * theta
	latent = load + beta + sigma * rng.standard_normal(beta.shape)

	# G rises as s falls, and np.interp wants its x increasing.
	s = _S_GRID[::-1]
	g, _ = spec.apply(raw, -jnp.expm1(jnp.asarray(s)))
	x = -np.expm1(np.interp(latent, np.asarray(g), s))

	return np.where(latent <= 0.0, 0.0, np.where(latent >= 1.0, 1.0, x))

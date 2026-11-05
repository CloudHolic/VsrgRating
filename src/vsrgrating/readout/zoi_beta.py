"""ZOI-Beta readouts on the theta scale, each at a response level y. Larger is harder.

- b*(y): the interior mean is y.
- b~(y): the mean response, endpoints included, is y.
- b50(y): P(Y >= y | theta) = 1/2.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.nn import log_sigmoid
from jax.scipy.special import betainc

from ..irt.zoi_beta.model import ItemParams
from .roots import increasing_root


def b_star(item: ItemParams, y: float) -> np.ndarray:
	"""Each item's b*(y)."""
	return (np.log(y / (1.0 - y)) - np.asarray(item.b)) / np.exp(np.asarray(item.log_a))


def b_tilde(item: ItemParams, y: float) -> np.ndarray:
	"""Each item's b~(y), solved in a·theta like b50."""
	guess = np.log(y / (1.0 - y)) - np.asarray(item.b)
	root = increasing_root(lambda s: _mean_response(item, jnp.asarray(s)), y, guess)

	return root / np.exp(np.asarray(item.log_a))


def exceedance(item: ItemParams, theta: Array, y: float) -> Array:
	"""P(Y >= y | theta): the mass at 1 plus the interior mass above y."""
	return _exceedance(item, jnp.exp(item.log_a) * theta, y)


def b50(item: ItemParams, y: float) -> np.ndarray:
	"""Each item's b50(y), started from where the interior mean is y."""
	guess = np.log(y / (1.0 - y)) - np.asarray(item.b)
	root = increasing_root(lambda s: _exceedance(item, jnp.asarray(s), y), 0.5, guess)

	return root / np.exp(np.asarray(item.log_a))


def _exceedance(item: ItemParams, a_theta: Array, y: float) -> Array:
	p_one, p_interior = _masses(item, a_theta)
	eta = a_theta + item.b
	alpha = jnp.exp((eta + item.omega) / 2.0)
	beta = jnp.exp((item.omega - eta) / 2.0)

	# I_{1-y}(beta, alpha) is 1 - I_y(alpha, beta) without subtracting from one.
	return p_one + p_interior * betainc(beta, alpha, 1.0 - y)


def _mean_response(item: ItemParams, a_theta: Array) -> Array:
	p_one, p_interior = _masses(item, a_theta)
	return p_one + p_interior * jax.nn.sigmoid(a_theta + item.b)


def _masses(item: ItemParams, a_theta: Array) -> tuple[Array, Array]:
	"""P(y = 1) and P(0 < y < 1) at a_theta."""
	u0 = item.gamma_0 - a_theta
	u1 = item.gamma_1 - a_theta
	p_interior = jnp.exp(
		log_sigmoid(u1) + log_sigmoid(-u0) + jnp.log(-jnp.expm1(-(item.gamma_1 - item.gamma_0)))
	)

	return jax.nn.sigmoid(-u1), p_interior

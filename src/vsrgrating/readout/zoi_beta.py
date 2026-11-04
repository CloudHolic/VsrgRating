"""ZOI-Beta b50(y): the ability at which P(Y >= y | theta) = 1/2."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.nn import log_sigmoid
from jax.scipy.special import betainc

from ..irt.zoi_beta.model import ItemParams
from .roots import increasing_root


def exceedance(item: ItemParams, theta: Array, y: float) -> Array:
	"""P(Y >= y | theta): the mass at 1 plus the interior mass above y."""
	a_theta = jnp.exp(item.log_a) * theta
	u0 = item.gamma_0 - a_theta
	u1 = item.gamma_1 - a_theta
	p_interior = jnp.exp(
		log_sigmoid(u1) + log_sigmoid(-u0) + jnp.log(-jnp.expm1(-(item.gamma_1 - item.gamma_0)))
	)

	eta = a_theta + item.b
	alpha = jnp.exp((eta + item.omega) / 2.0)
	beta = jnp.exp((item.omega - eta) / 2.0)

	# I_{1-y}(beta, alpha) is 1 - I_y(alpha, beta) without subtracting from one.
	return jax.nn.sigmoid(-u1) + p_interior * betainc(beta, alpha, 1.0 - y)


def b50(item: ItemParams, y: float) -> np.ndarray:
	"""Each item's b50(y), started from where the interior mean is y."""
	guess = (np.log(y / (1.0 - y)) - np.asarray(item.b)) / np.exp(np.asarray(item.log_a))

	return increasing_root(lambda t: exceedance(item, jnp.asarray(t), y), 0.5, guess)

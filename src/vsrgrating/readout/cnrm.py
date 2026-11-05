"""CNRM b^{1/2}(y): the ability at which P(Y >= y | theta) = 1/2. Larger is harder."""

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..irt.cnrm.model import ItemParams
from ..irt.cnrm.transform import TransformSpec


def b_half(item: ItemParams, spec: TransformSpec, raw: Array, y: float) -> np.ndarray:
	"""Each item's b^{1/2}(y) = (G(y) - beta) / alpha.

	P(Y >= y | theta) = Phi((alpha theta + beta - G(y)) / sigma), so sigma drops out.
	"""
	g, _ = spec.apply(raw, jnp.asarray(y))
	return (float(g) - np.asarray(item.beta)) / np.asarray(item.alpha)

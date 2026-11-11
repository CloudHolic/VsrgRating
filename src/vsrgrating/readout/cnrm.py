"""CNRM d(y): how far the boundary where P(Y >= y | theta) = 1/2 lies from the population's
centre, on every chart's own ability axis. It is b^{1/2}(y) in one dimension. Larger is harder."""

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..irt.cnrm.model import ItemParams
from ..irt.cnrm.transform import TransformSpec


def difficulty(item: ItemParams, spec: TransformSpec, raw: Array, y: float) -> np.ndarray:
	"""Each item's d(y) = (G(y) - beta) / |alpha|."""
	g, _ = spec.apply(raw, jnp.asarray(y))
	alpha = np.asarray(item.alpha)
	size = np.linalg.norm(alpha, axis=-1) if alpha.ndim == 2 else alpha

	return (float(g) - np.asarray(item.beta)) / size

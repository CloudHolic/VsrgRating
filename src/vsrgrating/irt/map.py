"""MAP of the item parameters with theta marginalized by adaptive Gauss-Hermite quadrature."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from .aghq import adaptive_nodes, hermite, posterior_modes
from .cnrm import model as cnrm
from .cnrm.transform import TransformSpec
from .zoi_beta import model as zoi_beta

_NEWTON_STEPS = 8
_NEWTON_CLIP = 0.5
# With D >= 2 the extra axes start off zero: all-zero is a saddle where their gradient vanishes.
_ALPHA_JITTER = 0.1
INIT_SEED = 20261111
# Modes sought from 0 rather than from a previous step need more than 1 Newton round.
_SUMMARY_ROUNDS = 5
_SUMMARY_CHUNK = 4096


@dataclass(frozen=True)
class MAPConfig:
	"""Quadrature, optimizer (Adam with cosine decay), node re-centring schedule, and the seed of
	the start's extra alpha axes."""

	n_nodes: int
	steps: int
	learning_rate: float
	final_lr_fraction: float
	scan_every: int
	prior_scale: float
	init_seed: int = INIT_SEED


class Cells(NamedTuple):
	"""Observed train cells. x_in is x with the endpoints replaced by 0.5 (transform input)."""

	person: Array
	item: Array
	x: Array
	x_in: Array


@dataclass(frozen=True)
class Model:
	"""ZOI-Beta, or CNRM with a shared response-scale transform. Hashable (jit-static)."""

	family: str
	transform: TransformSpec | None = None
	dims: int = 1

	def __post_init__(self) -> None:
		if self.dims < 1 or (self.dims > 1 and self.family != "cnrm"):
			raise ValueError(f"only the cnrm takes dims > 1, got {self.family} with {self.dims}")

		if self.family == "cnrm":
			if self.transform is None:
				raise ValueError("cnrm needs a transform (kind: linear for the paper's v)")
		elif self.family != "zoi_beta":
			raise ValueError(f"unknown family {self.family!r}")
		elif self.transform is not None:
			raise ValueError("only the cnrm takes a response-scale transform")

	def cell_log_k(self, params: dict[str, Array], cells: Cells, theta: Array) -> Array:
		"""log k at every cell; theta is (cells,) or (cells, nodes)."""
		return self.item_log_k(self.item_params(params), params.get("transform"), cells, theta)

	def item_params(self, params: dict[str, Array]) -> zoi_beta.ItemParams | cnrm.ItemParams:
		"""The model's coordinates from the optimizer's."""
		if self.family == "cnrm":
			alpha = jnp.exp(params["log_alpha"]) if "log_alpha" in params else params["alpha"]
			return cnrm.ItemParams(alpha, params["beta"], jnp.exp(params["log_sigma"]))

		return zoi_beta.ItemParams(*zoi_tau(params))

	def item_log_k(
		self,
		item: zoi_beta.ItemParams | cnrm.ItemParams,
		raw: Array | None,
		cells: Cells,
		theta: Array,
	) -> Array:
		"""log k at every cell from the model's coordinates; `raw` is G's parameters (cnrm only)."""
		# theta is (cells,) or (cells, nodes) in 1 dimension, with a trailing D axis in more.
		on_nodes = theta.ndim > (1 if self.dims == 1 else 2)

		col = (lambda v: v[cells.item][:, None]) if on_nodes else (lambda v: v[cells.item])
		x = cells.x[:, None] if on_nodes else cells.x
		picked = jax.tree.map(col, item)

		if self.family == "cnrm":
			assert self.transform is not None
			assert raw is not None

			g, log_dg = self.transform.apply(raw, cells.x_in)
			scale = (g[:, None], log_dg[:, None]) if on_nodes else (g, log_dg)

			return cnrm.log_k(x, theta, picked, scale)

		return zoi_beta.log_k(x, theta, picked)

	def log_prior(self, params: dict[str, Array], scale: float) -> Array:
		"""ZOI: the paper's h(tau) at tau(z), no Jacobian (MAP). CNRM: N(0, scale) everywhere,
		plus the transform's smoothness penalty (P-spline only)."""
		if self.family == "cnrm":
			assert self.transform is not None

			flat = sum(
				(norm.logpdf(v, scale=scale).sum() for v in jax.tree_util.tree_leaves(params)),
				jnp.array(0.0),
			)

			return flat + self.transform.log_penalty(params["transform"])

		return zoi_beta.log_prior(zoi_beta.ItemParams(*zoi_tau(params)), scale)


@dataclass(frozen=True)
class MAPResult:
	"""Point estimates, the loss trace, and train-person EAPs at the estimate."""

	params: dict[str, np.ndarray]
	loss: np.ndarray
	best_loss: float
	best_step: int
	log_marginal: float
	eap: np.ndarray
	psd: np.ndarray

	@property
	def drop_last(self) -> float:
		"""How much the loss fell over the last 200 steps: the convergence check."""
		k = min(200, len(self.loss) - 1)
		return float(self.loss[-k - 1] - self.loss[-1]) if k > 0 else float("nan")


def zoi_tau(params: dict[str, Array]) -> tuple[Array, ...]:
	"""(log_a, b, omega, gamma_0, gamma_1) with gamma_1 = gamma_0 + exp(lambda) > gamma_0."""
	return (
		params["log_a"],
		params["b"],
		params["omega"],
		params["gamma_0"],
		params["gamma_0"] + jnp.exp(params["lambda"]),
	)


def zoi_params(
	log_a: np.ndarray,
	b: np.ndarray,
	omega: np.ndarray,
	gamma_0: np.ndarray,
	gamma_1: np.ndarray,
) -> dict[str, Array]:
	"""The optimizer's coordinates from tau (inverse of zoi_tau)."""
	gap = np.maximum(np.asarray(gamma_1) - np.asarray(gamma_0), 1e-3)
	return {
		"log_a": jnp.asarray(log_a),
		"b": jnp.asarray(b),
		"omega": jnp.asarray(omega),
		"gamma_0": jnp.asarray(gamma_0),
		"lambda": jnp.asarray(np.log(gap)),
	}


def cnrm_params(
	alpha: np.ndarray,
	beta: np.ndarray,
	sigma: np.ndarray,
	spec: TransformSpec,
	dims: int = 1,
	seed: int = INIT_SEED,
) -> dict[str, Array]:
	"""The optimizer's coordinates: (log alpha, beta, log sigma) as in the paper, plus G's."""
	result = {
		"beta": jnp.asarray(beta),
		"log_sigma": jnp.asarray(np.log(sigma)),
		"transform": spec.init(),
	}

	if dims == 1:
		result["log_alpha"] = jnp.asarray(np.log(np.maximum(np.asarray(alpha), 1e-3)))
	else:
		start = np.random.default_rng(seed).normal(0.0, _ALPHA_JITTER, (len(alpha), dims))
		start[:, 0] = np.maximum(np.asarray(alpha), 1e-3)
		result["alpha"] = jnp.asarray(start)

	return result


def fit_map(
	model: Model,
	person: np.ndarray,
	item: np.ndarray,
	x: np.ndarray,
	n_persons: int,
	init: dict[str, Array],
	config: MAPConfig,
	log_every: int = 100,
) -> MAPResult:
	"""Adam on the AGHQ-marginalized MAP objective, keeping the best iterate."""
	cells = _cells(person, item, x)
	x_h, log_w_h = hermite(config.n_nodes, model.dims)
	params = init

	m1 = jax.tree_util.tree_map(jnp.zeros_like, params)
	m2 = jax.tree_util.tree_map(jnp.zeros_like, params)
	mode = jnp.zeros(n_persons if model.dims == 1 else (n_persons, model.dims))

	trace: list[float] = []
	best = (np.inf, params, 0)

	for step in range(config.steps):
		scan = model.dims == 1 and step % config.scan_every == 0
		mode, scale = _modes(model, params, cells, n_persons, mode, scan)
		nodes, log_w = adaptive_nodes(mode, scale, x_h, log_w_h)

		value, grads = _value_and_grad(
			model, params, cells, n_persons, nodes, log_w, config.prior_scale
		)
		value = float(value)

		if not np.isfinite(value) or not all(
			bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads)
		):
			raise FloatingPointError(f"non-finite loss or gradient at step {step}")

		trace.append(value)
		if value < best[0]:
			best = (value, params, step)

		if log_every and step % log_every == 0:
			print(f"step {step} loss {value:.3f}", flush=True)

		lr = _cosine(step, config)
		params, m1, m2 = _adam(params, grads, m1, m2, step + 1, lr)

	params = best[1]
	log_marginal, eap, cov = person_summary(
		model, params, person, item, x, n_persons, config.n_nodes, mode
	)

	return MAPResult(
		params={k: np.asarray(v) for k, v in params.items()},
		loss=np.asarray(trace),
		best_loss=float(best[0]),
		best_step=int(best[2]),
		log_marginal=float(log_marginal.sum()),
		eap=np.asarray(eap),
		psd=np.asarray(posterior_sd(cov)),
	)


def person_summary(
	model: Model,
	params: dict[str, Array],
	person: np.ndarray,
	item: np.ndarray,
	x: np.ndarray,
	n_persons: int,
	n_nodes: int,
	start: Array | None = None,
) -> tuple[Array, Array, Array]:
	"""Each person's log marginal likelihood, posterior mean and covariance at `params`, by adaptive
	Gauss-Hermite around the posterior mode. Without `start` the modes are sought from zero."""
	cells = _cells(person, item, x)
	x_h, log_w_h = hermite(n_nodes, model.dims)

	if start is None:
		mode = jnp.zeros(n_persons if model.dims == 1 else (n_persons, model.dims))
		rounds = _SUMMARY_ROUNDS
	else:
		mode, rounds = start, 1

	for _ in range(rounds):
		mode, scale = _modes(model, params, cells, n_persons, mode, model.dims == 1)

	nodes, log_w = adaptive_nodes(mode, scale, x_h, log_w_h)
	joint = _chunked_joint(model, params, person, item, x, n_persons, nodes) + log_w
	log_marginal = logsumexp(joint, axis=1)
	post = jnp.exp(joint - log_marginal[:, None])

	if nodes.ndim == 2:
		mean = (post * nodes).sum(axis=1)
		return log_marginal, mean, (post * (nodes - mean[:, None]) ** 2).sum(axis=1)

	mean = jnp.einsum("ph,phi->pi", post, nodes)
	diff = nodes - mean[:, None, :]

	return log_marginal, mean, jnp.einsum("ph,phi,phj->pij", post, diff, diff)


def posterior_sd(cov: Array) -> Array:
	"""Per-axis posterior sd from person_summary's covariance."""
	return jnp.sqrt(cov if cov.ndim == 1 else jnp.diagonal(cov, axis1=1, axis2=2))


def _cells(person: np.ndarray, item: np.ndarray, x: np.ndarray) -> Cells:
	xs = jnp.asarray(x)
	return Cells(
		jnp.asarray(person), jnp.asarray(item), xs, jnp.where((xs == 0.0) | (xs == 1.0), 0.5, xs)
	)


def _cosine(step: int, config: MAPConfig) -> float:
	frac = config.final_lr_fraction
	return config.learning_rate * (
		frac + (1.0 - frac) * 0.5 * (1.0 + np.cos(np.pi * step / config.steps))
	)


Params = dict[str, Array]


@jax.jit
def _adam(
	params: Params, grads: Params, m1: Params, m2: Params, t: int, lr: float
) -> tuple[Params, Params, Params]:
	b1, b2, eps = 0.9, 0.999, 1e-8

	m1 = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m1, grads)
	m2 = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g**2, m2, grads)
	c1, c2 = 1 - b1**t, 1 - b2**t

	params = jax.tree_util.tree_map(
		lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + eps), params, m1, m2
	)

	return params, m1, m2


def _joint(
	model: Model, params: dict[str, Array], cells: Cells, n_persons: int, nodes: Array, log_w: Array
) -> Array:
	"""log L_p(theta_ph) + log w_ph, shape (persons, nodes)."""
	lk = model.cell_log_k(params, cells, nodes[cells.person])
	return jax.ops.segment_sum(lk, cells.person, num_segments=n_persons) + log_w


def _chunked_joint(
	model: Model,
	params: dict[str, Array],
	person: np.ndarray,
	item: np.ndarray,
	x: np.ndarray,
	n_persons: int,
	nodes: Array,
) -> Array:
	"""log L_p at every node, shape (persons, nodes), summed over fixed-size chunks of cells."""
	# The (cells, nodes) array of 11^3 nodes does not fit; padded cells goto a sentinel person.
	pad = (-x.size) % _SUMMARY_CHUNK
	person = np.concatenate([person, np.full(pad, n_persons)])
	item = np.concatenate([item, np.zeros(pad, dtype=item.dtype)])
	x = np.concatenate([x, np.full(pad, 0.5)])
	padded = jnp.concatenate([nodes, jnp.zeros((1, *nodes.shape[1:]))])

	total = jnp.zeros((n_persons + 1, nodes.shape[1]))
	for start in range(0, x.size, _SUMMARY_CHUNK):
		sl = slice(start, start + _SUMMARY_CHUNK)
		cells = _cells(person[sl], item[sl], x[sl])
		total = total + _joint_chunk(model, params, cells, padded, n_persons)

	return total[:n_persons]


@partial(jax.jit, static_argnames=("model", "n_persons"))
def _joint_chunk(
	model: Model, params: dict[str, Array], cells: Cells, nodes: Array, n_persons: int
) -> Array:
	lk = model.cell_log_k(params, cells, nodes[cells.person])
	return jax.ops.segment_sum(lk, cells.person, num_segments=n_persons + 1)


@partial(jax.jit, static_argnames=("model", "n_persons", "prior_scale"))
def _value_and_grad(
	model: Model,
	params: dict[str, Array],
	cells: Cells,
	n_persons: int,
	nodes: Array,
	log_w: Array,
	prior_scale: float,
) -> tuple[Array, dict[str, Array]]:
	def loss(p: dict[str, Array]) -> Array:
		lm = logsumexp(_joint(model, p, cells, n_persons, nodes, log_w), axis=1).sum()
		return -(lm + model.log_prior(p, prior_scale))

	return jax.value_and_grad(loss)(params)


@partial(jax.jit, static_argnames=("model", "n_persons", "scan"))
def _modes(
	model: Model, params: dict[str, Array], cells: Cells, n_persons: int, start: Array, scan: bool
) -> tuple[Array, Array]:
	def log_post(theta: Array) -> Array:
		lk = model.cell_log_k(params, cells, theta[cells.person])
		prior = norm.logpdf(theta)
		segment = jax.ops.segment_sum(lk, cells.person, num_segments=n_persons)
		return segment + (prior if theta.ndim == 1 else prior.sum(axis=-1))

	return posterior_modes(log_post, start, steps=_NEWTON_STEPS, clip=_NEWTON_CLIP, scan=scan)


def zoi_moment_init(item: np.ndarray, x: np.ndarray, n_items: int) -> dict[str, Array]:
	"""A data-only starting point: each item's interior mean and spread, and endpoint rates."""
	n = np.maximum(np.bincount(item, minlength=n_items), 1).astype(float)
	inside = (x > 0.0) & (x < 1.0)
	n_in = np.maximum(np.bincount(item, inside, n_items), 1.0)
	x_in = np.where(inside, x, 0.0)

	mean = np.clip(np.bincount(item, x_in, n_items) / n_in, 0.02, 0.98)
	var = np.bincount(item, np.where(inside, (x - mean[item]) ** 2, 0.0), n_items) / n_in
	var = np.clip(var, 1e-6, mean * (1.0 - mean) * 0.99)

	b = np.log(mean / (1.0 - mean))
	precision = np.maximum(mean * (1.0 - mean) / var - 1.0, 1.0)
	omega = 2.0 * np.log(precision / (2.0 * np.cosh(b / 2.0)))

	p0 = np.clip(np.bincount(item, x == 0.0, n_items) / n, 1e-4, 0.5)
	p1 = np.clip(np.bincount(item, x == 1.0, n_items) / n, 1e-4, 0.5)
	gamma_0 = np.log(p0 / (1.0 - p0))
	gamma_1 = np.log((1.0 - p1) / p1)

	return zoi_params(np.full(n_items, np.log(0.5)), b, omega, gamma_0, gamma_1)

"""A MAP fit of one split: its model and settings, its data-only start, and its run tables."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from omegaconf import DictConfig

from .data import Split
from .irt.cnrm.em import init_person_mean_ols
from .irt.cnrm.transform import TransformSpec
from .irt.map import MAPConfig, MAPResult, Model, cnrm_params, fit_map, zoi_moment_init, zoi_tau
from .irt.zoi_beta.model import COORDS


def build_model(cfg: DictConfig) -> Model:
	"""The model, rejecting an estimator that does not apply to it."""
	transform = cfg.get("transform")
	spec = None if transform is None else TransformSpec.from_config(transform)
	model = Model(cfg.model.family, spec)

	kind = cfg.estimator.kind
	if kind == "nuts" and model.family != "zoi_beta":
		raise ValueError("nuts samples zoi_beta only")
	if kind == "em" and (spec is None or spec.kind != "linear"):
		raise ValueError("em fits the cnrm with a linear transform only")

	return model


def map_settings(cfg: DictConfig) -> MAPConfig:
	"""The MAP settings; a value left at ??? fails here, before any data is read."""
	e = cfg.estimator
	return MAPConfig(
		n_nodes=e.n_nodes,
		steps=e.steps,
		learning_rate=e.learning_rate,
		final_lr_fraction=e.final_lr_fraction,
		scan_every=e.scan_every,
		prior_scale=cfg.model.prior_scale,
	)


def run_tags(cfg: DictConfig, model: Model, data: Split, response: str) -> dict[str, str]:
	"""What a run records about itself in metrics.json."""
	device = jax.devices()[0]

	tags = {
		"family": model.family,
		"estimator": cfg.estimator.kind,
		"response": response,
		"dataset_hash": data.dataset_hash,
		"device": f"{device.platform}:{device.device_kind}",
	}
	if model.transform is not None:
		tags["transform"] = model.transform.kind

	return tags


def fit_map_split(model: Model, config: MAPConfig, data: Split) -> MAPResult:
	"""MAP from a data-only start: moments for zoi_beta, person-mean OLS for the cnrm."""
	if model.family == "cnrm":
		assert model.transform is not None
		ols = init_person_mean_ols(data.person, data.item, data.x, data.n_persons, data.n_items)
		init = cnrm_params(*ols, model.transform)
	else:
		init = zoi_moment_init(data.item, data.x, data.n_items)

	return fit_map(model, data.person, data.item, data.x, data.n_persons, init, config)


def write_map(out: Path, model: Model, data: Split, result: MAPResult) -> dict[str, float]:
	"""Writes a MAP fit's tables into `out` and returns its convergence metrics."""
	p = result.params

	if model.family == "cnrm":
		assert model.transform is not None
		items = {
			"alpha": np.exp(p["log_alpha"]),
			"beta": p["beta"],
			"sigma": np.exp(p["log_sigma"]),
		}
		_write_transform(out, model.transform, p["transform"])
	else:
		tau = zoi_tau({name: jnp.asarray(value) for name, value in p.items()})
		items = {c: np.asarray(v) for c, v in zip(COORDS, tau, strict=True)}

	pl.DataFrame({"item_idx": np.arange(data.n_items), **items}).write_parquet(
		out / "items.parquet"
	)
	write_persons(out, data, result.eap, result.psd)
	pl.DataFrame({"step": np.arange(len(result.loss)), "loss": result.loss}).write_parquet(
		out / "trace.parquet"
	)

	return {
		"best_loss": result.best_loss,
		"best_step": result.best_step,
		"drop_last": result.drop_last,
		"log_marginal": result.log_marginal,
	}


def write_persons(out: Path, data: Split, eap: np.ndarray, psd: np.ndarray) -> None:
	"""persons.parquet: each person's EAP and posterior sd."""
	pl.DataFrame({"person_idx": data.person_idx, "eap": eap, "psd": psd}).write_parquet(
		out / "persons.parquet"
	)


def _write_transform(out: Path, spec: TransformSpec, raw: np.ndarray) -> None:
	knots = np.asarray(spec.knots)
	record = {
		"kind": spec.kind,
		"knots": knots.tolist(),
		"penalty": spec.penalty,
		"raw": np.atleast_1d(raw).tolist(),
		"exponent_at_knots": np.asarray(
			spec.exponent(jnp.asarray(raw), jnp.asarray(knots))
		).tolist(),
	}
	(out / "transform.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

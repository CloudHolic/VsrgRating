"""Fit one IRT model to each response variable of a dataset and record each fit.

uv run --env-file .env scripts/fit.py +experiment=core300_cnrm_em
uv run --env-file .env scripts/fit.py +experiment=core300_zoi_beta_map tracking=mlflow
"""

import json
import time
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import typer
from omegaconf import DictConfig

from vsrgrating.config import compose_config
from vsrgrating.data import Split, load_split, locate_dataset
from vsrgrating.irt.cnrm.em import EMConfig, fit_em, init_person_mean_ols, person_loglik
from vsrgrating.irt.cnrm.transform import TransformSpec
from vsrgrating.irt.map import MAPConfig, Model, cnrm_params, fit_map, zoi_moment_init, zoi_tau
from vsrgrating.irt.zoi_beta.model import COORDS
from vsrgrating.irt.zoi_beta.nuts import NUTSConfig, fit_nuts, summarize
from vsrgrating.tracking import run_output, write_metrics

type Settings = NUTSConfig | MAPConfig | EMConfig
type Record = tuple[dict[str, float], dict[str, list[float]]]


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("fit", overrides)
	jax.config.update("jax_enable_x64", True)
	model = build_model(cfg)
	settings = build_settings(cfg)
	dataset = locate_dataset(Path(cfg.dataset_root), cfg.dataset.name)

	for response in cfg.responses:
		data = load_split(dataset, response, "train")
		with run_output(cfg, overrides, response) as out:
			metrics, series = fit(cfg, model, settings, data, out)
			write_metrics(out, run_tags(cfg, model, data, response), metrics, series)


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


def build_settings(cfg: DictConfig) -> Settings:
	"""The estimator's settings; a value left at ??? fails here, before any data is read."""
	e = cfg.estimator
	match e.kind:
		case "nuts":
			return NUTSConfig(
				chains=e.chains,
				warmup=e.warmup,
				samples=e.samples,
				target_accept=e.target_accept,
				max_tree_depth=e.max_tree_depth,
				seed=e.seed,
				chain_method=e.chain_method,
			)
		case "map":
			return MAPConfig(
				n_nodes=e.n_nodes,
				steps=e.steps,
				learning_rate=e.learning_rate,
				final_lr_fraction=e.final_lr_fraction,
				scan_every=e.scan_every,
				prior_scale=cfg.model.prior_scale,
			)
		case "em":
			return EMConfig(n_nodes=e.n_nodes, tol=e.tol, max_iter=e.max_iter)

	raise ValueError(f"unknown estimator {e.kind!r}")


def fit(cfg: DictConfig, model: Model, settings: Settings, data: Split, out: Path) -> Record:
	"""Runs the estimator and writes its tables into `out`."""
	match settings:
		case NUTSConfig():
			return run_nuts(settings, data, out, cfg.model.prior_scale, cfg.estimator.thin)
		case MAPConfig():
			return run_map(model, settings, data, out)
		case EMConfig():
			return run_em(settings, data, out, cfg.estimator.ref_nodes)


def run_tags(cfg: DictConfig, model: Model, data: Split, response: str) -> dict[str, str]:
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


def run_nuts(config: NUTSConfig, data: Split, out: Path, scale: float, thin: int) -> Record:
	start = time.perf_counter()
	result = fit_nuts(data.person, data.item, data.x, data.n_persons, data.n_items, scale, config)
	runtime = time.perf_counter() - start
	items = summarize(result.item_draws)
	persons = summarize(result.theta_draws)

	pl.DataFrame(
		{
			"item_idx": np.repeat(np.arange(data.n_items), len(COORDS)),
			"coord": np.tile(COORDS, data.n_items),
			**{name: value.reshape(-1) for name, value in items.items()},
		}
	).write_parquet(out / "items_summary.parquet")
	pl.DataFrame({"person_idx": data.person_idx, **persons}).write_parquet(
		out / "persons_summary.parquet"
	)

	item_draws = result.item_draws[:, ::thin]
	chain, draw, idx, coord = np.indices(item_draws.shape).reshape(4, -1)
	pl.DataFrame(
		{
			"chain": chain,
			"draw": draw * thin,
			"item_idx": idx,
			"coord": np.asarray(COORDS)[coord],
			"value": item_draws.reshape(-1),
		}
	).write_parquet(out / "draws_items.parquet")

	theta_draws = result.theta_draws[:, ::thin]
	chain, draw, p = np.indices(theta_draws.shape).reshape(3, -1)
	pl.DataFrame(
		{
			"chain": chain,
			"draw": draw * thin,
			"person_idx": data.person_idx[p],
			"value": theta_draws.reshape(-1),
		}
	).write_parquet(out / "draws_persons.parquet")

	metrics = {
		"runtime_s": runtime,
		"max_rhat": float(max(items["rhat"].max(), persons["rhat"].max())),
		"min_ess_bulk": float(min(items["ess_bulk"].min(), persons["ess_bulk"].min())),
		"min_ess_tail": float(min(items["ess_tail"].min(), persons["ess_tail"].min())),
		"n_divergent": result.n_divergent,
		"tree_depth_hit_frac": result.tree_depth_hit_frac,
	}
	return metrics, {}


def run_map(model: Model, config: MAPConfig, data: Split, out: Path) -> Record:
	"""MAP from a data-only start: moments for zoi_beta, person-mean OLS for the cnrm."""
	if model.family == "cnrm":
		assert model.transform is not None
		ols = init_person_mean_ols(data.person, data.item, data.x, data.n_persons, data.n_items)
		init = cnrm_params(*ols, model.transform)
	else:
		init = zoi_moment_init(data.item, data.x, data.n_items)

	start = time.perf_counter()
	result = fit_map(model, data.person, data.item, data.x, data.n_persons, init, config)
	runtime = time.perf_counter() - start

	p = result.params
	if model.family == "cnrm":
		assert model.transform is not None
		items = {
			"alpha": np.exp(p["log_alpha"]),
			"beta": p["beta"],
			"sigma": np.exp(p["log_sigma"]),
		}
		write_transform(out, model.transform, p["transform"])
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

	metrics = {
		"runtime_s": runtime,
		"best_loss": result.best_loss,
		"best_step": result.best_step,
		"drop_last": result.drop_last,
		"log_marginal": result.log_marginal,
	}
	return metrics, {"loss": result.loss.tolist()}


def run_em(config: EMConfig, data: Split, out: Path, ref_nodes: int) -> Record:
	"""EM, and each person's log marginal at the fit's nodes against `ref_nodes` nodes."""
	start = time.perf_counter()
	result = fit_em(data.person, data.item, data.x, data.n_persons, data.n_items, config)
	runtime = time.perf_counter() - start

	cells = (data.person, data.item, data.x, data.n_persons)
	loglik = person_loglik(*cells, result.item, config.n_nodes)
	loglik_ref = person_loglik(*cells, result.item, ref_nodes)
	items = {name: np.asarray(value) for name, value in result.item._asdict().items()}
	item_idx = np.arange(data.n_items)

	pl.DataFrame(
		{"item_idx": item_idx, **items, "degenerate": np.isin(item_idx, result.degenerate_items)}
	).write_parquet(out / "items.parquet")
	write_persons(out, data, result.eap, result.psd)
	pl.DataFrame(
		{
			"person_idx": data.person_idx,
			"n_responses": np.bincount(data.person, minlength=data.n_persons),
			"loglik": loglik,
			"loglik_ref": loglik_ref,
			"diff": loglik - loglik_ref,
		}
	).write_parquet(out / "quadrature.parquet")

	sigma_q = np.quantile(items["sigma"], [0.05, 0.25, 0.5])
	metrics = {
		"runtime_s": runtime,
		"n_iter": result.n_iter,
		"converged": float(result.converged),
		"degenerate": float(result.degenerate),
		"n_degenerate_items": result.degenerate_items.size,
		"final_loglik": float(loglik.sum()),
		"sigma_min": float(items["sigma"].min()),
		"sigma_q05": float(sigma_q[0]),
		"sigma_q25": float(sigma_q[1]),
		"sigma_q50": float(sigma_q[2]),
	}
	return metrics, {"loglik": result.loglik.tolist()}


def write_persons(out: Path, data: Split, eap: np.ndarray, psd: np.ndarray) -> None:
	pl.DataFrame({"person_idx": data.person_idx, "eap": eap, "psd": psd}).write_parquet(
		out / "persons.parquet"
	)


def write_transform(out: Path, spec: TransformSpec, raw: np.ndarray) -> None:
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


if __name__ == "__main__":
	typer.run(main)

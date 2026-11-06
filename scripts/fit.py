"""Fit one IRT model to each response variable of a dataset and record each fit.

uv run --env-file .env scripts/fit.py +experiment=core300_cnrm_em
uv run --env-file .env scripts/fit.py +experiment=core300_zoi_beta_map tracking=mlflow
"""

import time
from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import polars as pl
import typer
from omegaconf import DictConfig

from vsrgrating.config import compose_config
from vsrgrating.data import Split, load_split, locate_dataset
from vsrgrating.fitting import (
	build_model,
	fit_map_split,
	map_settings,
	run_tags,
	write_map,
	write_persons,
)
from vsrgrating.irt.cnrm.em import EMConfig, fit_em, person_loglik
from vsrgrating.irt.map import MAPConfig, Model
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
			return map_settings(cfg)
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
	start = time.perf_counter()
	result = fit_map_split(model, config, data)
	runtime = time.perf_counter() - start

	return {"runtime_s": runtime, **write_map(out, model, data, result)}, {
		"loss": result.loss.tolist()
	}


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


if __name__ == "__main__":
	typer.run(main)

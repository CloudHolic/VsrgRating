"""Fixed person posterior: a CNRM MAP fit on the train k-core T from several starts, the lowest
kept, and every person's posterior N(mu, Sigma) on T at the held-out node count.

uv run --env-file .env scripts/fit_posterior.py +experiment=keys4to8_posterior_d3
uv run --env-file .env scripts/fit_posterior.py +experiment=keys4to8_posterior_d3_no6k
"""

import json
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import typer
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

from vsrgrating.config import compose_config
from vsrgrating.data import PosteriorCells, locate_dataset, posterior_cells
from vsrgrating.evaluation import Fitted
from vsrgrating.evaluation.heldout import HELDOUT_NODES
from vsrgrating.fitting import Refits, build_model, map_settings, refit, run_tags, write_transform
from vsrgrating.irt.cnrm import model as cnrm
from vsrgrating.irt.map import Model, person_summary
from vsrgrating.readout.cnrm import difficulty
from vsrgrating.tracking import METRICS, run_output, write_metrics

RESPONSE = "acc"
# Below this gap the 2 lowest starts count as one solution; D = 3 local optima were 4,000-5,000
# nat apart and a re-found solution 45 nat.
SAME_SOLUTION_NAT = 100.0
T_KEYS = ("min_item_responses", "min_person_responses", "drop_keys")
READOUT_Y = 0.95

type Start = tuple[Fitted, dict[str, float]]


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("posterior", overrides)
	jax.config.update("jax_enable_x64", True)
	check_config(cfg)

	model = build_model(cfg)
	pc = load_cells(cfg)
	report_size(pc)

	starts = {s: fit_start(cfg, overrides, model, pc, s) for s in seeds(cfg)}
	chosen = choose_start({s: metrics["best_loss"] for s, (_, metrics) in starts.items()})
	fitted = starts[chosen][0]
	nodes = HELDOUT_NODES[model.dims]

	with run_output(cfg, overrides, RESPONSE) as out:
		t = pc.t
		_, mean, cov = person_summary(
			model, item_params(fitted), t.person, t.item, t.x, t.n_persons, nodes
		)
		metrics = write_posterior(out, model, fitted, pc, np.asarray(mean), np.asarray(cov))
		metrics |= compare_starts(starts, pc) | {"posterior_nodes": nodes}
		tags = {
			**run_tags(cfg, model, t, RESPONSE),
			"split_hash": pc.split_hash,
			"split_chart_index": pc.split_chart_index,
			"chosen_seed": str(chosen),
		}
		write_metrics(out, tags, metrics, {})


def check_config(cfg: DictConfig) -> None:
	"""What can be rejected before any data is read."""
	if cfg.tracking.mlflow:
		raise ValueError("the starts are read back from disk: use tracking=local")
	if cfg.model.family != "cnrm" or cfg.estimator.kind != "map":
		raise ValueError("the posterior is a cnrm MAP fit")

	given = list(cfg.posterior.init_seeds)
	if len(given) < 2 or len(set(given)) != len(given):
		raise ValueError(f"posterior.init_seeds needs 2 or more distinct seeds, got {given}")

	out = Path(cfg.runs_root) / cfg.name / RESPONSE
	if out.exists():
		raise FileExistsError(f"{out} already holds a run")


def seeds(cfg: DictConfig) -> list[int]:
	return [int(s) for s in cfg.posterior.init_seeds]


def load_cells(cfg: DictConfig) -> PosteriorCells:
	dataset = locate_dataset(Path(cfg.dataset_root), cfg.dataset.name)
	split_dir = Path(cfg.split_root) / cfg.dataset.name / dataset.name
	p = cfg.posterior

	return posterior_cells(
		dataset,
		split_dir,
		RESPONSE,
		p.min_item_responses,
		p.min_person_responses,
		[int(k) for k in p.drop_keys],
	)


def report_size(pc: PosteriorCells) -> None:
	t = pc.frame.filter(pl.col("in_t"))
	by_key = (
		t.group_by("keys")
		.agg(pl.col("item_idx").n_unique().alias("items"), pl.len().alias("cells"))
		.sort("keys")
	)
	print(f"T: {pc.t.n_persons} of {pc.n_persons} persons, {pc.t.x.size} cells", flush=True)
	for keys, items, cells in by_key.iter_rows():
		print(f"  {keys}K: {items} items, {cells} cells", flush=True)


def fit_start(
	cfg: DictConfig, overrides: list[str], model: Model, pc: PosteriorCells, seed: int
) -> Start:
	"""One start as its own run `acc-start<seed>`, read back if it finished under the same T."""
	start_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
	assert isinstance(start_cfg, DictConfig)
	start_cfg.estimator.init_seed = seed

	label = f"start{seed}"
	path = Path(cfg.runs_root) / cfg.name / f"{RESPONSE}-{label}"
	if (path / METRICS).exists():
		check_resumable(cfg, path, pc)

	refits = Refits(start_cfg, overrides, model, map_settings(start_cfg))
	extra = {"split_hash": pc.split_hash, "split_chart_index": pc.split_chart_index}
	fitted = refit(refits, RESPONSE, label, pc.t, extra)
	record = json.loads((path / METRICS).read_text(encoding="utf-8"))

	return fitted, record["metrics"]


def check_resumable(cfg: DictConfig, path: Path, pc: PosteriorCells) -> None:
	"""A finished start counts only if it saw the same split and the same T rules."""
	other = OmegaConf.load(path / "config.yaml")
	assert isinstance(other, DictConfig)
	mine = OmegaConf.to_container(cfg.posterior)
	theirs = OmegaConf.to_container(other.posterior)
	assert isinstance(mine, dict) and isinstance(theirs, dict)

	for key in T_KEYS:
		if mine[key] != theirs[key]:
			raise ValueError(
				f"{path} was fitted with posterior.{key}={theirs[key]}, not {mine[key]}"
			)

	tags = json.loads((path / METRICS).read_text(encoding="utf-8"))["tags"]
	if tags.get("split_hash") != pc.split_hash:
		raise ValueError(
			f"{path} was fitted on split {tags.get('split_hash')}, not {pc.split_hash}"
		)


def choose_start(best_losses: dict[int, float]) -> int:
	"""The seed of the lowest best_loss, once a second start has come within SAME_SOLUTION_NAT."""
	if len(best_losses) < 2:
		raise ValueError("choosing a start needs 2 or more starts")

	ranked = sorted(best_losses, key=best_losses.__getitem__)
	gap = best_losses[ranked[1]] - best_losses[ranked[0]]
	if gap > SAME_SOLUTION_NAT:
		listed = ", ".join(f"{s}: {best_losses[s]:.1f}" for s in ranked)
		raise RuntimeError(
			f"the lowest start is {gap:.1f} nat below the next ({listed}); "
			"add a start to posterior.init_seeds"
		)

	return ranked[0]


def item_params(fitted: Fitted) -> dict[str, jax.Array]:
	item = fitted.item
	assert isinstance(item, cnrm.ItemParams) and fitted.raw is not None

	return {
		"alpha": item.alpha,
		"beta": item.beta,
		"log_sigma": jnp.log(item.sigma),
		"transform": fitted.raw,
	}


def write_posterior(
	out: Path, model: Model, fitted: Fitted, pc: PosteriorCells, mean: np.ndarray, cov: np.ndarray
) -> dict[str, float]:
	"""items (T only), transform, persons (all, prior N(0, I) outside T) and cells with G."""
	item = fitted.item
	assert isinstance(item, cnrm.ItemParams) and fitted.raw is not None
	assert model.transform is not None
	t, dims = pc.t, model.dims

	t_items = np.unique(t.item)
	alpha = np.asarray(item.alpha)[t_items]
	sigma = np.asarray(item.sigma)[t_items]
	pl.DataFrame(
		{
			"item_idx": t_items,
			**{f"alpha_{k + 1}": alpha[:, k] for k in range(dims)},
			"beta": np.asarray(item.beta)[t_items],
			"sigma": sigma,
			"n_t": np.bincount(t.item, minlength=t.n_items)[t_items],
		}
	).write_parquet(out / "items.parquet")
	write_transform(out, model.transform, np.asarray(fitted.raw))

	in_t = np.zeros(pc.n_persons, dtype=bool)
	in_t[t.person_idx] = True
	mu = np.zeros((pc.n_persons, dims))
	mu[t.person_idx] = mean
	sigma_p = np.tile(np.eye(dims), (pc.n_persons, 1, 1))
	sigma_p[t.person_idx] = cov
	upper = [(i, j) for i in range(dims) for j in range(i, dims)]
	pl.DataFrame(
		{
			"person_idx": np.arange(pc.n_persons),
			"in_t": in_t,
			"n_t": np.bincount(t.person_idx[t.person], minlength=pc.n_persons),
			**{f"mu_{k + 1}": mu[:, k] for k in range(dims)},
			**{f"cov_{i + 1}{j + 1}": sigma_p[:, i, j] for i, j in upper},
		}
	).write_parquet(out / "persons.parquet")

	x = pc.frame["x"].to_numpy()
	g, log_dg = model.transform.apply(
		fitted.raw, jnp.asarray(np.where((x == 0.0) | (x == 1.0), 0.5, x))
	)
	pc.frame.with_columns(
		pl.Series("g", np.asarray(g)), pl.Series("log_dg", np.asarray(log_dg))
	).write_parquet(out / "cells.parquet")

	keys = item_keys(pc)
	metrics: dict[str, float] = {
		"n_t_persons": int(in_t.sum()),
		"n_prior_persons_total": int((~in_t).sum()),
		"sigma_min": float(sigma.min()),
		"sigma_q05": float(np.quantile(sigma, 0.05)),
	}
	for part in ("train", "dev", "valid"):
		persons = pc.frame.filter(pl.col("part") == part)["person_idx"].unique().to_numpy()
		metrics[f"n_prior_persons_{part}"] = int((~in_t[persons]).sum())
	for k in np.unique(keys):
		metrics[f"n_t_items.{k}K"] = int((keys == k).sum())

	return metrics


def compare_starts(starts: dict[int, Start], pc: PosteriorCells) -> dict[str, float]:
	"""Every start's best_loss and drop_last, and how the 2 lowest agree on d per key mode."""
	metrics: dict[str, float] = {}
	for s, (_, m) in starts.items():
		metrics[f"best_loss.start{s}"] = m["best_loss"]
		metrics[f"drop_last.start{s}"] = m["drop_last"]

	first, second = sorted(starts, key=lambda s: starts[s][1]["best_loss"])[:2]
	metrics["best_loss_gap"] = starts[second][1]["best_loss"] - starts[first][1]["best_loss"]

	t_items = np.unique(pc.t.item)
	keys = item_keys(pc)
	d = [readout(starts[s][0])[t_items] for s in (first, second)]
	for k in np.unique(keys):
		m = keys == k
		metrics[f"d_spearman_top2.{k}K"] = float(spearmanr(d[0][m], d[1][m]).statistic)

	return metrics


def item_keys(pc: PosteriorCells) -> np.ndarray:
	"""The key mode of every T item, in item_idx order."""
	t = pc.frame.filter(pl.col("in_t")).unique("item_idx").sort("item_idx")
	return t["keys"].to_numpy()


def readout(fitted: Fitted) -> np.ndarray:
	item = fitted.item
	assert isinstance(item, cnrm.ItemParams) and fitted.raw is not None
	assert fitted.model.transform is not None

	return difficulty(item, fitted.model.transform, fitted.raw, READOUT_Y)


if __name__ == "__main__":
	typer.run(main)

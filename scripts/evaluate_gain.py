"""Score item parameters for one part by each person's predictive log likelihood under the fixed
person posterior, log p(Y_T, Y_V) - log p(Y_T), overall and by subset.

uv run --env-file .env scripts/evaluate_gain.py name=keys4to8_gain_floor part=dev \
	item_params=runs/keys4to8_reference/floor-dev/items_dev.parquet
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
from omegaconf import DictConfig, OmegaConf
from scipy.special import log_ndtr

from vsrgrating.config import compose_config
from vsrgrating.data import Split
from vsrgrating.evaluation import (
	HELDOUT_NODES,
	Fitted,
	jackknife_totals,
	mapset_blocks,
	posterior_fitted,
	posterior_splits,
	predictive_gain,
)
from vsrgrating.irt.cnrm import model as cnrm
from vsrgrating.readout.cnrm import difficulty
from vsrgrating.tracking import run_output, write_metrics

PARTS = ("dev", "valid")
JACKKNIFE_NODES = 3
JACKKNIFE_BLOCKS = 30
# Items under this sigma are where the fit collapsed (experiment 6); their scores are also given
# without them.
COLLAPSED_SIGMA = 0.002
RESPONSE_BANDS = ((10, 19), (20, 99), (100, None))
EQUAL_MEAN_KEYS = (4, 5, 6, 7)
DIFFICULTY_Y = 0.95
EXTREMES = 5


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("evaluate_gain", overrides)
	if cfg.part not in PARTS:
		raise ValueError(f"part must be one of {PARTS}, got {cfg.part!r}")

	jax.config.update("jax_enable_x64", True)

	run = Path(cfg.posterior_run)
	cells, n_persons, tags = read_posterior(run)
	psi = pl.read_parquet(cfg.item_params).sort("item_idx")
	check_items(cells, psi, cfg.part, drop_keys(run))

	base, valid = posterior_splits(
		cells, psi["item_idx"].to_numpy(), n_persons, tags["dataset_hash"]
	)
	fitted = posterior_fitted(run, psi, base.n_items)
	held = cells.filter(pl.col("item_idx").is_in(psi["item_idx"].implode()))
	device = jax.devices()[0]

	tags |= {
		"part": str(cfg.part),
		"item_params": str(cfg.item_params),
		"device": f"{device.platform}:{device.device_kind}",
	}

	with run_output(cfg, overrides, str(cfg.part)) as out:
		start = time.perf_counter()
		items = item_scores(run, cells, held, psi, fitted)
		items.write_parquet(out / "items.parquet")
		metrics = evaluate(out, "", fitted, base, valid, held, psi)
		collapsed = psi.filter(pl.col("sigma") < COLLAPSED_SIGMA)["item_idx"]

		if collapsed.len():
			kept = ~held["item_idx"].is_in(collapsed.implode()).to_numpy()
			filtered = evaluate(
				out, "_filtered", fitted, base, _cells(valid, kept), held.filter(kept), psi
			)
			metrics |= {f"filtered.{k}": v for k, v in filtered.items()}

		ranked = items.sort("contribution", descending=True)

		metrics |= {
			"collapsed_items": float(collapsed.len()),
			"top_contribution": float(ranked["contribution"].head(EXTREMES).sum()),
			"bottom_contribution": float(ranked["contribution"].tail(EXTREMES).sum()),
			"runtime_s": time.perf_counter() - start,
		}
		tags |= {
			"top_items": _listing(ranked.head(EXTREMES)),
			"bottom_items": _listing(ranked.tail(EXTREMES)),
		}
		write_metrics(out, tags, metrics, {})


def read_posterior(run: Path) -> tuple[pl.DataFrame, int, dict[str, str]]:
	"""The run's cells, how many dataset persons its posterior covers, and its tags."""
	cells = pl.read_parquet(run / "cells.parquet")
	n_persons = pl.read_parquet(run / "persons.parquet", columns=["person_idx"]).height
	record = json.loads((run / "metrics.json").read_text(encoding="utf-8"))

	return cells, n_persons, record["tags"]


def drop_keys(run: Path) -> list[int]:
	cfg = OmegaConf.load(run / "config.yaml")
	assert isinstance(cfg, DictConfig)

	return [int(k) for k in cfg.posterior.drop_keys]


def check_items(cells: pl.DataFrame, psi: pl.DataFrame, part: str, dropped: list[int]) -> None:
	"""psi must hold exactly the part's items: on dev without the posterior's dropped key modes,
	on valid all of them."""
	items = cells.filter(pl.col("part") == part).select("item_idx", "keys").unique()

	if part == "dev":
		items = items.filter(~pl.col("keys").is_in(dropped))

	expected = set(items["item_idx"].to_list())
	got = set(psi["item_idx"].to_list())

	if got != expected or psi.height != len(got):
		raise ValueError(
			f"{part} has {len(expected)} items; the item file gives {psi.height} rows, "
			f"{len(got - expected)} outside it and {len(expected - got)} missing"
		)


def evaluate(
	out: Path,
	suffix: str,
	fitted: Fitted,
	base: Split,
	valid: Split,
	held: pl.DataFrame,
	psi: pl.DataFrame,
) -> dict[str, float]:
	"""table, persons and jackknife parquet files (with `suffix`) and the table's main values."""
	item = held["item_idx"].to_numpy()
	by_item = np.zeros(item.max() + 1)
	by_item[psi["item_idx"].to_numpy()] = psi["sigma"].to_numpy()

	sigma = by_item[item]
	n = np.bincount(item)[item]
	keys = held["keys"].to_numpy()

	masks = {"all": np.ones(held.height, dtype=bool)}
	masks |= {f"{k}K": keys == k for k in np.unique(keys)}
	masks["5K+6K"] = (keys == 5) | (keys == 6)
	stored = set(masks)

	for low, high in RESPONSE_BANDS:
		masks[f"{low}+" if high is None else f"{low}-{high}"] = (n >= low) & (
			n <= (high if high is not None else np.iinfo(n.dtype).max)
		)

	rows, persons, jackknife = [], [], []
	for name, mask in masks.items():
		if not mask.any():
			continue

		subset = _cells(valid, mask)
		gain = predictive_gain(fitted, base, subset, HELDOUT_NODES[fitted.model.dims])
		coarse = predictive_gain(fitted, base, subset, JACKKNIFE_NODES)
		total = float(gain.sum())

		rows.append(
			{
				"subset": name,
				"n_items": int(np.unique(subset.item).size),
				"n_cells": int(mask.sum()),
				"n_persons": int(np.unique(subset.person).size),
				"total": total,
				"per_cell": total / mask.sum(),
				"total_3": float(coarse.sum()),
				"node_gap": total - float(coarse.sum()),
				"sigma_min": float(sigma[mask].min()),
			}
		)

		if name not in stored:
			continue

		who = np.unique(subset.person)
		persons.append(pl.DataFrame({"subset": name, "person_idx": who, "gain": gain[who]}))
		sets = held["beatmapset_id"].to_numpy()[mask]

		n_blocks = min(JACKKNIFE_BLOCKS, np.unique(sets).size)
		block = mapset_blocks(sets, keys[mask], n_blocks)
		totals = jackknife_totals(fitted, base, subset, block, n_blocks, JACKKNIFE_NODES)
		jackknife.append(
			pl.DataFrame({"subset": name, "block": np.arange(n_blocks), "total": totals})
		)

	table = pl.DataFrame(rows)
	equal = table.filter(pl.col("subset").is_in([f"{k}K" for k in EQUAL_MEAN_KEYS]))
	if equal.height == len(EQUAL_MEAN_KEYS):
		table = pl.concat(
			[
				table,
				pl.DataFrame({"subset": ["4-7K mean"], "per_cell": [equal["per_cell"].mean()]}),
			],
			how="diagonal",
		)

	table.write_parquet(out / f"table{suffix}.parquet")
	pl.concat(persons).write_parquet(out / f"persons{suffix}.parquet")
	pl.concat(jackknife).write_parquet(out / f"jackknife{suffix}.parquet")

	values = {}
	for row in table.iter_rows(named=True):
		values[f"per_cell.{row['subset']}"] = float(row["per_cell"])

		if row["total"] is not None:
			values[f"total.{row['subset']}"] = float(row["total"])
			values[f"node_gap.{row['subset']}"] = float(row["node_gap"])

	return values


def item_scores(
	run: Path, cells: pl.DataFrame, held: pl.DataFrame, psi: pl.DataFrame, fitted: Fitted
) -> pl.DataFrame:
	"""Per item: its cells' summed log E_q[k] under psi and under the floor (its key mode's median
	T item, or every T item's where the key mode has none), their difference, sigma and d."""
	persons = pl.read_parquet(run / "persons.parquet").sort("person_idx")
	dims = fitted.model.dims
	mu = persons.select(f"mu_{k + 1}" for k in range(dims)).to_numpy()
	cov = np.empty((persons.height, dims, dims))

	for i in range(dims):
		for j in range(i, dims):
			cov[:, i, j] = cov[:, j, i] = persons[f"cov_{i + 1}{j + 1}"].to_numpy()

	alphas = [f"alpha_{k + 1}" for k in range(dims)]
	psi_t = pl.read_parquet(run / "items.parquet").join(
		cells.select("item_idx", "keys").unique(), on="item_idx"
	)
	medians = psi_t.group_by("keys").agg(pl.col(*alphas, "beta", "sigma").median())
	overall = psi_t.select(pl.col(*alphas, "beta", "sigma").median())
	keys = held.select("item_idx", "keys").unique()
	floor = keys.join(medians, on="keys", how="left").with_columns(
		pl.col(c).fill_null(overall[c][0]) for c in (*alphas, "beta", "sigma")
	)

	scores = {}
	for name, params in (("loglik", psi), ("floor_loglik", floor)):
		frame = held.join(params.select("item_idx", *alphas, "beta", "sigma"), on="item_idx")
		value = log_expected_k(
			frame["x"].to_numpy(),
			frame["g"].to_numpy(),
			frame["log_dg"].to_numpy(),
			frame.select(alphas).to_numpy(),
			frame["beta"].to_numpy(),
			frame["sigma"].to_numpy(),
			mu[frame["person_idx"].to_numpy()],
			cov[frame["person_idx"].to_numpy()],
		)
		scores[name] = frame.select("item_idx").with_columns(value=value).group_by("item_idx").sum()

	item = cnrm.ItemParams(
		jnp.asarray(psi.select(alphas).to_numpy() if dims > 1 else psi["alpha_1"].to_numpy()),
		jnp.asarray(psi["beta"].to_numpy()),
		jnp.asarray(psi["sigma"].to_numpy()),
	)

	assert fitted.model.transform is not None and fitted.raw is not None
	d = difficulty(item, fitted.model.transform, fitted.raw, DIFFICULTY_Y)

	return (
		psi.select("item_idx", "sigma")
		.with_columns(d=d)
		.join(keys, on="item_idx")
		.join(held.group_by("item_idx").len("n_cells"), on="item_idx")
		.join(scores["loglik"].rename({"value": "loglik"}), on="item_idx")
		.join(scores["floor_loglik"].rename({"value": "floor_loglik"}), on="item_idx")
		.with_columns(contribution=pl.col("loglik") - pl.col("floor_loglik"))
		.sort("item_idx")
	)


def log_expected_k(
	x: np.ndarray,
	g: np.ndarray,
	log_dg: np.ndarray,
	alpha: np.ndarray,
	beta: np.ndarray,
	sigma: np.ndarray,
	mu: np.ndarray,
	cov: np.ndarray,
) -> np.ndarray:
	"""log E_q[k] per cell under q = N(mu, cov), as the training objective computes it."""
	m = (alpha * mu).sum(-1) + beta
	v = np.clip(np.einsum("ni,nij,nj->n", alpha, cov, alpha), 0.0, None)

	scale = np.sqrt(sigma**2 + v)
	interior = (x != 0.0) & (x != 1.0)
	inside = -0.5 * np.log(2 * np.pi) - np.log(scale) + log_dg - (g - m) ** 2 / (2 * scale**2)
	edge = log_ndtr(np.where(x == 1.0, (m - 1.0) / scale, -m / scale))

	return np.where(interior, inside, edge)


def _cells(split: Split, mask: np.ndarray) -> Split:
	return Split(
		person=split.person[mask],
		item=split.item[mask],
		x=split.x[mask],
		n_persons=split.n_persons,
		n_items=split.n_items,
		person_idx=split.person_idx,
		dataset_hash=split.dataset_hash,
	)


def _listing(rows: pl.DataFrame) -> str:
	return ", ".join(f"{i}:{c:.2f}" for i, c in rows.select("item_idx", "contribution").iter_rows())


if __name__ == "__main__":
	typer.run(main)

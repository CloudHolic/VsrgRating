"""Validity criteria Q1-Q5 for one model's readouts, each response in turn.

uv run scripts/criteria.py +experiment=random3000_criteria_zoi_beta
"""

import json
from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import polars as pl
import typer
from omegaconf import DictConfig

from vsrgrating.config import compose_config
from vsrgrating.criteria import (
	dan,
	invariance,
	mapset,
	player_ordering,
	readouts,
	reproducibility,
)
from vsrgrating.data import Split, load_split, locate_dataset, subset
from vsrgrating.evaluation import Fitted, load_run, locate_run
from vsrgrating.fitting import Refits, build_model, check_same, map_settings, refit, run_tags
from vsrgrating.irt.cnrm import model as cnrm
from vsrgrating.readout.grids import GRIDS
from vsrgrating.tracking import run_output, write_metrics

CHECKS = ("q1", "q2", "q3", "q4", "q5")


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("criteria", overrides)
	jax.config.update("jax_enable_x64", True)

	if cfg.tracking.mlflow:
		raise ValueError("criteria read their refits back from disk: use tracking=local")

	if unknown := set(cfg.checks) - set(CHECKS):
		raise ValueError(f"unknown checks {sorted(unknown)}; pick from {CHECKS}")

	refits = Refits(cfg, overrides, build_model(cfg), map_settings(cfg))
	dataset = locate_dataset(Path(cfg.dataset_root), cfg.dataset.name)
	pending = [r for r in cfg.responses if not (Path(cfg.runs_root) / cfg.name / r).exists()]
	bases = {response: locate_base(cfg, response, dataset.name) for response in pending}

	manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
	anchors = manifest["anchors"]
	items = pl.read_parquet(dataset / "items.parquet")
	ordered = mapset.pairs(anchors["mapset"]["units"], items)

	if "q5" in cfg.checks:
		chart_set, players = player_ordering.common_set(load_split(dataset, "acc", "train"))

	for response, base in bases.items():
		data = load_split(dataset, response, "train")
		grid = GRIDS[response]

		best_fit = load_run("base", base)
		values, domain = readouts.of(best_fit, grid)

		tables, counts = {}, {}

		if "q3" in cfg.checks:
			tables["q3"] = mapset.table(values, domain, ordered)
		if "q4" in cfg.checks:
			tables["q4"] = dan.table(anchors["dan"], items, values, domain)
		if "q1" in cfg.checks:
			tables["q1"] = reproducibility_table(refits, response, data, grid)
		if "q2" in cfg.checks:
			theta = base_theta(base, data, best_fit)
			tables["q2"] = invariance_table(refits, response, data, theta, grid)
		if "q5" in cfg.checks:
			rest = player_ordering.left_out(data, players)
			fitted = refit(refits, response, "q5", rest)
			tables["q5"], tables["q5_bands"], counts = player_ordering.evaluate(
				fitted, data, rest, chart_set, players, grid
			)

		with run_output(cfg, overrides, response) as out:
			for name, table in tables.items():
				table.write_csv(out / f"{name}.csv")

			tags = {**run_tags(cfg, refits.model, data, response), "base": cfg.base}
			write_metrics(out, tags, {**counts, "items_in_domain": float(domain.sum())}, {})


def locate_base(cfg: DictConfig, response: str, dataset_hash: str) -> Path:
	"""The confirmed fit, checked to be comparable, before any refit starts."""
	path = locate_run(Path(cfg.runs_root), cfg.base, response, dataset_hash)
	check_same(cfg, path)

	return path


def reproducibility_table(
	refits: Refits, response: str, data: Split, grid: tuple[float, ...]
) -> pl.DataFrame:
	"""Q1: both halves of each split of the people, fitted apart."""
	tables = []

	for index in range(refits.cfg.q1.repeats):
		seed = refits.cfg.q1.seed + index
		first, second = reproducibility.halves(data.n_persons, seed)
		left_data = subset(data, first[data.person])
		right_data = subset(data, second[data.person])
		left = refit(refits, response, f"q1-{seed}-a", left_data)
		right = refit(refits, response, f"q1-{seed}-b", right_data)

		seen = readouts.seen(left_data) & readouts.seen(right_data)
		tables.append(reproducibility.table(seed, left, right, seen, grid))

	return pl.concat(tables)


def invariance_table(
	refits: Refits, response: str, data: Split, theta: np.ndarray, grid: tuple[float, ...]
) -> pl.DataFrame:
	"""Q2: each split's two groups inside one fit. The random split is Q1's first."""
	tables = []

	for split in invariance.SPLITS:
		group = invariance.groups(split, data, theta, refits.cfg.q1.seed)
		paired = invariance.paired(data, group)
		fitted = refit(refits, response, f"q2-{split}", paired)
		tables.append(invariance.compare(split, fitted, paired, grid))

	return pl.concat(tables)


def base_theta(path: Path, data: Split, fitted: Fitted) -> np.ndarray:
	"""The confirmed fit's EAP for each person of `data`."""
	persons = pl.read_parquet(path / "persons.parquet").sort("person_idx")

	if not np.array_equal(persons["person_idx"].to_numpy(), data.person_idx):
		raise ValueError(f"{path} has other persons than the dataset")

	columns = [c for c in persons.columns if c == "eap" or c.startswith("eap_")]
	theta = persons.select(columns).to_numpy()
	if theta.shape[1] == 1:
		return theta[:, 0]

	item = fitted.item
	assert isinstance(item, cnrm.ItemParams)

	direction = np.asarray(item.alpha).mean(axis=0)
	return theta @ (direction / np.linalg.norm(direction))


if __name__ == "__main__":
	typer.run(main)

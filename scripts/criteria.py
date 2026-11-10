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
from vsrgrating.evaluation import load_run, locate_run
from vsrgrating.fitting import Refits, build_model, check_same, map_settings, refit, run_tags
from vsrgrating.readout.grids import GRIDS
from vsrgrating.tracking import run_output, write_metrics


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("criteria", overrides)
	jax.config.update("jax_enable_x64", True)

	if cfg.tracking.mlflow:
		raise ValueError("criteria read their refits back from disk: use tracking=local")

	refits = Refits(cfg, overrides, build_model(cfg), map_settings(cfg))
	dataset = locate_dataset(Path(cfg.dataset_root), cfg.dataset.name)
	pending = [r for r in cfg.responses if not (Path(cfg.runs_root) / cfg.name / r).exists()]
	bases = {response: locate_base(cfg, response, dataset.name) for response in pending}

	manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
	anchors = manifest["anchors"]
	items = pl.read_parquet(dataset / "items.parquet")
	ordered = mapset.pairs(anchors["mapset"]["units"], items)
	chart_set, players = player_ordering.common_set(load_split(dataset, "acc", "train"))

	for response, base in bases.items():
		data = load_split(dataset, response, "train")
		grid = GRIDS[response]
		values, domain = readouts.of(load_run("base", base), grid)

		q3 = mapset.table(values, domain, ordered)
		q4 = dan.table(anchors["dan"], items, values, domain)
		q1 = reproducibility_table(refits, response, data, grid)
		q2 = invariance_table(refits, response, data, base_theta(base, data), grid)

		rest = player_ordering.left_out(data, players)
		fitted = refit(refits, response, "q5", rest)
		q5, q5_bands, counts = player_ordering.evaluate(
			fitted, data, rest, chart_set, players, grid
		)

		with run_output(cfg, overrides, response) as out:
			tables = {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q5_bands": q5_bands}
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


def base_theta(path: Path, data: Split) -> np.ndarray:
	"""The confirmed fit's EAP for each person of `data`."""
	persons = pl.read_parquet(path / "persons.parquet").sort("person_idx")

	if not np.array_equal(persons["person_idx"].to_numpy(), data.person_idx):
		raise ValueError(f"{path} has other persons than the dataset")

	return persons["eap"].to_numpy()


if __name__ == "__main__":
	typer.run(main)

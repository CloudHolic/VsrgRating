"""What training reads: the fixed person posterior, the items of chosen parts with their charts,
and each chart as it was played."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from urc_converter import Chart, parse, scale_rate

from ...data import RATE_GROUP_RATES, split_hash
from ..inputs import count_split_chords

# Parts whose items a posterior with `drop_keys` leaves out; valid keeps them for transfer.
_DROPPABLE = ("train", "dev")


@dataclass(frozen=True)
class Posterior:
	"""A fixed person posterior run."""

	cells: pl.DataFrame
	mu: np.ndarray
	cov: np.ndarray
	psi_t: pl.DataFrame
	tags: dict[str, str]
	dataset_name: str
	drop_keys: tuple[int, ...]


@dataclass(frozen=True)
class ItemCharts:
	"""Items of the chosen parts and the chart index snapshot their charts come from."""

	table: pl.DataFrame
	chart_index: str


def load_posterior(run: Path) -> Posterior:
	"""The fixed posterior."""
	persons = pl.read_parquet(run / "persons.parquet").sort("person_idx")
	if persons["person_idx"].to_list() != list(range(persons.height)):
		raise ValueError(f"{run}: persons.parquet does not hold every person in order")

	dims = sum(name.startswith("mu_") for name in persons.columns)
	cov = np.empty((persons.height, dims, dims))
	for i in range(dims):
		for j in range(i, dims):
			cov[:, i, j] = cov[:, j, i] = persons[f"cov_{i + 1}{j + 1}"].to_numpy()

	config = OmegaConf.load(run / "config.yaml")
	record = json.loads((run / "metrics.json").read_text(encoding="utf-8"))

	return Posterior(
		cells=pl.read_parquet(run / "cells.parquet"),
		mu=persons.select(f"mu_{k + 1}" for k in range(dims)).to_numpy(),
		cov=cov,
		psi_t=pl.read_parquet(run / "items.parquet"),
		tags=record["tags"],
		dataset_name=str(config.dataset.name),
		drop_keys=tuple(int(k) for k in config.posterior.drop_keys),
	)


def load_items(
	posterior: Posterior, split_dir: Path, charts_root: Path, parts: Sequence[str]
) -> ItemCharts:
	"""The items of `parts` and their chart files."""
	split = pl.read_parquet(split_dir / "mapset_split.parquet")
	if split_hash(split) != posterior.tags["split_hash"]:
		raise ValueError(f"{split_dir} is not the split the posterior was fitted on")
	manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
	if manifest["dataset_hash"] != posterior.tags["dataset_hash"]:
		raise ValueError(f"{split_dir} splits {manifest['dataset_hash']}, not the posterior's")

	index = (
		pl.read_parquet(charts_root / "index" / manifest["chart_index"] / "index.parquet")
		.filter((pl.col("chart_source") == "osu_archive") & (pl.col("convert_status") == "ok"))
		.select("beatmap_id", "urc_path")
	)
	dropped = pl.col("part").is_in(_values(_DROPPABLE)) & pl.col("keys").is_in(
		_values(posterior.drop_keys, pl.Int64)
	)
	table = (
		split.filter(pl.col("part").is_in(_values(parts)) & ~dropped)
		.join(index, on="beatmap_id", how="left")
		.select("item_idx", "beatmap_id", "rate_group", "keys", "beatmapset_id", "part", "urc_path")
		.sort("item_idx")
	)
	if missing := table["urc_path"].null_count():
		raise ValueError(f"{missing} items have no converted chart in {manifest['chart_index']}")

	return ItemCharts(table, manifest["chart_index"])


def load_chart(charts_root: Path, row: Mapping) -> tuple[Chart, int]:
	"""The chart of one item as played, and how many of its chords the rate change split."""
	chart = parse((charts_root / row["urc_path"]).read_text(encoding="utf-8"))
	if row["rate_group"] == "NM":
		return chart, 0

	scaled = scale_rate(chart, RATE_GROUP_RATES[row["rate_group"]]).chart
	return scaled, count_split_chords(chart, scaled)


def _values(values: Sequence, dtype: type[pl.DataType] = pl.String) -> pl.Series:
	return pl.Series(list(values), dtype=dtype).implode()

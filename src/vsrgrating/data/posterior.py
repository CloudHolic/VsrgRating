"""The cells a fixed person posterior is built from."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .load import Split
from .tables import RESPONSES, k_core

KEPT_PARTS = ("train", "dev", "valid")
SPLIT_PARTS = (*KEPT_PARTS, "test", "dropped")


@dataclass(frozen=True)
class PosteriorCells:
	"""Train, dev and valid cells with `in_t` marking T, and T as a Split over its own persons."""

	frame: pl.DataFrame
	t: Split
	n_persons: int
	split_hash: str
	split_chart_index: str


def posterior_cells(
	dataset_dir: Path,
	split_dir: Path,
	response: str,
	min_item: int,
	min_person: int,
	drop_keys: Sequence[int],
) -> PosteriorCells:
	"""Cells of every person on train, dev and valid items."""
	if response not in RESPONSES:
		raise ValueError(f"response must be one of {RESPONSES}, got {response!r}")

	manifest = _read_json(dataset_dir / "manifest.json")
	if manifest["hash"] != dataset_dir.name:
		raise ValueError(f"manifest hash {manifest['hash']} does not match {dataset_dir.name}")

	split_manifest = _read_json(split_dir / "manifest.json")
	if split_manifest["dataset_hash"] != manifest["hash"]:
		raise ValueError(
			f"{split_dir} splits dataset {split_manifest['dataset_hash']}, not {manifest['hash']}"
		)

	n_items = manifest["counts"]["items"]
	split = pl.read_parquet(split_dir / "mapset_split.parquet").sort("item_idx")
	if split["item_idx"].to_list() != list(range(n_items)):
		raise ValueError(f"{split_dir} does not hold every item 0..{n_items - 1}")

	unknown = split.filter(pl.col("part").is_null() | ~pl.col("part").is_in(SPLIT_PARTS))
	if unknown.height:
		raise ValueError(f"{split_dir}: {unknown.height} items without a known part")

	persons = pl.read_parquet(dataset_dir / "persons.parquet").sort("person_idx")
	if persons["person_idx"].to_list() != list(range(persons.height)):
		raise ValueError(f"{dataset_dir}: person_idx is not 0..{persons.height - 1}")

	cells = (
		pl.read_parquet(dataset_dir / "responses.parquet")
		.select("person_idx", "item_idx", pl.col(response).alias("x"))
		.join(
			split.select("item_idx", "beatmap_id", "rate_group", "beatmapset_id", "keys", "part"),
			on="item_idx",
		)
		.filter(pl.col("part").is_in(KEPT_PARTS))
		.join(persons.select("person_idx", "user_id"), on="person_idx")
	)

	dropped = pl.Series(list(drop_keys), dtype=pl.Int64).implode()
	candidates = cells.filter((pl.col("part") == "train") & ~pl.col("keys").is_in(dropped))
	core = k_core(candidates, min_item, min_person, pl.lit(True))
	if core.height == 0:
		raise ValueError("T is empty")

	frame = (
		cells.join(
			core.select("person_idx", "item_idx", in_t=pl.lit(True)),
			on=["person_idx", "item_idx"],
			how="left",
		)
		.with_columns(pl.col("in_t").fill_null(False))
		.select("person_idx", "item_idx", "beatmapset_id", "keys", "part", "in_t", "x")
		.sort("person_idx", "item_idx")
	)

	return PosteriorCells(
		frame=frame,
		t=_t_split(frame, n_items, manifest["hash"]),
		n_persons=persons.height,
		split_hash=split_hash(split),
		split_chart_index=split_manifest["chart_index"],
	)


def split_hash(split: pl.DataFrame) -> str:
	"""The content hash of a mapset split table."""
	# The split manifest carries no hash of its own; hashing the content, not the file bytes,
	# keeps a rewritten copy of the same split equal.
	content = split.sort("item_idx").select("item_idx", "beatmapset_id", "keys", "part")
	return hashlib.blake2b(content.write_csv().encode(), digest_size=6).hexdigest()


def _t_split(frame: pl.DataFrame, n_items: int, dataset_hash: str) -> Split:
	t = frame.filter(pl.col("in_t"))
	person_idx = np.unique(t["person_idx"].to_numpy())

	return Split(
		person=np.searchsorted(person_idx, t["person_idx"].to_numpy()),
		item=t["item_idx"].to_numpy(),
		x=t["x"].to_numpy(),
		n_persons=person_idx.size,
		n_items=n_items,
		person_idx=person_idx,
		dataset_hash=dataset_hash,
	)


def _read_json(path: Path) -> dict:
	return json.loads(path.read_text(encoding="utf-8"))

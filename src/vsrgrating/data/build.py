"""Builds a dataset directory from the database: core, items, persons, split, tables, manifest."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import psycopg

from .anchors import DAN_LADDERS, UNIT_SCHEMA, eligible_units, record
from .config import DatasetConfig
from .manifest import content_hash, make_manifest
from .query import connect, fetch_beatmaps, fetch_cells, fetch_ingest_log, fetch_view_definitions
from .sampling import (
	allocate,
	sample_items,
	sample_units,
	spawn_rngs,
	split_persons,
	subsample_persons,
)
from .tables import build_tables, k_core

ITEM_KEY = ["beatmap_id", "rate_group"]


def _size(cells: pl.DataFrame) -> dict[str, int]:
	return {
		"cells": cells.height,
		"items": cells.select(ITEM_KEY).n_unique(),
		"persons": cells["user_id"].n_unique(),
	}


def build(config: DatasetConfig, dsn: str, out_root: Path) -> Path:
	"""Builds the dataset and returns its directory."""
	rngs = spawn_rngs(config.seed)
	with connect(dsn) as conn:
		cells = fetch_cells(conn)
		view_definitions = fetch_view_definitions(conn)
		ingest_log = fetch_ingest_log(conn)
		stages = {"pool": _size(cells)}

		core = k_core(cells, config.min_item_responses, config.min_person_responses, pl.lit(True))
		stages["core"] = _size(core)
		units, n_eligible_units = _choose_units(conn, core, config.mapset_items, rngs["units"])

	candidates = core.group_by(*ITEM_KEY, "keys").len("n_responses").sort("keys", *ITEM_KEY)
	eligible = dict(candidates.group_by("keys").len().iter_rows())
	allocation = allocate(eligible, config.n_items)
	picked = sample_items(candidates, allocation, rngs["items"])

	nm = pl.col("rate_group") == "NM"
	unit_ids = [b for order in units["order"] for b in order]
	dan_ids = [b for ladder in DAN_LADDERS.values() for b in ladder] if config.dan_courses else []
	on_picked = pl.concat(
		[
			core.join(picked.select(ITEM_KEY), on=ITEM_KEY, how="semi"),
			core.filter(nm & pl.col("beatmap_id").is_in(unit_ids)),
			# Dan charts skip the core: how far up the ladder they reach is itself the result.
			cells.filter(nm & pl.col("beatmap_id").is_in(dan_ids)),
		]
	).unique(subset=["user_id", *ITEM_KEY], maintain_order=True)

	drawn_items = on_picked.select(*ITEM_KEY, "keys").unique(maintain_order=True)
	active = (
		on_picked.group_by("user_id").len().filter(pl.col("len") >= config.min_person_responses)
	)
	on_picked = on_picked.join(active.select("user_id"), on="user_id", how="semi")
	stages["picked"] = _size(on_picked)

	drawn = subsample_persons(
		np.sort(active["user_id"].to_numpy()), config.n_persons, rngs["persons"]
	)
	test = (
		np.zeros(len(drawn), dtype=bool)
		if config.test_fraction is None
		else split_persons(len(drawn), config.test_fraction, rngs["split"])
	)
	split = pl.DataFrame({"user_id": drawn, "split": np.where(test, "test", "train")})
	sample = on_picked.join(split, on="user_id")
	stages["drawn"] = _size(sample)

	final = k_core(
		sample,
		config.min_item_train_responses,
		config.min_person_responses,
		pl.col("split") == "train",
	)
	stages["final"] = _size(final)
	kept_items = drawn_items.join(final.select(ITEM_KEY).unique(), on=ITEM_KEY, how="semi")
	items, persons, responses = build_tables(kept_items, final)
	drawn_nm = set(picked.filter(nm)["beatmap_id"].to_list())
	anchors = record(
		units, n_eligible_units, config.mapset_items, drawn_nm, final, config.dan_courses
	)

	hashed = make_manifest(
		config,
		view_definitions,
		ingest_log,
		eligible,
		allocation,
		stages,
		anchors,
		items,
		persons,
		responses,
	)

	tables = {"items": items, "persons": persons, "responses": responses}
	digest = content_hash(tables, hashed)

	target = out_root / config.name / digest
	if target.exists():
		return target

	staging = out_root / config.name / f".staging-{digest}-{os.getpid()}"
	staging.mkdir(parents=True)

	for name, table in tables.items():
		table.write_parquet(staging / f"{name}.parquet")

	manifest = {**hashed, "hash": digest, "built_at": datetime.now(UTC).isoformat()}

	with (staging / "manifest.json").open("w", encoding="utf-8") as f:
		json.dump(manifest, f, indent=2, ensure_ascii=False)

	staging.rename(target)
	return target


def _choose_units(
	conn: psycopg.Connection, core: pl.DataFrame, target: int, rng: np.random.Generator
) -> tuple[pl.DataFrame, int]:
	"""Mapset units among the core's NM charts, drawn up to `target` charts, and how many were
	eligible."""
	if target == 0:
		return pl.DataFrame(schema=UNIT_SCHEMA), 0

	charts = core.filter(pl.col("rate_group") == "NM").select("beatmap_id", "keys").unique()
	beatmaps = fetch_beatmaps(conn, charts["beatmap_id"].to_list()).join(charts, on="beatmap_id")
	eligible = eligible_units(beatmaps)

	return sample_units(eligible, target, rng), eligible.height

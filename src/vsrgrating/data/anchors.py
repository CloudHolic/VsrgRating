"""Items added because their order is known: mapset ladders and dan courses."""

import re
from typing import Any

import polars as pl

DAN_LADDERS: dict[str, tuple[int, ...]] = {
	"regular": (
		965651, 965652, 965653, 965654,
		969187, 969188, 969189, 969190, 1673133,
		1942650, 1966467, 1979885, 2030894, 2194084,
	),
	"ln": (
		966813, 966814, 966815, 966816,
		1875999, 1872112, 1872902, 1870418, 1877529,
		2539251, 2545327, 2556940, 2570281, 2764028,
	),
}  # fmt: skip
"""Jinjin's osu!mania 7K Dan Course, Regular and LN, from 0th Dan to Zenith Dan.

One mapper's judgement of the ladder, not a fact about the game. Beatmapsets 450069, 451788 and
930218 (Regular Phase I-III); 450649, 895138 and 1220647 (LN Phase I-III).
"""

NAME_LADDERS: dict[str, dict[str, int]] = {
	"osu": {
		"beginner": 0,
		"easy": 1,
		"normal": 2,
		"advanced": 3,
		"hard": 4,
		"hyper": 5,
		"insane": 6,
		"expert": 7,
		"extra": 7,
	},
	"djmax": {"ez": 0, "nm": 1, "hd": 2, "mx": 3, "sc": 4},
	"sdvx": {
		"novice": 0,
		"advanced": 1,
		"exhaust": 2,
		"maximum": 3,
		"infinite": 4,
		"gravity": 4,
		"heavenly": 4,
		"vivid": 4,
		"exceed": 4,
	},
	"iidx": {"beginner": 0, "normal": 1, "hyper": 2, "another": 3, "leggendaria": 4},
	"arcaea": {"past": 0, "present": 1, "future": 2, "beyond": 3},
	"touhou": {"easy": 0, "normal": 1, "hard": 2, "lunatic": 3},
}
"""Difficulty names whose order is plain within one naming scheme, easiest first."""

MIN_UNIT_ITEMS = 2
MAX_UNIT_ITEMS = 8

_RATE_EDIT = re.compile(r"\d+(\.\d+)?\s*x\b|\bx\s*\d+(\.\d+)?|\d+\s*bpm", re.IGNORECASE)
_POSSESSIVE = re.compile(r"^[^']*'s?\s+")
_KEY_TAG = re.compile(r"\b\d+k\b")
_BRACKETS = re.compile(r"[\[\]()]")

UNIT_SCHEMA = {
	"beatmapset_id": pl.Int64,
	"keys": pl.Int64,
	"ladder": pl.String,
	"order": pl.List(pl.Int64),
}


def eligible_units(beatmaps: pl.DataFrame) -> pl.DataFrame:
	"""(beatmapset_id, keys) units whose difficulty names alone give their order."""
	grouped = (
		beatmaps.group_by("beatmapset_id", "keys")
		.agg(pl.col("beatmap_id"), pl.col("version"))
		.filter(pl.col("beatmap_id").list.len().is_between(MIN_UNIT_ITEMS, MAX_UNIT_ITEMS))
	)

	rows = []
	for beatmapset_id, keys, beatmap_ids, versions in grouped.iter_rows():
		found = _order(beatmap_ids, versions)
		if found is not None:
			ladder, order = found
			rows.append(
				{"beatmapset_id": beatmapset_id, "keys": keys, "ladder": ladder, "order": order}
			)

	return pl.DataFrame(rows, schema=UNIT_SCHEMA)


def record(
	units: pl.DataFrame,
	eligible: int,
	target: int,
	drawn: set[int],
	final: pl.DataFrame,
	dan_courses: bool,
) -> dict[str, Any]:
	"""The mapset units and dan charts as they survived into the dataset, for the manifest."""
	nm = final.filter(pl.col("rate_group") == "NM").group_by("beatmap_id").len()
	responses = dict(nm.iter_rows())

	kept = []
	for beatmapset_id, keys, ladder, order in units.iter_rows():
		present = [b for b in order if b in responses]
		if len(present) >= MIN_UNIT_ITEMS:
			kept.append(
				{"beatmapset_id": beatmapset_id, "keys": keys, "ladder": ladder, "order": present}
			)

	unit_items = [b for unit in kept for b in unit["order"]]

	return {
		"mapset_items_target": target,
		"mapset": {
			"eligible_units": eligible,
			"chosen_units": units.height,
			"units_kept": len(kept),
			"units_dropped": units.height - len(kept),
			"items": len(unit_items),
			"items_already_drawn": sum(b in drawn for b in unit_items),
			"pairs": sum(len(u["order"]) * (len(u["order"]) - 1) // 2 for u in kept),
			"units": kept,
		},
		"dan": {
			ladder: [{"beatmap_id": b, "responses": responses.get(b, 0)} for b in beatmap_ids]
			for ladder, beatmap_ids in DAN_LADDERS.items()
		}
		if dan_courses
		else {},
	}


def _normalize(version: str) -> str:
	name = _POSSESSIVE.sub("", version.strip().lower())
	name = _BRACKETS.sub(" ", _KEY_TAG.sub(" ", name))

	return " ".join(name.split())


def _order(beatmap_ids: list[int], versions: list[str | None]) -> tuple[str, list[int]] | None:
	if any(v is None or _RATE_EDIT.search(v) for v in versions):
		return None

	names = [_normalize(v) for v in versions if v is not None]
	orders: dict[str, list[int]] = {}

	for ladder, ranks in NAME_LADDERS.items():
		if not all(name in ranks for name in names):
			continue

		levels = [ranks[name] for name in names]
		if len(set(levels)) != len(levels):
			return None

		orders[ladder] = [b for _, b in sorted(zip(levels, beatmap_ids, strict=True))]

	if not orders:
		return None

	first = next(iter(orders.values()))
	if any(order != first for order in orders.values()):
		return None

	return "+".join(sorted(orders)), first

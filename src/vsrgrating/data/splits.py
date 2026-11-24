"""Fixed train/test assignment of beatmapsets, stable as the source database grows."""

import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from urc_converter import UrcError, parse, scale_rate, write

TEST_SALT = "vsrgrating-test-2026-11-20"
TEST_MODULUS = 10_000
TEST_THRESHOLD = 1_000
TEST_FRACTION = TEST_THRESHOLD / TEST_MODULUS

# The 7K dan course mapsets, held out so the final evaluation can check the course order.
TEST_PINNED_MAPSETS: frozenset[int] = frozenset({450069, 450649, 451788, 895138, 930218, 1220647})

VALID_FRACTION = 0.20
DEV_FRACTION = 0.10
# A part listed earlier keeps a chart that several parts share.
PART_PRIORITY = ("test", "valid", "dev", "train")
UNUSABLE = ("chart_failed", "rate_failed")
RATE_GROUP_RATES = {"NM": 1.0, "DT": 1.5, "HT": 0.75}


def is_test_mapset(beatmapset_id: int) -> bool:
	"""Whether a beatmapset belongs to the final test set, by a salted hash of its id."""
	if isinstance(beatmapset_id, bool) or not isinstance(beatmapset_id, int) or beatmapset_id <= 0:
		raise ValueError(f"beatmapset_id must be a positive int, got {beatmapset_id!r}")

	if beatmapset_id in TEST_PINNED_MAPSETS:
		return True

	digest = hashlib.blake2b(f"{TEST_SALT}:{beatmapset_id}".encode(), digest_size=8).digest()

	return int.from_bytes(digest, "big") % TEST_MODULUS < TEST_THRESHOLD


def split_mapsets(
	items: pl.DataFrame, seed: int, *, test_hashes: set[str] | frozenset[str] = frozenset()
) -> pl.DataFrame:
	"""Each item's mapset part and its own part: mapset-level, stratified by the rarest key mode."""
	reasons = set(items["unusable_reason"].drop_nulls())
	if unknown := reasons - set(UNUSABLE):
		raise ValueError(f"unknown unusable reasons {sorted(unknown)}")
	if items["beatmapset_id"].null_count():
		raise ValueError("items without a beatmapset_id")

	rows = items.select("beatmapset_id", "keys", "unusable_reason").iter_rows()
	usable_keys: dict[int, set[int]] = {}
	for set_id, keys, reason in rows:
		found = usable_keys.setdefault(set_id, set())
		if reason is None:
			found.add(keys)

	mapset_part: dict[int, str | None] = {}
	candidates = []
	for set_id, keys in usable_keys.items():
		if is_test_mapset(set_id):
			mapset_part[set_id] = "test"
		elif keys:
			candidates.append(set_id)
		else:
			mapset_part[set_id] = None

	sets_with_key = Counter(key for set_id in candidates for key in usable_keys[set_id])
	strata: dict[int, list[int]] = {}
	for set_id in candidates:
		stratum = min(usable_keys[set_id], key=lambda key: (sets_with_key[key], key))
		strata.setdefault(stratum, []).append(set_id)

	rng = np.random.default_rng(seed)
	for stratum in sorted(strata):
		members = sorted(strata[stratum])
		if len(members) < 3:
			raise ValueError(f"{stratum}K has {len(members)} non-test mapsets, fewer than 3")

		order = [members[i] for i in rng.permutation(len(members))]
		n_valid = math.ceil(VALID_FRACTION * len(order))
		n_dev = math.ceil(DEV_FRACTION * (len(order) - n_valid))
		for i, set_id in enumerate(order):
			mapset_part[set_id] = (
				"valid" if i < n_valid else "dev" if i < n_valid + n_dev else "train"
			)

	split = items.with_columns(
		pl.col("beatmapset_id")
		.replace_strict(mapset_part, return_dtype=pl.String)
		.alias("mapset_part")
	)

	rank = {part: i for i, part in enumerate(PART_PRIORITY)}
	best = dict.fromkeys(test_hashes, 0)
	for part, content, reason in split.select(
		"mapset_part", "content_hash", "unusable_reason"
	).iter_rows():
		if reason is None and content is not None:
			best[content] = min(best.get(content, rank[part]), rank[part])

	parts, drop_reasons = [], []
	for part, content, reason in split.select(
		"mapset_part", "content_hash", "unusable_reason"
	).iter_rows():
		if reason is not None:
			parts.append("dropped")
			drop_reasons.append(reason)
		elif content is not None and best[content] < rank[part]:
			parts.append("dropped")
			drop_reasons.append("duplicate")
		else:
			parts.append(part)
			drop_reasons.append(None)

	return split.with_columns(
		pl.Series("part", parts, dtype=pl.String),
		pl.Series("drop_reason", drop_reasons, dtype=pl.String),
	)


def build_mapset_split(
	dataset_dir: Path, charts_root: Path, chart_index: str, out_root: Path, seed: int
) -> Path:
	"""Joins dataset items to a chart index snapshot, checks rates, splits, and writes it."""
	target = out_root / dataset_dir.parent.name / dataset_dir.name
	if target.exists():
		raise FileExistsError(target)

	index = pl.read_parquet(charts_root / "index" / chart_index / "index.parquet")
	archive = index.filter(pl.col("chart_source") == "osu_archive").select(
		"beatmap_id",
		"beatmapset_id",
		pl.col("keys").alias("urc_keys"),
		"keys_db",
		"urc_path",
		"convert_status",
		"content_hash",
	)
	items = pl.read_parquet(dataset_dir / "items.parquet").join(
		archive, on="beatmap_id", how="left"
	)
	if missing := items.filter(pl.col("convert_status").is_null())["beatmap_id"].to_list():
		raise ValueError(f"{len(missing)} dataset charts are not in the index, e.g. {missing[:5]}")
	if unknown := set(items["rate_group"]) - set(RATE_GROUP_RATES):
		raise ValueError(f"unknown rate groups {sorted(unknown)}")

	start = time.monotonic()
	reasons, rates = _usable(items, charts_root)
	items = items.with_columns(
		pl.Series("unusable_reason", reasons, dtype=pl.String),
		pl.coalesce("urc_keys", "keys").alias("keys"),
	)
	rates["seconds"] = round(time.monotonic() - start, 1)

	test_hashes = set(
		index.filter(pl.col("test") & pl.col("content_hash").is_not_null())["content_hash"]
	)
	split = split_mapsets(items, seed, test_hashes=test_hashes)

	staging = target.with_name(f".staging-{target.name}-{os.getpid()}")
	staging.mkdir(parents=True)
	columns = ["item_idx", "beatmap_id", "rate_group", "keys", "beatmapset_id"]
	split.select(*columns, "mapset_part", "part", "drop_reason").write_parquet(
		staging / "mapset_split.parquet"
	)
	manifest = {
		"dataset": dataset_dir.parent.name,
		"dataset_hash": dataset_dir.name,
		"chart_index": chart_index,
		"seed": seed,
		"test_salt": TEST_SALT,
		"test_pinned_mapsets": sorted(TEST_PINNED_MAPSETS),
		"parts": _sizes(split),
		"drop_reason": dict(Counter(split["drop_reason"].drop_nulls())),
		"keys_mismatch": split.filter(pl.col("urc_keys") != pl.col("keys_db")).height,
		"rates": rates,
		"built_at": datetime.now(UTC).isoformat(timespec="seconds"),
	}
	(staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
	staging.rename(target)

	return target


def _usable(items: pl.DataFrame, charts_root: Path) -> tuple[list[str | None], dict]:
	"""Why each item cannot be used, checking DT and HT on the NM chart."""
	reasons: list[str | None] = []
	failures: Counter[str] = Counter()
	checked = warnings = 0

	for status, urc_path, rate_group in items.select(
		"convert_status", "urc_path", "rate_group"
	).iter_rows():
		if status != "ok":
			reasons.append("chart_failed")
			continue
		if rate_group == "NM":
			reasons.append(None)
			continue

		checked += 1
		chart = parse((charts_root / urc_path).read_text(encoding="utf-8"))
		try:
			scaled = scale_rate(chart, RATE_GROUP_RATES[rate_group])
			parse(write(scaled.chart))
		except UrcError as e:
			failures[e.category] += 1
			reasons.append("rate_failed")
			continue

		warnings += len(scaled.warnings)
		reasons.append(None)

	return reasons, {"checked": checked, "failed": dict(failures), "warnings": warnings}


def _sizes(split: pl.DataFrame) -> list[dict]:
	return (
		split.group_by("part", "keys")
		.agg(
			pl.len().alias("items"),
			pl.col("beatmapset_id").n_unique().alias("mapsets"),
			(pl.col("n_train") + pl.col("n_test")).sum().alias("cells"),
		)
		.sort("part", "keys")
		.to_dicts()
	)

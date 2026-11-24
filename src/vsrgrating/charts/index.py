"""Builds one dated index over every converted chart, whatever its source."""

import hashlib
import json
import os
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ..data import connect, fetch_mania_beatmaps, is_test_mapset
from .convert import CONVERTER_VERSION, SCHEMA_VERSION, convert_cached
from .sources import (
	ExternalFiles,
	SourceFile,
	bms_packages,
	bms_sabun,
	etterna,
	iter_osu_archive,
	osu_external,
	quaver,
)

SOURCES = ("osu_archive", "osu_external", "etterna", "quaver", "bms_sabun", "bms_package")
EXTERNAL = (
	("osu_external", osu_external),
	("etterna", etterna),
	("quaver", quaver),
	("bms_sabun", bms_sabun),
	("bms_package", bms_packages),
)
PROGRESS_EVERY = 5000

_SCHEMA = {
	"chart_id": pl.String,
	"chart_source": pl.String,
	"status": pl.String,
	"group_id": pl.String,
	"beatmap_id": pl.Int64,
	"beatmapset_id": pl.Int64,
	"pack": pl.String,
	"raw_path": pl.String,
	"raw_md5": pl.String,
	"urc_path": pl.String,
	"convert_status": pl.String,
	"error": pl.String,
	"keys": pl.Int64,
	"special_keys": pl.Int64,
	"keys_db": pl.Int64,
	"n_notes": pl.Int64,
	"n_tokens": pl.Int64,
	"content_hash": pl.String,
	"test": pl.Boolean,
}


def build_index(
	sources: list[str],
	dsn: str,
	archive: Path,
	external_root: Path,
	out_root: Path,
	name: str,
) -> Path:
	"""Converts what the cache lacks, writes `out_root/index/<name>`, and returns that directory."""
	if unknown := set(sources) - set(SOURCES):
		raise ValueError(f"unknown sources {sorted(unknown)}; pick from {SOURCES}")

	target = out_root / "index" / name
	if target.exists():
		raise FileExistsError(target)

	start = time.monotonic()
	cache_root = out_root / "urc"
	rows: list[dict] = []
	manifest: dict = {"name": name, "sources": sources}

	if "osu_archive" in sources:
		with connect(dsn) as conn:
			beatmaps = fetch_mania_beatmaps(conn)
		keys_db = dict(beatmaps.select("beatmap_id", "keys").iter_rows())
		found = _convert(iter_osu_archive(archive, beatmaps), cache_root, rows, keys_db)
		for beatmap_id, set_id in beatmaps.select("beatmap_id", "beatmapset_id").iter_rows():
			if beatmap_id not in found:
				rows.append(_missing(beatmap_id, set_id, archive, keys_db[beatmap_id]))
		manifest["osu_archive"] = {
			"file": archive.name,
			"bytes": archive.stat().st_size,
			"sha256": _sha256(archive),
			"database_charts": beatmaps.height,
		}

	for source, read in EXTERNAL:
		if source in sources:
			external = read(external_root)
			_convert(external.files, cache_root, rows, {})
			manifest[source] = _external_summary(external)

	index = _finish(pl.DataFrame(rows, schema=_SCHEMA), manifest)
	manifest |= _counts(index) | {
		"urc_converter": CONVERTER_VERSION,
		"cache_schema": SCHEMA_VERSION,
		"built_at": datetime.now(UTC).isoformat(timespec="seconds"),
		"seconds": round(time.monotonic() - start, 1),
	}

	staging = target.with_name(f".staging-{name}-{os.getpid()}")
	staging.mkdir(parents=True)
	index.write_parquet(staging / "index.parquet")
	(staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
	staging.rename(target)

	return target


def _convert(
	files: Iterable[SourceFile], cache_root: Path, rows: list[dict], keys_db: dict[int, int]
) -> set[int]:
	"""Appends one row per chart (or per failed file) and returns the beatmap ids seen."""
	seen: set[int] = set()
	start = time.monotonic()

	for n, file in enumerate(files, 1):
		entry = convert_cached(file.source_format, file.raw_md5, file.read, cache_root)
		base = {
			"chart_source": file.chart_source,
			"status": file.status,
			"group_id": file.group_id,
			"beatmap_id": file.beatmap_id,
			"beatmapset_id": file.beatmapset_id,
			"pack": file.pack,
			"raw_path": file.raw_path,
			"raw_md5": file.raw_md5,
			"convert_status": entry.convert_status,
			"error": entry.error,
			"keys_db": keys_db.get(file.beatmap_id) if file.beatmap_id is not None else None,
		}
		if file.beatmap_id is not None:
			seen.add(file.beatmap_id)

		if not entry.charts:
			rows.append(base | {"chart_id": _chart_id(file, None)})
		for chart, urc_path in zip(entry.charts, entry.urc_paths, strict=True):
			rows.append(
				base
				| {
					"chart_id": _chart_id(file, chart.k),
					"urc_path": urc_path,
					"keys": chart.keys,
					"special_keys": chart.special_keys,
					"n_notes": chart.n_notes,
					"n_tokens": chart.n_tokens,
					"content_hash": chart.content_hash,
				}
			)

		if n % PROGRESS_EVERY == 0:
			print(f"{file.chart_source}: {n} files, {time.monotonic() - start:.0f}s", flush=True)

	return seen


def _chart_id(file: SourceFile, k: int | None) -> str:
	if file.source_format == "sm" and k is not None:
		return f"{file.file_id}#{k}"
	return file.file_id


def _missing(beatmap_id: int, set_id: int, archive: Path, keys_db: int) -> dict:
	return {
		"chart_id": f"osu:{beatmap_id}",
		"chart_source": "osu_archive",
		"status": "ranked_loved",
		"group_id": f"osu:{set_id}",
		"beatmap_id": beatmap_id,
		"beatmapset_id": set_id,
		"raw_path": f"{archive.name}:{beatmap_id}.osu",
		"convert_status": "missing",
		"keys_db": keys_db,
	}


def _finish(index: pl.DataFrame, manifest: dict) -> pl.DataFrame:
	"""Drops external copies of archive charts, marks the final test charts, checks the keys."""
	archive_ids = index.filter(pl.col("chart_source") == "osu_archive")["beatmap_id"]
	duplicate = (pl.col("chart_source") == "osu_external") & pl.col("beatmap_id").is_in(
		archive_ids.implode()
	)
	manifest["external_rows_also_in_archive"] = index.filter(duplicate).height
	index = index.filter(~duplicate)

	test = [
		source == "osu_archive" and is_test_mapset(set_id)
		for source, set_id in index.select("chart_source", "beatmapset_id").iter_rows()
	]
	index = index.with_columns(pl.Series("test", test, dtype=pl.Boolean))

	if index["chart_id"].n_unique() != index.height:
		raise ValueError("chart_id is not unique")
	if "osu_archive" in manifest:
		archived = index.filter(pl.col("chart_source") == "osu_archive").height
		if archived != manifest["osu_archive"]["database_charts"]:
			raise ValueError(
				f"{archived} archive rows for {manifest['osu_archive']['database_charts']} charts"
			)

	return index


def _counts(index: pl.DataFrame) -> dict:
	hashed = index.filter(pl.col("content_hash").is_not_null())
	sources_per_hash = hashed.group_by("content_hash").agg(
		pl.col("chart_source").n_unique().alias("sources"), pl.col("test").any().alias("test")
	)
	external = hashed.filter(pl.col("chart_source") != "osu_archive")["content_hash"]
	return {
		"rows": index.height,
		"convert_status": _tally(index, "chart_source", "convert_status"),
		"keys": _tally(
			index.filter(pl.col("convert_status") == "ok").with_columns(
				pl.when(pl.col("special_keys") > 0)
				.then(pl.format("{}+{}", "keys", "special_keys"))
				.otherwise(pl.col("keys").cast(pl.String))
				.alias("layout")
			),
			"chart_source",
			"layout",
		),
		"keys_db_mismatch": index.filter(pl.col("keys_db") != pl.col("keys")).height,
		"hashes_in_several_sources": sources_per_hash.filter(pl.col("sources") > 1).height,
		"test_hashes_also_external": sources_per_hash.filter(
			pl.col("test") & pl.col("content_hash").is_in(external.implode())
		).height,
	}


def _tally(index: pl.DataFrame, outer: str, inner: str) -> dict:
	counts = index.group_by(outer, inner).len().sort(outer, inner)
	result: dict = {}
	for a, b, n in counts.iter_rows():
		result.setdefault(str(a), {})[str(b)] = n
	return result


def _external_summary(external: ExternalFiles) -> dict:
	return {
		"index_rows": external.rows,
		"files": len(external.files),
		"skipped": external.skipped,
		"collector_errors": external.errors,
		"last_fetched": external.last_fetched,
	}


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with open(path, "rb") as f:
		while chunk := f.read(1 << 24):
			digest.update(chunk)
	return digest.hexdigest()

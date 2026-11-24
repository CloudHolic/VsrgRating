"""Source chart files: the osu! `osu_files` archive and the externally collected charts."""

import hashlib
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .archive import iter_osu
from .convert import SourceFormat

OSU_EXTERNAL_INDEX = "index.sqlite"
ETTERNA_INDEX = "index_etterna.sqlite"
QUAVER_INDEX = "index_quaver.sqlite"
BMS_SABUN_INDEX = "index_bms.sqlite"
BMS_PACKAGE_INDEX = "index_bms_packages.sqlite"


@dataclass(frozen=True)
class SourceFile:
	"""One source file to convert, with where it came from and how it is graded there."""

	file_id: str
	chart_source: str
	source_format: SourceFormat
	status: str
	group_id: str
	beatmap_id: int | None
	beatmapset_id: int | None
	pack: str | None
	raw_path: str
	raw_md5: str
	read: Callable[[], bytes]


@dataclass(frozen=True)
class ExternalFiles:
	"""The files an external collection has on disk, and what its index says about the rest."""

	files: list[SourceFile]
	rows: int
	skipped: dict[str, int]
	errors: dict[str, int]
	last_fetched: str | None


def iter_osu_archive(archive: Path, beatmaps: pl.DataFrame) -> Iterator[SourceFile]:
	"""The database's osu!mania charts that are in the archive, read in one streaming pass."""
	sets = dict(beatmaps.select("beatmap_id", "beatmapset_id").iter_rows())

	for beatmap_id, raw in iter_osu(archive, set(sets)):
		yield SourceFile(
			file_id=f"osu:{beatmap_id}",
			chart_source="osu_archive",
			source_format="osu",
			status="ranked_loved",
			group_id=f"osu:{sets[beatmap_id]}",
			beatmap_id=beatmap_id,
			beatmapset_id=sets[beatmap_id],
			pack=None,
			raw_path=f"{archive.name}:{beatmap_id}.osu",
			raw_md5=hashlib.md5(raw).hexdigest(),
			read=lambda raw=raw: raw,
		)


def osu_external(root: Path) -> ExternalFiles:
	"""The unranked osu!mania charts the external collector has downloaded."""
	rows = _read(
		root / OSU_EXTERNAL_INDEX,
		"SELECT chart_id, set_id, status, path, md5, error, fetched_at"
		" FROM charts WHERE source = 'osu'",
	)
	files = [
		SourceFile(
			file_id=f"osu:{chart_id}",
			chart_source="osu_external",
			source_format="osu",
			status=status,
			group_id=f"osu:{set_id}",
			beatmap_id=chart_id,
			beatmapset_id=set_id,
			pack=None,
			raw_path=path,
			raw_md5=md5,
			read=lambda path=path: (root / path).read_bytes(),
		)
		for chart_id, set_id, status, path, md5, _error, _fetched in rows
		if path is not None
	]
	return ExternalFiles(
		files=files,
		rows=len(rows),
		skipped=dict(
			Counter(
				"not_fetched" if row[6] is None else "no_file" for row in rows if row[3] is None
			)
		),
		errors=dict(Counter(str(row[5]) for row in rows)),
		last_fetched=max((row[6] for row in rows if row[6] is not None), default=None),
	)


def etterna(root: Path) -> ExternalFiles:
	"""The `.sm` and `.ssc` files the external collector kept from Etterna packs."""
	rows = _read(
		root / ETTERNA_INDEX,
		"SELECT f.path, f.md5, f.fetched_at, p.name FROM files f JOIN packs p USING (pack_id)",
	)
	files = [
		SourceFile(
			file_id=f"etterna:{path}",
			chart_source="etterna",
			source_format="sm",
			status="etterna_pack",
			group_id=f"etterna:{Path(path).parent}",
			beatmap_id=None,
			beatmapset_id=None,
			pack=name,
			raw_path=path,
			raw_md5=md5,
			read=lambda path=path: (root / path).read_bytes(),
		)
		for path, md5, _fetched, name in rows
	]
	return ExternalFiles(
		files=files,
		rows=len(rows),
		skipped={},
		errors={},
		last_fetched=max((row[2] for row in rows if row[2] is not None), default=None),
	)


def quaver(root: Path) -> ExternalFiles:
	"""The Quaver maps the external collector has downloaded."""
	rows = _read(
		root / QUAVER_INDEX,
		"SELECT map_id, mapset_id, status, path, md5_file, error, fetched_at FROM maps",
	)
	files = [
		SourceFile(
			file_id=f"quaver:{map_id}",
			chart_source="quaver",
			source_format="qua",
			status=status,
			group_id=f"quaver:{mapset_id}",
			beatmap_id=None,
			beatmapset_id=None,
			pack=None,
			raw_path=path,
			raw_md5=md5,
			read=lambda path=path: (root / path).read_bytes(),
		)
		for map_id, mapset_id, status, path, md5, _error, _fetched in rows
		if path is not None and md5 is not None
	]
	return ExternalFiles(
		files=files,
		rows=len(rows),
		skipped=dict(
			Counter(
				_skipped(path, md5, fetched)
				for _id, _set, _status, path, md5, _error, fetched in rows
				if path is None or md5 is None
			)
		),
		errors=dict(Counter(str(row[5]) for row in rows)),
		last_fetched=max((row[6] for row in rows if row[6] is not None), default=None),
	)


def bms_sabun(root: Path) -> ExternalFiles:
	"""Chart-only BMS files (sabun) linked from difficulty tables, graded by their table."""
	rows = _read(
		root / BMS_SABUN_INDEX,
		"SELECT f.path, f.md5, f.fetched_at, s.table_name, s.entry_md5 FROM bms_files f"
		" JOIN bms_sabun s ON f.origin = s.entry_md5 AND f.source = 'bms:sabun:' || s.table_name",
	)
	files = [
		_bms_file(root, "bms_sabun", path, md5, table, f"bms:sabun:{table}:{entry}")
		for path, md5, _fetched, table, entry in rows
	]
	return ExternalFiles(
		files=files,
		rows=len(rows),
		skipped={},
		errors={},
		last_fetched=max((row[2] for row in rows if row[2] is not None), default=None),
	)


def bms_packages(root: Path) -> ExternalFiles:
	"""BMS chart files taken out of song packages, graded by the event they were entered in."""
	rows = _read(
		root / BMS_PACKAGE_INDEX,
		"SELECT f.path, f.md5, f.fetched_at, s.exhibition, f.bms_id FROM files f"
		" JOIN songs s USING (bms_id)",
	)
	files = [
		_bms_file(root, "bms_package", path, md5, exhibition or "no_event", f"bms:{bms_id}")
		for path, md5, _fetched, exhibition, bms_id in rows
	]
	return ExternalFiles(
		files=files,
		rows=len(rows),
		skipped={},
		errors={},
		last_fetched=max((row[2] for row in rows if row[2] is not None), default=None),
	)


def _skipped(path: str | None, md5: str | None, fetched: str | None) -> str:
	if path is None:
		return "not_fetched" if fetched is None else "no_file"
	return "no_md5"


def _bms_file(
	root: Path, chart_source: str, path: str, md5: str, status: str, group_id: str
) -> SourceFile:
	return SourceFile(
		file_id=f"{chart_source}:{path}",
		chart_source=chart_source,
		source_format="pms" if path.lower().endswith(".pms") else "bms",
		status=status,
		group_id=group_id,
		beatmap_id=None,
		beatmapset_id=None,
		pack=None,
		raw_path=path,
		raw_md5=md5,
		read=lambda: (root / path).read_bytes(),
	)


def _read(path: Path, query: str) -> list[tuple]:
	# The collectors keep writing; a long read transaction would make their commits fail.
	conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
	try:
		return conn.execute(query).fetchall()
	finally:
		conn.close()

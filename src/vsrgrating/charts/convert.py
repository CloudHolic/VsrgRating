"""Converts source charts to URC, summarizes each chart, and caches the result by source md5."""

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from urc_converter import Chart, NoteType, UrcError, parse, write
from urc_converter.sources import (
	convert_bms,
	convert_osu,
	convert_qua,
	convert_sm,
	parse_bms,
	parse_osu,
	parse_qua,
	parse_sm,
)

# Bump when the summary or the hash changes; old caches stay in their own directory.
SCHEMA_VERSION = 1
CONVERTER_VERSION = version("urc-converter")
ENCODINGS = ("utf-8-sig", "cp932", "cp1252")
# Notes and timing are ASCII; only metadata can be lost to a mixed or unknown encoding.
ENCODING_FALLBACK = "utf-8"
ERROR_CHARS = 200

type SourceFormat = Literal["osu", "sm", "qua", "bms", "pms"]


@dataclass(frozen=True)
class ChartSummary:
	"""What the index needs from one converted chart, so a cache hit never parses URC again."""

	k: int
	keys: int
	special_keys: int
	n_notes: int
	n_tokens: int
	content_hash: str | None


@dataclass(frozen=True)
class CacheEntry:
	"""One source file's conversion: its status, its charts, and their URC paths."""

	convert_status: str
	error: str | None
	charts: list[ChartSummary]
	urc_paths: list[str]


def decode_text(raw: bytes) -> str:
	"""The chart text. Shift-JIS comes before cp1252 because older charts often use it."""
	for encoding in ENCODINGS:
		try:
			return raw.decode(encoding)
		except UnicodeDecodeError:
			continue

	return raw.decode(ENCODING_FALLBACK, errors="replace")


def osu_to_urc(raw: bytes) -> list[tuple[str, Chart]]:
	"""An osu!mania chart as one validated URC text with its parsed chart."""
	return [_validated(convert_osu(parse_osu(decode_text(raw))))]


def sm_to_urc(raw: bytes) -> list[tuple[str, Chart]]:
	"""Every chart of a StepMania or Etterna simfile, validated."""
	return [_validated(chart) for chart in convert_sm(parse_sm(decode_text(raw)))]


def qua_to_urc(raw: bytes) -> list[tuple[str, Chart]]:
	"""A Quaver map as one validated URC text with its parsed chart."""
	return [_validated(convert_qua(parse_qua(decode_text(raw))))]


def bms_to_urc(raw: bytes) -> list[tuple[str, Chart]]:
	"""A BMS chart as one validated URC text; the parser detects the encoding itself."""
	return [_validated(convert_bms(parse_bms(raw)))]


def pms_to_urc(raw: bytes) -> list[tuple[str, Chart]]:
	"""A PMS (pop'n) chart as one validated URC text."""
	return [_validated(convert_bms(parse_bms(raw, pms=True)))]


def content_hash(chart: Chart) -> str | None:
	"""sha256 of the notes up to a shift in time and a left-right mirror; None without notes."""
	notes = [note for note in chart.notes if note.type is not NoteType.F]
	if not notes:
		return None

	lanes = chart.layout.total_lanes
	start = min(note.timestamp_ms for note in notes)
	plain = sorted((note.timestamp_ms - start, note.lane, note.type.value) for note in notes)
	mirror = sorted((time, lanes - 1 - lane, kind) for time, lane, kind in plain)
	body = "\n".join(f"{time},{lane},{kind}" for time, lane, kind in min(plain, mirror))

	return hashlib.sha256(f"{lanes}\n{body}".encode()).hexdigest()


def summarize(k: int, chart: Chart) -> ChartSummary:
	"""The index columns of the k-th chart of a source file."""
	notes = [note for note in chart.notes if note.type is not NoteType.F]
	return ChartSummary(
		k=k,
		keys=chart.layout.keys,
		special_keys=chart.layout.special_keys,
		n_notes=len(notes),
		n_tokens=len({note.timestamp_ms for note in notes}),
		content_hash=content_hash(chart),
	)


_CONVERTERS: dict[str, Callable[[bytes], list[tuple[str, Chart]]]] = {
	"osu": osu_to_urc,
	"sm": sm_to_urc,
	"qua": qua_to_urc,
	"bms": bms_to_urc,
	"pms": pms_to_urc,
}


def convert_cached(
	source_format: SourceFormat, raw_md5: str, read: Callable[[], bytes], cache_root: Path
) -> CacheEntry:
	"""The cached conversion of a source file, converting and caching it on a miss."""
	target = (
		cache_root
		/ f"{CONVERTER_VERSION}-s{SCHEMA_VERSION}"
		/ source_format
		/ raw_md5[:2]
		/ raw_md5
	)
	if (target / "status.json").exists():
		return _load(target, cache_root)

	raw = read()
	if hashlib.md5(raw).hexdigest() != raw_md5:
		return CacheEntry("md5_mismatch", f"file md5 differs from {raw_md5}", [], [])

	try:
		converted = _CONVERTERS[source_format](raw)
		status, error = "ok", None
	except UnicodeDecodeError as e:
		converted, status, error = [], "decode", str(e)
	except UrcError as e:
		converted, status, error = [], e.category, str(e)
	except Exception as e:
		# A converter bug in one file is recorded in the index instead of stopping the whole run.
		converted, status, error = [], type(e).__name__, str(e)

	staging = target.with_name(f".tmp-{raw_md5}-{os.getpid()}")
	staging.mkdir(parents=True)
	for k, (text, _chart) in enumerate(converted):
		(staging / f"{k}.urc").write_text(text, encoding="utf-8")

	record = {
		"schema_version": SCHEMA_VERSION,
		"convert_status": status,
		"error": None if error is None else error[:ERROR_CHARS],
		"charts": [asdict(summarize(k, chart)) for k, (_text, chart) in enumerate(converted)],
	}
	(staging / "status.json").write_text(json.dumps(record), encoding="utf-8")

	try:
		staging.rename(target)
	except OSError:
		# Another run finished the same file first; its cache is equally valid.
		shutil.rmtree(staging)

	return _load(target, cache_root)


def _validated(chart: Chart) -> tuple[str, Chart]:
	text = write(chart)
	return text, parse(text)


def _load(target: Path, cache_root: Path) -> CacheEntry:
	record = json.loads((target / "status.json").read_text(encoding="utf-8"))
	if record["schema_version"] != SCHEMA_VERSION:
		raise ValueError(
			f"{target} has cache schema {record['schema_version']}, not {SCHEMA_VERSION}"
		)

	charts = [ChartSummary(**chart) for chart in record["charts"]]
	base = cache_root.parent
	return CacheEntry(
		convert_status=record["convert_status"],
		error=record["error"],
		charts=charts,
		urc_paths=[str((target / f"{chart.k}.urc").relative_to(base)) for chart in charts],
	)

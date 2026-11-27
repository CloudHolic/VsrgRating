"""Pick the ranked and loved osu!mania charts pretraining may read, set a watch split aside, and
fit the input and load scalers on the rest.

uv run scripts/build_pretrain_corpus.py
"""

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import torch
import typer
from omegaconf import DictConfig
from urc_converter import Chart, UrcError, parse, scale_rate

from vsrgrating.config import compose_config
from vsrgrating.data import is_test_mapset
from vsrgrating.encoder.inputs import (
	ChartTokens,
	LaneState,
	LayoutInput,
	layout_input,
	layouts,
	tokenize,
)
from vsrgrating.encoder.network import Scaler
from vsrgrating.encoder.pretrain import ChartLoad, LayoutLoad, LoadScaler, chart_load, layout_load

# Every rate pretraining may draw (0.75 to 1.50 by 0.01) must convert, or the chart is left out.
RATES = tuple(round(0.75 + 0.01 * i, 2) for i in range(76))
# Lane states 0..5 are real; MASKED comes after them and never occurs in a chart.
STATES = int(LaneState.MASKED)


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	cfg = compose_config("build_pretrain_corpus", overrides or [])
	charts_root = Path(cfg.charts_root)
	start = time.monotonic()

	candidates, excluded = select(cfg, charts_root)
	usable, failures = check_rates(candidates, charts_root, cfg.workers)
	charts = split_watch(usable, cfg)
	target = Path(cfg.out_root) / cfg.name / corpus_hash(cfg, charts)
	if target.exists():
		raise FileExistsError(target)

	train = charts.filter(pl.col("part") == "train")
	frequencies: Counter[tuple[int, int]] = Counter()
	scaler = Scaler.fit(input_stream(train, charts_root, cfg.workers, frequencies))
	load_scaler = LoadScaler.fit(load_stream(train, charts_root, cfg.workers))

	staging = target.with_name(f".staging-{target.name}-{os.getpid()}")
	staging.mkdir(parents=True)
	charts.write_parquet(staging / "charts.parquet")

	torch.save(scaler.to_dict(), staging / "scaler.pt")
	torch.save(load_scaler.to_dict(), staging / "load_scaler.pt")

	manifest = {
		"hash": target.name,
		"chart_index": cfg.chart_index,
		"split": str(cfg.split),
		"salt": cfg.salt,
		"watch_fraction": cfg.watch_fraction,
		"excluded": excluded | {"failed": failures},
		"charts": counts_by_lanes(charts),
		"state_frequencies": state_frequencies(frequencies),
		"limits": [
			"content_hash is not rate invariant: a copy uploaded at another rate to another "
			"mapset is not caught"
		],
		"seconds": round(time.monotonic() - start, 1),
		"built_at": datetime.now(UTC).isoformat(timespec="seconds"),
	}
	(staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
	staging.rename(target)
	print(target)


def select(cfg: DictConfig, charts_root: Path) -> tuple[pl.DataFrame, dict[str, int]]:
	"""Usable archive charts outside the evaluation mapsets and their content duplicates."""
	index = pl.read_parquet(charts_root / "index" / cfg.chart_index / "index.parquet")
	archive = index.filter(pl.col("chart_source") == "osu_archive")
	split = pl.read_parquet(Path(cfg.split) / "mapset_split.parquet")
	held = set(split.filter(pl.col("mapset_part").is_in(["valid", "dev"]))["beatmapset_id"])

	sets = archive["beatmapset_id"].unique().to_list()
	excluded_sets = held | {s for s in sets if is_test_mapset(int(s))}
	in_excluded = pl.col("beatmapset_id").is_in(pl.Series(sorted(excluded_sets)).implode())
	excluded_hashes = archive.filter(in_excluded & pl.col("content_hash").is_not_null())[
		"content_hash"
	].implode()

	ok = archive.filter(pl.col("convert_status") == "ok").with_columns(
		lanes=pl.col("keys") + pl.col("special_keys")
	)
	counts = {"not_converted": archive.height - ok.height}

	steps = [
		("evaluation_mapset", ~in_excluded),
		("evaluation_duplicate", ~pl.col("content_hash").is_in(excluded_hashes)),
		("no_notes", pl.col("n_notes") > 0),
		("one_token", pl.col("n_tokens") >= 2),
		("no_layout", pl.col("lanes").map_elements(_has_layout, return_dtype=pl.Boolean)),
	]

	for name, keep in steps:
		before = ok.height
		ok = ok.filter(keep)
		counts[name] = before - ok.height

	columns = ["chart_id", "urc_path", "keys", "lanes", "n_tokens", "beatmapset_id"]

	return ok.select(*columns, "content_hash"), counts


def check_rates(
	charts: pl.DataFrame, charts_root: Path, workers: int
) -> tuple[pl.DataFrame, dict[str, int]]:
	"""The charts whose inputs and loads are finite and that convert at every rate."""
	paths = [str(charts_root / p) for p in charts["urc_path"]]
	reasons: list[str | None] = []
	start = time.monotonic()

	with Pool(workers) as pool:
		for i, reason in enumerate(pool.imap(_failure, paths, chunksize=8), 1):
			reasons.append(reason)

			if i % 1000 == 0 or i == len(paths):
				elapsed = time.monotonic() - start
				left = elapsed / i * (len(paths) - i)

				print(f"rates {i}/{len(paths)}, {elapsed:.0f} s, {left:.0f} s left", flush=True)

	kept = charts.filter(pl.Series([r is None for r in reasons]))

	return kept, dict(Counter(r for r in reasons if r is not None))


def split_watch(charts: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
	"""Each mapset to `watch` by a salted hash, and train charts sharing content with a watch
	chart dropped."""
	salt = cfg.salt

	def watched(set_id: int) -> bool:
		digest = hashlib.blake2b(f"{salt}:{set_id}".encode(), digest_size=8).digest()
		return int.from_bytes(digest, "big") / 2**64 < cfg.watch_fraction

	part = charts["beatmapset_id"].map_elements(
		lambda s: "watch" if watched(int(s)) else "train", return_dtype=pl.String
	)
	charts = charts.with_columns(part=part)
	watch_hashes = charts.filter(
		(pl.col("part") == "watch") & pl.col("content_hash").is_not_null()
	)["content_hash"].implode()
	duplicate = (pl.col("part") == "train") & pl.col("content_hash").is_in(watch_hashes)

	return charts.filter(~duplicate).sort("chart_id")


def corpus_hash(cfg: DictConfig, charts: pl.DataFrame) -> str:
	"""A hash of the sources and of every chart with its part."""
	h = hashlib.blake2b(digest_size=6)
	h.update(f"{cfg.chart_index}|{cfg.split}|{cfg.salt}|{cfg.watch_fraction}".encode())
	h.update(charts.select("chart_id", "part").write_csv().encode())

	return h.hexdigest()


def input_stream(
	charts: pl.DataFrame, charts_root: Path, workers: int, frequencies: Counter
) -> Iterator[LayoutInput]:
	"""Every layout input of `charts` as played at rate 1, counting lane states on the way."""
	paths = [str(charts_root / p) for p in charts["urc_path"]]

	with Pool(workers) as pool:
		for lanes, states, inputs in pool.imap(_inputs, paths, chunksize=8):
			for state, n in states.items():
				frequencies[lanes, state] += n

			yield from inputs


def load_stream(
	charts: pl.DataFrame, charts_root: Path, workers: int
) -> Iterator[tuple[list[LayoutLoad], ChartLoad]]:
	"""Every chart's loads under all its layouts, as played at rate 1."""
	paths = [str(charts_root / p) for p in charts["urc_path"]]

	with Pool(workers) as pool:
		yield from pool.imap(_loads, paths, chunksize=8)


def counts_by_lanes(charts: pl.DataFrame) -> list[dict]:
	return charts.group_by("lanes", "part").len().sort("lanes", "part").to_dicts()


def state_frequencies(frequencies: Counter) -> dict[str, list[float]]:
	"""Lane state shares per lane count of the train charts, and over all of them."""
	by: dict[str, np.ndarray] = {}

	for (lanes, state), n in frequencies.items():
		for key in (str(lanes), "all"):
			by.setdefault(key, np.zeros(STATES))[state] += n

	return {key: (values / values.sum()).tolist() for key, values in sorted(by.items())}


def _has_layout(lanes: int) -> bool:
	try:
		layouts(lanes)
	except ValueError:
		return False

	return True


def _failure(path: str) -> str | None:
	"""Why a chart cannot be drawn at some rate, or why its inputs or loads are not finite."""
	chart = parse(Path(path).read_text(encoding="utf-8"))

	with np.errstate(all="ignore"):
		tokens = tokenize(chart)
		layout_loads, load = _loads_of(chart, tokens)
		arrays = [load.judgment, load.visual]

		for m, layout_ in zip(layouts(tokens.lanes), layout_loads, strict=True):
			x = layout_input(tokens, m)
			arrays += [x.token, x.lane, x.finger, x.pair, x.meta, layout_.finger, layout_.pair]

	if not all(np.isfinite(a).all() for a in arrays):
		return "non_finite"

	for rate in RATES:
		try:
			scaled = chart if rate == 1.0 else scale_rate(chart, rate).chart
		except UrcError as e:
			return e.category

		if tokenize(scaled).times.size < 2:
			return "one_token"

	return None


def _inputs(path: str) -> tuple[int, dict[int, int], list[LayoutInput]]:
	tokens = tokenize(parse(Path(path).read_text(encoding="utf-8")))
	states = np.bincount(tokens.state.ravel(), minlength=STATES)[:STATES]

	return (
		tokens.lanes,
		{s: int(n) for s, n in enumerate(states)},
		[layout_input(tokens, m) for m in layouts(tokens.lanes)],
	)


def _loads(path: str) -> tuple[list[LayoutLoad], ChartLoad]:
	chart = parse(Path(path).read_text(encoding="utf-8"))
	return _loads_of(chart, tokenize(chart))


def _loads_of(chart: Chart, tokens: ChartTokens) -> tuple[list[LayoutLoad], ChartLoad]:
	return [layout_load(tokens, m) for m in layouts(tokens.lanes)], chart_load(chart, tokens)


if __name__ == "__main__":
	typer.run(main)

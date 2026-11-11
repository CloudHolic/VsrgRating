"""Dataset manifest and the content hash that names the dataset directory."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

import polars as pl

from .config import DatasetConfig
from .query import CELLS_COPY
from .sampling import STREAMS
from .tables import RESPONSES

RULES = {
	"source": "Database views v_response_acc and v_response_score; definitions under `views`.",
	"population": "All plays of users with at least one play from a random-pool dump.",
	"mods": "Plays whose mods lie within NF TD HD SD DT HT NC PF (bitmask 17261).",
	"item": "(beatmap_id, rate_group); DT covers DT and NC, HT is its own group, else NM.",
	"score": "score / (1e6 * 0.5 if NF * 0.5 if HT).",
	"dedup": "Per (user, item) each response keeps its own best play: highest accuracy for acc, "
	"highest normalized score for score. The two may come from different plays.",
	"core": "k-core of the pool: items with at least min_item_responses responses and persons "
	"with at least min_person_responses, repeated until neither changes.",
	"items": "Drawn uniformly within each key mode of the core; key modes get items in "
	"proportion to their core counts, at least one each, or with key_items the given count "
	"per listed key mode (every eligible item where null) and none of the rest. The anchor "
	"items below are added.",
	"anchors": "Q3: (beatmapset, keys) units of the core's NM charts whose difficulty names give "
	"their order (only key_items' key modes when given), drawn in a seeded order until "
	"mapset_items charts. Q4: the dan course charts (NM), taken from the pool without the core. "
	"A unit is listed only with the charts left in the final dataset, and only if at least two "
	"remain.",
	"persons": "Persons with at least min_person_responses responses on the drawn items; "
	"n_persons of them drawn uniformly (all if null) and split into train and test (all "
	"train if test_fraction is null).",
	"final": "k-core of the drawn sample, anchor items included: items with at least "
	"min_item_train_responses train responses and persons with at least "
	"min_person_responses, repeated until neither changes.",
	"boundary": "A response is an endpoint iff it equals 0.0 or 1.0 exactly; no tolerance.",
}

LIMITATIONS = [
	"Keeping one best play per cell selects on the outcome and breaks the iid assumption.",
	"Players choose which charts to play, so missingness is not at random.",
]


def make_manifest(
	config: DatasetConfig,
	view_definitions: dict[str, str],
	ingest_log: list[dict[str, Any]],
	eligible: dict[int, int],
	allocation: dict[int, int],
	stages: dict[str, dict[str, int]],
	anchors: dict[str, Any],
	items: pl.DataFrame,
	persons: pl.DataFrame,
	responses: pl.DataFrame,
) -> dict[str, Any]:
	"""The hashed part of the manifest: everything but the hash and build time."""
	split = responses.join(persons.select("person_idx", "split"), on="person_idx")
	train = split.filter(pl.col("split") == "train")
	kept = dict(items.group_by("keys").len().iter_rows())
	return {
		# Without key_items the config reads as it did before the field existed.
		"config": {k: v for k, v in asdict(config).items() if k != "key_items" or v is not None},
		"seed_streams": list(STREAMS),
		"rules": RULES,
		"query": CELLS_COPY,
		"views": view_definitions,
		"ingest_log": ingest_log,
		"keys": {
			str(k): {"core": eligible[k], "allocated": n, "kept": kept.get(k, 0)}
			for k, n in sorted(allocation.items())
		},
		"anchors": anchors,
		"counts": {
			"stages": stages,
			"items": items.height,
			"persons": {s: persons.filter(pl.col("split") == s).height for s in ("train", "test")},
			"responses": {s: split.filter(pl.col("split") == s).height for s in ("train", "test")},
		},
		"train_endpoint_fraction": {
			name: {
				"zero": (train[name] == 0.0).mean(),
				"one": (train[name] == 1.0).mean(),
			}
			for name in RESPONSES
		},
		"limitations": LIMITATIONS,
	}


def content_hash(tables: dict[str, pl.DataFrame], core: dict[str, Any]) -> str:
	"""sha256 over column values and the canonical manifest, independent of parquet encoding."""
	h = hashlib.sha256()
	for name in sorted(tables):
		table = tables[name]
		for col in sorted(table.columns):
			h.update(f"{name}.{col}\0".encode())
			series = table[col]
			if series.dtype == pl.String:
				h.update("\0".join(series.to_list()).encode())
			elif series.dtype.is_integer():
				h.update(series.cast(pl.Int64).to_numpy().astype("<i8").tobytes())
			elif series.dtype.is_float():
				h.update(series.cast(pl.Float64).to_numpy().astype("<f8").tobytes())
			else:
				raise TypeError(f"cannot hash {name}.{col} of type {series.dtype}")
	h.update(json.dumps(core, sort_keys=True, ensure_ascii=False).encode())
	return h.hexdigest()[:12]

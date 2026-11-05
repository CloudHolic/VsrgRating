"""Reads one split of a built dataset for fitting."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .tables import RESPONSES

SPLITS = ("train", "test")


@dataclass(frozen=True)
class Split:
	"""The cells of one split and one response variable. `person` indexes the split's persons."""

	person: np.ndarray
	item: np.ndarray
	x: np.ndarray
	n_persons: int
	n_items: int
	person_idx: np.ndarray
	dataset_hash: str


def locate_dataset(root: Path, name: str) -> Path:
	"""The one built dataset `root/name/<hash>`; none or several is an error."""
	base = root / name
	found = sorted(p for p in base.glob("*") if p.is_dir() and not p.name.startswith("."))
	if not found:
		raise FileNotFoundError(f"no built dataset under {base}")
	if len(found) > 1:
		raise ValueError(f"several datasets under {base}: {', '.join(p.name for p in found)}")

	return found[0]


def load_split(path: Path, response: str, split: str) -> Split:
	"""Loads one split; the test persons never enter a fit and are what held-out scores use."""
	if response not in RESPONSES:
		raise ValueError(f"response must be one of {RESPONSES}, got {response!r}")
	if split not in SPLITS:
		raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

	manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
	if manifest["hash"] != path.name:
		raise ValueError(f"manifest hash {manifest['hash']} does not match directory {path.name}")

	persons = (
		pl.read_parquet(path / "persons.parquet")
		.filter(pl.col("split") == split)
		.sort("person_idx")
		.with_row_index("p")
		.with_columns(pl.col("p").cast(pl.Int64))
	)
	if persons.height == 0:
		raise ValueError(f"{path.name} has no {split} persons")

	cells = (
		pl.read_parquet(path / "responses.parquet")
		.join(persons.select("p", "person_idx"), on="person_idx")
		.sort("p", "item_idx")
	)

	return Split(
		person=cells["p"].to_numpy(),
		item=cells["item_idx"].to_numpy(),
		x=cells[response].to_numpy(),
		n_persons=persons.height,
		n_items=manifest["counts"]["items"],
		person_idx=persons["person_idx"].to_numpy(),
		dataset_hash=manifest["hash"],
	)


def split_persons(split: Split, fraction: float, seed: int) -> tuple[Split, Split]:
	"""(kept, held out): a random `fraction` of the persons and all their cells held out."""
	if not 0.0 < fraction < 1.0:
		raise ValueError(f"fraction must be in (0, 1), got {fraction}")

	rng = np.random.default_rng(seed)
	held = np.zeros(split.n_persons, dtype=bool)
	held[rng.choice(split.n_persons, size=round(fraction * split.n_persons), replace=False)] = True

	return subset(split, ~held[split.person]), subset(split, held[split.person])


def subset(split: Split, cells: np.ndarray) -> Split:
	"""The cells under the mask; persons left without cells drop out, the rest keep their order."""
	kept, person = np.unique(split.person[cells], return_inverse=True)

	return Split(
		person=person,
		item=split.item[cells],
		x=split.x[cells],
		n_persons=kept.size,
		n_items=split.n_items,
		person_idx=split.person_idx[kept],
		dataset_hash=split.dataset_hash,
	)

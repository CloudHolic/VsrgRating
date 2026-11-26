"""Each person's predictive log likelihood of held-out cells given their fitting cells,
log p(Y_T, Y_V) - log p(Y_T), and its mapset-block jackknife."""

import hashlib
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
from jax import Array

from ..data import Split
from ..irt.cnrm import model as cnrm
from ..irt.map import Model, person_summary
from .fitted import Fitted, read_transform

BLOCK_SALT = "vsrgrating-jackknife-2026-11-25"

type CellArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


def predictive_gain(fitted: Fitted, base: Split, valid: Split, n_nodes: int) -> np.ndarray:
	"""log p(Y_T, Y_V) - log p(Y_T) per person, T the `base` cells and V the `valid` cells."""
	_check_pair(base, valid)
	gain = np.zeros(base.n_persons)
	persons = np.unique(valid.person)
	if persons.size:
		gain[persons] = _gain_of(fitted, base, valid, persons, n_nodes)

	return gain


def posterior_splits(
	cells: pl.DataFrame, items: np.ndarray, n_persons: int, dataset_hash: str
) -> tuple[Split, Split]:
	"""T (the `in_t` cells of a fixed posterior run's cells.parquet) and V (every cell of `items`),
	both over all `n_persons` dataset persons."""
	held = cells.filter(pl.col("item_idx").is_in(pl.Series(items).implode()))
	if held["in_t"].any():
		raise ValueError("some evaluated items have cells in T")
	n_items = int(cells["item_idx"].to_numpy().max()) + 1

	def split(frame: pl.DataFrame) -> Split:
		return Split(
			person=frame["person_idx"].to_numpy(),
			item=frame["item_idx"].to_numpy(),
			x=frame["x"].to_numpy(),
			n_persons=n_persons,
			n_items=n_items,
			person_idx=np.arange(n_persons),
			dataset_hash=dataset_hash,
		)

	return split(cells.filter(pl.col("in_t"))), split(held)


def posterior_fitted(run: Path, psi_v: pl.DataFrame, n_items: int) -> Fitted:
	"""The fixed posterior run's G and psi_T, with `psi_v` (item_idx, alpha_1..D, beta, sigma) at
	the evaluated items."""
	psi_t = pl.read_parquet(run / "items.parquet")
	if not set(psi_t["item_idx"]).isdisjoint(psi_v["item_idx"]):
		raise ValueError("the evaluated items overlap T")

	names = [c for c in psi_t.columns if c.startswith("alpha_")]
	table = pl.concat([psi.select("item_idx", *names, "beta", "sigma") for psi in (psi_t, psi_v)])
	idx = table["item_idx"].to_numpy()
	alpha = np.zeros((n_items, len(names)))
	beta, sigma = np.zeros(n_items), np.ones(n_items)
	alpha[idx], beta[idx], sigma[idx] = (
		table.select(names).to_numpy(),
		table["beta"].to_numpy(),
		table["sigma"].to_numpy(),
	)
	spec, raw = read_transform(run)
	item = cnrm.ItemParams(
		jnp.asarray(alpha[:, 0] if len(names) == 1 else alpha),
		jnp.asarray(beta),
		jnp.asarray(sigma),
	)

	return Fitted(run.name, Model("cnrm", spec, len(names)), item, raw, "map")


def mapset_blocks(beatmapset_id: np.ndarray, keys: np.ndarray, n_blocks: int) -> np.ndarray:
	"""Each row's jackknife block."""
	set_keys: dict[int, set[int]] = {}
	for set_id, key in zip(beatmapset_id.tolist(), keys.tolist(), strict=True):
		set_keys.setdefault(set_id, set()).add(key)

	sets_with_key = Counter(key for found in set_keys.values() for key in found)
	strata: dict[int, list[int]] = {}
	for set_id, found in set_keys.items():
		stratum = min(found, key=lambda key: (sets_with_key[key], key))
		strata.setdefault(stratum, []).append(set_id)

	block_of: dict[int, int] = {}
	dealt = 0
	for stratum in sorted(strata):
		for set_id in sorted(strata[stratum], key=_block_order):
			block_of[set_id] = dealt % n_blocks
			dealt += 1

	return np.array([block_of[s] for s in beatmapset_id.tolist()], dtype=np.int64)


def jackknife_totals(
	fitted: Fitted, base: Split, valid: Split, block: np.ndarray, n_blocks: int, n_nodes: int
) -> np.ndarray:
	"""T_{-g}: the summed predictive_gain with block g's V cells removed, for every block g."""
	_check_pair(base, valid)
	if block.shape != valid.x.shape or block.min(initial=0) < 0 or block.max(initial=0) >= n_blocks:
		raise ValueError(f"block must give each valid cell a block in 0..{n_blocks - 1}")

	full = predictive_gain(fitted, base, valid, n_nodes)
	total = full.sum()
	touched = [np.unique(valid.person[block == g]) for g in range(n_blocks)]
	size = _shared_size(base, valid, block, touched)

	totals = np.full(n_blocks, total)
	for g, persons in enumerate(touched):
		if persons.size == 0:
			continue
		kept = _cells(valid, block != g)
		new = _gain_of(fitted, base, kept, persons, n_nodes, size)
		totals[g] = total - full[persons].sum() + new.sum()

	return totals


def _gain_of(
	fitted: Fitted,
	base: Split,
	valid: Split,
	persons: np.ndarray,
	n_nodes: int,
	size: tuple[int, int, int] | None = None,
) -> np.ndarray:
	"""The gain of `persons` only. `size` pads (persons, joint cells, base cells) so that every
	call shares one shape."""
	local = np.full(base.n_persons, -1)
	local[persons] = np.arange(persons.size)
	b, v = local[base.person], local[valid.person]

	alone: CellArrays = (b[b >= 0], base.item[b >= 0], base.x[b >= 0])
	extra: CellArrays = (v[v >= 0], valid.item[v >= 0], valid.x[v >= 0])
	joint: CellArrays = (
		np.concatenate([alone[0], extra[0]]),
		np.concatenate([alone[1], extra[1]]),
		np.concatenate([alone[2], extra[2]]),
	)
	n = persons.size

	if size is not None:
		n = size[0]
		joint, alone = _pad(joint, size[1], n - 1), _pad(alone, size[2], n - 1)

	params = _params(fitted)
	with_v = person_summary(fitted.model, params, *joint, n, n_nodes)[0]
	without_v = person_summary(fitted.model, params, *alone, n, n_nodes)[0]

	return np.asarray(with_v - without_v)[: persons.size]


def _shared_size(
	base: Split, valid: Split, block: np.ndarray, touched: list[np.ndarray]
) -> tuple[int, int, int]:
	"""The largest (persons + 1, joint cells, base cells) over the blocks' recomputations."""
	base_n = np.bincount(base.person, minlength=base.n_persons)
	valid_n = np.bincount(valid.person, minlength=base.n_persons)
	persons = joint = alone = 0

	for g, ids in enumerate(touched):
		removed = np.bincount(valid.person[block == g], minlength=base.n_persons)
		persons = max(persons, ids.size)
		alone = max(alone, int(base_n[ids].sum()))
		joint = max(joint, int((base_n[ids] + valid_n[ids] - removed[ids]).sum()))

	return persons + 1, joint, alone


def _pad(cells: CellArrays, length: int, person: int) -> CellArrays:
	person_col, item, x = cells
	extra = length - x.size

	return (
		np.concatenate([person_col, np.full(extra, person)]),
		np.concatenate([item, np.zeros(extra, dtype=item.dtype)]),
		np.concatenate([x, np.full(extra, 0.5)]),
	)


def _cells(split: Split, mask: np.ndarray) -> Split:
	return Split(
		person=split.person[mask],
		item=split.item[mask],
		x=split.x[mask],
		n_persons=split.n_persons,
		n_items=split.n_items,
		person_idx=split.person_idx,
		dataset_hash=split.dataset_hash,
	)


def _check_pair(base: Split, valid: Split) -> None:
	for name in ("n_persons", "n_items", "dataset_hash"):
		if getattr(base, name) != getattr(valid, name):
			raise ValueError(f"base and valid differ in {name}")

	for split in (base, valid):
		if not np.array_equal(split.person_idx, np.arange(split.n_persons)):
			raise ValueError("the splits must index every dataset person by person_idx")


def _params(fitted: Fitted) -> dict[str, Array]:
	item = fitted.item
	assert isinstance(item, cnrm.ItemParams) and fitted.raw is not None

	return {
		"alpha": item.alpha,
		"beta": item.beta,
		"log_sigma": jnp.log(item.sigma),
		"transform": fitted.raw,
	}


def _block_order(set_id: int) -> bytes:
	return hashlib.blake2b(f"{BLOCK_SALT}:{set_id}".encode(), digest_size=8).digest()

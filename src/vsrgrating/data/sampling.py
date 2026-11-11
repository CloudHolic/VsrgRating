"""Seeded draws: key-mode allocation, items within key modes, persons, and the person split."""

import numpy as np
import polars as pl

STREAMS = ("items", "persons", "split", "units")


def spawn_rngs(seed: int) -> dict[str, np.random.Generator]:
	"""One generator per stage, so changing one stage leaves the other stages' draws intact."""
	children = np.random.SeedSequence(seed).spawn(len(STREAMS))
	return {
		name: np.random.default_rng(child) for name, child in zip(STREAMS, children, strict=True)
	}


def allocate(eligible: dict[int, int], n_items: int) -> dict[int, int]:
	"""Splits n_items across key modes in proportion to their eligible item counts."""
	keys = sorted(k for k, count in eligible.items() if count > 0)
	if len(keys) > n_items:
		raise ValueError(f"{len(keys)} key modes cannot each get an item out of {n_items}")

	total = sum(eligible[k] for k in keys)
	if total < n_items:
		raise ValueError(f"only {total} eligible items for {n_items} requested")

	quotas = {k: n_items * eligible[k] / total for k in keys}
	alloc = {k: int(quotas[k]) for k in keys}
	shortfall = n_items - sum(alloc.values())
	by_remainder = sorted(keys, key=lambda k: (-(quotas[k] - alloc[k]), k))

	for k in by_remainder[:shortfall]:
		alloc[k] += 1

	for k in keys:
		if alloc[k] == 0:
			donor = max(keys, key=lambda d: (alloc[d], -d))
			alloc[donor] -= 1
			alloc[k] = 1

	return alloc


def allocate_keys(eligible: dict[int, int], key_items: dict[int, int | None]) -> dict[int, int]:
	"""The requested count per key mode, every eligible item where the request is None."""
	alloc = {}
	for k, wanted in sorted(key_items.items()):
		available = eligible.get(k, 0)
		count = available if wanted is None else wanted

		if count < 1 or count > available:
			raise ValueError(f"{k}K: {wanted} requested, {available} eligible")

		alloc[k] = count

	return alloc


def sample_items(
	eligible: pl.DataFrame, allocation: dict[int, int], rng: np.random.Generator
) -> pl.DataFrame:
	"""Draws allocation[k] items uniformly without replacement within each key mode."""
	picked = []

	for k in sorted(allocation):
		pool = eligible.filter(pl.col("keys") == k)
		rows = np.sort(rng.choice(pool.height, size=allocation[k], replace=False))
		picked.append(pool[rows])

	return pl.concat(picked).sort("keys", "beatmap_id", "rate_group")


def subsample_persons(
	user_ids: np.ndarray, n_persons: int | None, rng: np.random.Generator
) -> np.ndarray:
	"""Uniform subsample of sorted unique user ids; all of them when n_persons is None."""
	if n_persons is None:
		return user_ids
	if n_persons > len(user_ids):
		raise ValueError(f"n_persons={n_persons} exceeds the {len(user_ids)} available")

	return np.sort(rng.choice(user_ids, size=n_persons, replace=False))


def split_persons(n: int, test_fraction: float, rng: np.random.Generator) -> np.ndarray:
	"""Boolean mask over n persons, True for the test split."""
	test = np.zeros(n, dtype=bool)
	test[rng.permutation(n)[: round(n * test_fraction)]] = True
	return test


def sample_units(units: pl.DataFrame, target_items: int, rng: np.random.Generator) -> pl.DataFrame:
	"""Units in a seeded order, taken until their charts reach target_items."""
	ordered = units.sort("beatmapset_id", "keys")[rng.permutation(units.height)]
	size = pl.col("order").list.len()

	return (
		ordered.with_columns(size.cum_sum().alias("reached"))
		.filter(pl.col("reached") - size < target_items)
		.drop("reached")
	)

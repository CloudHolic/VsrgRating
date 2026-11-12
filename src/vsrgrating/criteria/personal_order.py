"""Q5': what reading one number costs each player. On charts held back from a player's ability
estimate, does the player's own predicted order follow their results better than d's order?"""

import jax.numpy as jnp
import numpy as np
import polars as pl

from ..data import Split, subset
from ..evaluation import Fitted, person_posterior
from ..irt.cnrm.model import ItemParams
from ..readout import cnrm
from . import simulate

BOUNDARY = 0.95
MIN_RESPONSES = 10
SEED = 20260916
NULL_DRAWS = 10
BOOTSTRAP = 1000


def table(fitted: Fitted, test: Split, keys: np.ndarray) -> pl.DataFrame:
	"""Both orders' agreement with the held-back results, for everyone and across key modes."""
	if not isinstance(fitted.item, ItemParams):
		raise ValueError("Q5' reads the cnrm only")

	assert fitted.model.transform is not None and fitted.raw is not None

	rng = np.random.default_rng(SEED)
	estimate, held = halves(rng, test)
	known, cells = subset(test, estimate), subset(test, held)

	if not np.array_equal(known.person_idx, cells.person_idx):
		raise ValueError("the two halves hold different people")

	theta = person_posterior(fitted, known).mean
	d = cnrm.difficulty(fitted.item, fitted.model.transform, fitted.raw, BOUNDARY)
	easiness = -d[cells.item]
	personal = margin(fitted, cells, theta)
	modes = keys[cells.item]

	observed = per_person(cells.person, cells.x, easiness, personal, modes)
	nulls = [
		per_person(
			cells.person,
			simulate.draw(rng, fitted, cells.item, theta[cells.person]),
			easiness,
			personal,
			modes,
		)
		for _ in range(NULL_DRAWS)
	]
	groups = (("all", pl.lit(True)), ("mixed_keys", pl.col("modes") >= 2))

	return pl.DataFrame([summary(name, keep, observed, nulls, rng) for name, keep in groups])


def halves(rng: np.random.Generator, data: Split) -> tuple[np.ndarray, np.ndarray]:
	"""Each eligible person's cells split at random into an estimating and a held-back half."""
	counts = np.bincount(data.person, minlength=data.n_persons)
	starts = np.cumsum(counts) - counts
	order = np.lexsort((rng.random(data.person.size), data.person))

	seat = np.empty(data.person.size, dtype=np.int64)
	seat[order] = np.arange(data.person.size) - starts[data.person[order]]

	eligible = counts[data.person] >= MIN_RESPONSES
	first = seat < counts[data.person] // 2

	return eligible & first, eligible & ~first


def margin(fitted: Fitted, cells: Split, theta: np.ndarray) -> np.ndarray:
	"""(alpha . theta + beta - G(y)) / sigma for each cell: 
	in the same order as P(Y >= y | theta)."""
	assert fitted.model.transform is not None and fitted.raw is not None

	g, _ = fitted.model.transform.apply(fitted.raw, jnp.asarray(BOUNDARY))
	alpha, beta, sigma = (np.asarray(c)[cells.item] for c in fitted.item)
	ability = theta[cells.person]
	load = (alpha * ability).sum(axis=-1) if alpha.ndim == 2 else alpha * ability

	return (load + beta - float(g)) / sigma


def per_person(
	person: np.ndarray,
	x: np.ndarray,
	easiness: np.ndarray,
	personal: np.ndarray,
	modes: np.ndarray,
) -> pl.DataFrame:
	"""Each person's Spearman of results with d's order and with their own, and their key modes."""
	return (
		pl.DataFrame(
			{"person": person, "x": x, "easiness": easiness, "personal": personal, "modes": modes}
		)
		.group_by("person")
		.agg(
			pl.corr("x", "easiness", method="spearman").alias("by_d"),
			pl.corr("x", "personal", method="spearman").alias("personal"),
			pl.col("modes").n_unique(),
		)
		.sort("person")
	)


def summary(
	name: str,
	keep: pl.Expr,
	observed: pl.DataFrame,
	nulls: list[pl.DataFrame],
	rng: np.random.Generator,
) -> dict[str, str | float]:
	"""Both medians, the mean personal gain with a person bootstrap SE,
	and the same under the null."""
	seen = readable(observed.filter(keep))
	gain = (seen["personal"] - seen["by_d"]).to_numpy()
	boot = rng.choice(gain, (BOOTSTRAP, gain.size)).mean(axis=1)
	drawn = [readable(null.filter(keep)) for null in nulls]

	return {
		"people": name,
		"persons": float(seen.height),
		"by_d": float(np.median(seen["by_d"].to_numpy())),
		"personal": float(np.median(seen["personal"].to_numpy())),
		"gain": float(gain.mean()),
		"gain_se": float(boot.std()),
		"null_by_d": float(np.mean([np.median(t["by_d"].to_numpy()) for t in drawn])),
		"null_personal": float(np.mean([np.median(t["personal"].to_numpy()) for t in drawn])),
		"null_gain": float(np.mean([(t["personal"] - t["by_d"]).to_numpy().mean() for t in drawn])),
	}


def readable(people: pl.DataFrame) -> pl.DataFrame:
	"""People whose results are not all tied, so both rank correlations exist."""
	return people.filter(pl.col("by_d").is_not_nan() & pl.col("personal").is_not_nan())

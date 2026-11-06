"""Q5: do the players' own results follow the difficulty order?"""

import numpy as np
import polars as pl
from scipy.stats import rankdata

from ..data import Split, subset
from ..evaluation import Fitted, person_posterior
from . import readouts, simulate

SET_SIZE = 9
BANDS = 3
MIN_ITEMS = 6
NULL_DRAWS = 200
NULL_SEED = 20260916
GAP_QUANTILES = (0.25, 0.5, 0.75)
BEAM_WIDTH = 20


def common_set(acc: Split) -> tuple[np.ndarray, np.ndarray]:
	"""Charts that one group of people all played, and that group."""
	counts = np.bincount(acc.item, minlength=acc.n_items)
	total = np.bincount(acc.item, weights=acc.x, minlength=acc.n_items)
	mean = np.divide(total, counts, out=np.zeros(acc.n_items), where=counts > 0)
	edges = np.quantile(mean[counts > 0], np.linspace(0.0, 1.0, BANDS + 1))
	band = np.clip(np.digitize(mean, edges[1:-1]), 0, BANDS - 1)

	beams: list[tuple[tuple[int, ...], np.ndarray]] = [((), np.ones(acc.n_persons, dtype=bool))]
	for step in range(SET_SIZE):
		beams = _extend(acc, beams, band == step % BANDS)
		if not beams:
			raise ValueError(f"no chart left in band {step % BANDS} that anyone shares")

	chosen, alive = beams[0]

	return np.array(chosen), np.flatnonzero(alive)


def _extend(
	acc: Split, beams: list[tuple[tuple[int, ...], np.ndarray]], allowed: np.ndarray
) -> list[tuple[tuple[int, ...], np.ndarray]]:
	"""The BEAM_WIDTH largest groups one allowed chart further on."""
	grown: dict[tuple[int, ...], tuple[int, int, np.ndarray]] = {}

	for chosen, alive in beams:
		shared = np.bincount(acc.item[alive[acc.person]], minlength=acc.n_items).astype(np.int64)
		shared[list(chosen)] = 0
		shared[~allowed] = 0

		for item in np.argsort(-shared, kind="stable")[:BEAM_WIDTH].tolist():
			if shared[item] > 0:
				grown[tuple(sorted((*chosen, item)))] = (int(shared[item]), item, alive)

	best = sorted(grown.items(), key=lambda entry: (-entry[1][0], entry[0]))[:BEAM_WIDTH]

	beams = []
	for chosen, (_, item, alive) in best:
		members = np.zeros(acc.n_persons, dtype=bool)
		members[acc.person[acc.item == item]] = True
		beams.append((chosen, alive & members))

	return beams


def left_out(data: Split, persons: np.ndarray) -> Split:
	"""The responses of everyone but `persons`."""
	keep = np.ones(data.n_persons, dtype=bool)
	keep[persons] = False

	return subset(data, keep[data.person])


def evaluate(
	fitted: Fitted,
	data: Split,
	rest: Split,
	items: np.ndarray,
	persons: np.ndarray,
	grid: tuple[float, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float]]:
	"""Agreement against the null per candidate, per gap band, and how much was read."""
	values, domain = readouts.of(fitted, grid)
	usable = domain & readouts.seen(rest)

	kept = items[usable[items]]
	if kept.size < MIN_ITEMS:
		raise ValueError(f"only {kept.size} of {items.size} charts can be read")

	table = results_on(data, kept, persons)
	observed = agreement(values, kept, table)
	names = list(observed)

	flipped = {name: discordance(values[name][kept], table)[0] for name in names}
	gap = discordance(values[names[0]][kept], table)[1]
	edges = np.quantile(gap[gap > 0.0], GAP_QUANTILES)
	bands = {name: band_rates(flipped[name], gap, edges) for name in names}
	weights = {name: weighted_rate(flipped[name], gap) for name in names}

	theta = abilities(fitted, data, items, persons)
	null, null_bands, null_weights = _null(fitted, values, kept, theta, edges)

	summary = pl.DataFrame(
		[
			{
				"candidate": name,
				"median": float(np.median(observed[name])),
				"q1": float(np.quantile(observed[name], 0.25)),
				"q3": float(np.quantile(observed[name], 0.75)),
				"as_expected": float(np.mean(observed[name] > 0.0)),
				"null_median": float(np.median(null[name])),
				"observed_minus_null": float(np.median(observed[name]) - np.median(null[name])),
				"weighted": float(np.nanmedian(weights[name])),
				"null_weighted": float(np.median(null_weights[name])),
			}
			for name in names
		]
	)

	limits = np.concatenate([[0.0], edges, [np.inf]])
	band_table = pl.DataFrame(
		[
			{
				"candidate": name,
				"band": index,
				"low": float(limits[index]),
				"high": float(limits[index + 1]),
				"rate": float(np.nanmedian(bands[name][:, index])),
				"null_rate": float(np.median(null_bands[name][:, index])),
				"people": int(np.isfinite(bands[name][:, index]).sum()),
			}
			for name in names
			for index in range(edges.size + 1)
		]
	)

	counts = {
		"q5_charts": float(kept.size),
		"q5_charts_unreadable": float(items.size - kept.size),
		"q5_persons": float(persons.size),
		"q5_persons_with_ability": float(np.isfinite(theta).sum()),
	}

	return summary, band_table, counts


def results_on(data: Split, items: np.ndarray, persons: np.ndarray) -> np.ndarray:
	"""Every person's response to every chart of the set, people by charts."""
	seat = np.full(data.n_persons, -1)
	seat[persons] = np.arange(persons.size)
	column = np.full(data.n_items, -1)
	column[items] = np.arange(items.size)

	rows = (seat[data.person] >= 0) & (column[data.item] >= 0)
	table = np.full((persons.size, items.size), np.nan)
	table[seat[data.person[rows]], column[data.item[rows]]] = data.x[rows]

	return table


def agreement(
	values: dict[str, np.ndarray], items: np.ndarray, table: np.ndarray
) -> dict[str, np.ndarray]:
	"""How far each person's own results follow each candidate's order, sign flipped."""
	ranks = rankdata(table, axis=1)
	ranks = ranks - ranks.mean(axis=1, keepdims=True)
	norm = np.sqrt((ranks**2).sum(axis=1))

	result = {}
	for name, value in values.items():
		order = rankdata(value[items])
		order = order - order.mean()

		with np.errstate(divide="ignore", invalid="ignore"):
			result[name] = -(ranks @ order) / (norm * np.sqrt((order**2).sum()))

	return result


def discordance(difficulty: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Which pairs each person's own results put the other way round, and how far apart they are."""
	first, second = np.triu_indices(table.shape[1], k=1)
	delta = table[:, first] - table[:, second]
	harder = np.sign(difficulty[first] - difficulty[second])

	return (np.sign(delta) == harder) & (delta != 0.0), np.abs(delta)


def band_rates(flipped: np.ndarray, gap: np.ndarray, edges: np.ndarray) -> np.ndarray:
	"""Each person's discordance inside each gap band, people by bands."""
	band = np.digitize(gap, edges)
	rates = np.full((flipped.shape[0], edges.size + 1), np.nan)

	for index in range(edges.size + 1):
		inside = (band == index) & (gap > 0.0)
		held = inside.sum(axis=1)
		rates[:, index] = np.where(
			held > 0, (flipped & inside).sum(axis=1) / np.maximum(held, 1), np.nan
		)

	return rates


def weighted_rate(flipped: np.ndarray, gap: np.ndarray) -> np.ndarray:
	"""Each person's discordance weighted by how far their own results separated the charts."""
	total = gap.sum(axis=1)
	return np.where(total > 0.0, (gap * flipped).sum(axis=1) / np.maximum(total, 1e-12), np.nan)


def abilities(fitted: Fitted, data: Split, items: np.ndarray, persons: np.ndarray) -> np.ndarray:
	"""Each person's posterior mean from responses outside the set, so the null sees no leak;
	NaN for a person with none."""
	inside = np.zeros(data.n_items, dtype=bool)
	inside[items] = True

	chosen = np.zeros(data.n_persons, dtype=bool)
	chosen[persons] = True

	cells = chosen[data.person] & ~inside[data.item]
	measured = np.unique(data.person[cells])
	theta = np.full(data.n_persons, np.nan)
	theta[measured] = person_posterior(fitted, subset(data, cells)).mean

	return theta[persons]


def _null(
	fitted: Fitted,
	values: dict[str, np.ndarray],
	items: np.ndarray,
	theta: np.ndarray,
	edges: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
	"""The same statistics when the model is exactly right, per draw."""
	people = theta[np.isfinite(theta)]
	item = np.tile(items, NULL_DRAWS * people.size)
	ability = np.repeat(np.tile(people, NULL_DRAWS), items.size)
	rng = np.random.default_rng(NULL_SEED)
	drawn = simulate.draw(rng, fitted, item, ability).reshape(NULL_DRAWS, people.size, items.size)

	rho = {name: np.empty(NULL_DRAWS) for name in values}
	bands = {name: np.empty((NULL_DRAWS, edges.size + 1)) for name in values}
	weights = {name: np.empty(NULL_DRAWS) for name in values}

	for draw, table in enumerate(drawn):
		for name, rank in agreement(values, items, table).items():
			flipped, gap = discordance(values[name][items], table)
			rho[name][draw] = float(np.nanmedian(rank))
			bands[name][draw] = np.nanmedian(band_rates(flipped, gap, edges), axis=0)
			weights[name][draw] = float(np.nanmedian(weighted_rate(flipped, gap)))

	return rho, bands, weights

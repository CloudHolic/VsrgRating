"""Whether ability is one dimension across key modes: the latent correlation of thetas fitted on
two key modes apart, read against the same correlation between random halves of one key mode.

uv run scripts/ability_dimension.py +experiment=random3000_ability_dimension
"""

import math
from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import polars as pl
import typer

from vsrgrating.config import compose_config
from vsrgrating.data import Split, load_split, locate_dataset, subset
from vsrgrating.evaluation import latent_correlation, person_grid, rho_se
from vsrgrating.fitting import Refits, build_model, map_settings, refit, run_tags
from vsrgrating.tracking import run_output, write_metrics

GRID = np.linspace(-8.0, 8.0, 801)
MIN_RESPONSES = (5, 10, 20)
JUDGED = 10
MIN_PERSONS = 30
GAP = 0.05

# Each comparison correlates two parts over the persons with enough responses in the two parts
# named last. The halves are read on the persons of the 4K-7K comparison, so that the baseline and
# the comparison it judges share one range of ability.
COMPARISONS: dict[str, tuple[str, str, tuple[str, str]]] = {
	"4K-7K": ("k4", "k7", ("k4", "k7")),
	"4K-6K": ("k4", "k6", ("k4", "k6")),
	"4Ka-4Kb": ("k4-a", "k4-b", ("k4", "k7")),
	"7Ka-7Kb": ("k7-a", "k7-b", ("k4", "k7")),
}
COLUMNS = ("rho", "se", "mean_1", "mean_2", "sd_1", "sd_2", "loglik")


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("ability_dimension", overrides)
	jax.config.update("jax_enable_x64", True)

	if cfg.tracking.mlflow:
		raise ValueError("the parts are read back from disk: use tracking=local")

	refits = Refits(cfg, overrides, build_model(cfg), map_settings(cfg))
	dataset = locate_dataset(Path(cfg.dataset_root), cfg.dataset.name)
	keys = pl.read_parquet(dataset / "items.parquet").sort("item_idx")["keys"].to_numpy()
	pending = [r for r in cfg.responses if not (Path(cfg.runs_root) / cfg.name / r).exists()]

	for response in pending:
		data = load_split(dataset, response, "train")
		masks = part_masks(data, keys, cfg.split_seed)
		counts = {
			part: np.bincount(data.person[m], minlength=data.n_persons) for part, m in masks.items()
		}
		grids = {part: person_rows(refits, response, part, data, m) for part, m in masks.items()}
		table = correlation_table(response, grids, counts)

		with run_output(cfg, overrides, response) as out:
			table.write_csv(out / "correlation.csv")
			write_metrics(out, run_tags(cfg, refits.model, data, response), judgement(table), {})


def part_masks(data: Split, keys: np.ndarray, seed: int) -> dict[str, np.ndarray]:
	"""Cell masks of each part: three key modes, and random halves of the 4K and 7K items."""
	key = keys[data.item]
	masks = {f"k{k}": key == k for k in (4, 7, 6)}

	rng = np.random.default_rng(seed)
	for k in (4, 7):
		items = rng.permutation(np.flatnonzero(keys == k))
		first = np.zeros(keys.size, dtype=bool)
		first[items[: items.size // 2]] = True
		masks[f"k{k}-a"] = (key == k) & first[data.item]
		masks[f"k{k}-b"] = (key == k) & ~first[data.item]

	return masks


def person_rows(
	refits: Refits, response: str, part: str, data: Split, mask: np.ndarray
) -> np.ndarray:
	"""Every person's log likelihood on GRID under the part's own fit, flat (zero) for a person
	with no responses in the part. Rows follow the persons of `data`."""
	split = subset(data, mask)
	fitted = refit(refits, response, part, split)

	rows = np.zeros((data.n_persons, GRID.size))
	rows[np.searchsorted(data.person_idx, split.person_idx)] = person_grid(fitted, split, GRID)

	return rows


def correlation_table(
	response: str, grids: dict[str, np.ndarray], counts: dict[str, np.ndarray]
) -> pl.DataFrame:
	"""One row per comparison and threshold; too few persons leaves the estimates empty."""
	rows = []
	for name, (first, second, (left, right)) in COMPARISONS.items():
		for least in MIN_RESPONSES:
			chosen = (counts[left] >= least) & (counts[right] >= least)
			row: dict[str, object] = {
				"comparison": name,
				"min_responses": least,
				"persons": int(chosen.sum()),
				**dict.fromkeys(COLUMNS),
			}

			if chosen.sum() >= MIN_PERSONS:
				a, b = grids[first][chosen], grids[second][chosen]
				fit = latent_correlation(a, b, GRID)
				row |= {
					"rho": fit.rho,
					"se": rho_se(a, b, GRID),
					"mean_1": fit.mean[0],
					"mean_2": fit.mean[1],
					"sd_1": fit.sd[0],
					"sd_2": fit.sd[1],
					"loglik": fit.loglik,
				}

			print(
				f"{response} {name} >= {least}: {row['persons']} persons, rho {row['rho']}",
				flush=True,
			)
			rows.append(row)

	return pl.DataFrame(rows)


def judgement(table: pl.DataFrame) -> dict[str, float]:
	"""The pre-registered rule at JUDGED responses: one dimension unless the cross-key correlation
	falls short of the lower within-key one by more than GAP and by more than two SEs."""
	at = table.filter(pl.col("min_responses") == JUDGED)

	def read(name: str, column: str) -> float:
		value = at.filter(pl.col("comparison") == name)[column][0]
		if value is None:
			raise ValueError(f"{name} has fewer than {MIN_PERSONS} persons at {JUDGED} responses")
		return float(value)

	within = min(("4Ka-4Kb", "7Ka-7Kb"), key=lambda name: read(name, "rho"))
	gap = read(within, "rho") - read("4K-7K", "rho")
	gap_se = math.hypot(read(within, "se"), read("4K-7K", "se"))

	return {
		"rho_cross": read("4K-7K", "rho"),
		"rho_within_4k": read("4Ka-4Kb", "rho"),
		"rho_within_7k": read("7Ka-7Kb", "rho"),
		"gap": gap,
		"gap_se": gap_se,
		"one_dimension": float(not (gap > GAP and gap > 2.0 * gap_se)),
	}


if __name__ == "__main__":
	typer.run(main)

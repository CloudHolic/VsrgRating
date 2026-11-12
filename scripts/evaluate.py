"""Compare finished fits on held-out persons and near the ceiling, one response at a time.

uv run scripts/evaluate.py +experiment=core300_family
"""

from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import polars as pl
import typer
from omegaconf import DictConfig

from vsrgrating.config import compose_config
from vsrgrating.data import Split, load_split, locate_dataset, subset
from vsrgrating.evaluation import (
	Fitted,
	cell_predictive,
	load_run,
	locate_run,
	paired_difference,
	person_posterior,
	shape,
)
from vsrgrating.tracking import run_output, write_metrics

THRESHOLDS = {
	"acc": (0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999),
	"score": (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999),
}
NEAR_CEILING = 0.003  # 1 - x below this counts as "just under the ceiling"


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("evaluate", overrides)
	jax.config.update("jax_enable_x64", True)

	if cfg.reference not in cfg.runs:
		raise ValueError(f"reference {cfg.reference!r} is not one of {list(cfg.runs)}")

	dataset = locate_dataset(Path(cfg.dataset_root), cfg.dataset.name)
	fits = {response: load_fits(cfg, response, dataset.name) for response in cfg.responses}

	if cfg.predictive and any(r.model.dims > 1 for runs in fits.values() for r in runs):
		raise ValueError("the predictive checks are one-dimensional: pass predictive=false")

	keys = pl.read_parquet(dataset / "items.parquet").sort("item_idx")["keys"].to_numpy()

	for response, runs in fits.items():
		train = load_split(dataset, response, "train")
		test = load_split(dataset, response, "test")
		judged = judged_cells(test, train, cfg.min_train_responses)

		with run_output(cfg, overrides, response) as out:
			tables = {"heldout": heldout(runs, cfg.reference, judged)}
			tables["heldout_breakdown"] = breakdown(runs, cfg.reference, judged, keys)

			if judged is not test:
				tables["heldout_all"] = heldout(runs, cfg.reference, test)

			if cfg.predictive:
				tables |= predictive(runs, cfg.reference, train, THRESHOLDS[response])

			for name, table in tables.items():
				table.write_csv(out / f"{name}.csv")

			tags = {
				"response": response,
				"dataset_hash": test.dataset_hash,
				"reference": cfg.reference,
			}
			write_metrics(out, tags, heldout_metrics(tables["heldout"]), {})


def load_fits(cfg: DictConfig, response: str, dataset_hash: str) -> list[Fitted]:
	"""Every fit named in `runs`; a missing fit or one on another dataset fails here."""
	root = Path(cfg.runs_root)
	return [
		load_run(label, locate_run(root, experiment, response, dataset_hash))
		for label, experiment in cfg.runs.items()
	]


def judged_cells(test: Split, train: Split, min_train_responses: int | None) -> Split:
	if min_train_responses is None:
		return test

	counts = np.bincount(train.item, minlength=test.n_items)
	return subset(test, counts[test.item] >= min_train_responses)


def heldout(runs: list[Fitted], reference: str, test: Split) -> pl.DataFrame:
	"""Each fit's held-out log marginal likelihood, paired over persons against the reference and
	against the run listed before it."""
	ll = {r.name: person_posterior(r, test).log_marginal for r in runs}
	rows = []

	for previous, r in zip([runs[0], *runs], runs, strict=False):
		diff, se = paired_difference(ll[reference], ll[r.name])
		step, step_se = paired_difference(ll[previous.name], ll[r.name])

		rows.append(
			{
				"run": r.name,
				"estimator": r.kind,
				"heldout_loglik": float(ll[r.name].sum()),
				"per_cell": float(ll[r.name].sum() / test.x.size),
				"diff": diff,
				"se": se,
				"step_diff": step,
				"step_se": step_se,
			}
		)

	return pl.DataFrame(rows)


def heldout_metrics(table: pl.DataFrame) -> dict[str, float]:
	return {
		f"{key}.{row['run']}": row[key]
		for row in table.iter_rows(named=True)
		for key in ("heldout_loglik", "diff", "se")
	}


def breakdown(runs: list[Fitted], reference: str, test: Split, keys: np.ndarray) -> pl.DataFrame:
	"""Where the held-out difference comes from: key modes, and the cells just under 1."""
	u = 1.0 - test.x
	near = (u > 0.0) & (u < NEAR_CEILING)
	subsets = {"all": np.ones(test.x.size, dtype=bool), f"without 0<1-x<{NEAR_CEILING}": ~near}

	for k in np.unique(keys[test.item]):
		subsets[f"{k}K"] = keys[test.item] == k

	ref = next(r for r in runs if r.name == reference)
	rows = []

	for label, mask in subsets.items():
		sub = subset(test, mask)
		base = person_posterior(ref, sub).log_marginal

		for r in runs:
			if r.name == reference:
				continue

			diff, se = paired_difference(base, person_posterior(r, sub).log_marginal)
			rows.append(
				{"cells": label, "n_cells": int(mask.sum()), "run": r.name, "diff": diff, "se": se}
			)

	return pl.DataFrame(rows)


def predictive(
	runs: list[Fitted], reference: str, train: Split, thresholds: tuple[float, ...]
) -> dict[str, pl.DataFrame]:
	"""Endpoint and threshold rates, PIT deciles, and the spread of u = 1 - x, on train cells."""
	preds, posts = {}, {}

	for r in runs:
		posts[r.name] = person_posterior(r, train)
		preds[r.name] = cell_predictive(r, train, posts[r.name], thresholds)

	x = train.x
	observed = {"run": "observed", "x=1": float((x == 1.0).mean())}
	observed |= {f"x<={t}": float((x <= t).mean()) for t in thresholds}
	rows = [observed]

	for name, p in preds.items():
		row = {"run": name, "x=1": float(p["p_one"].mean())}
		row |= {f"x<={t}": float(p[f"cdf_{t}"].mean()) for t in thresholds}
		rows.append(row)

	pit_rows = []
	for name, p in preds.items():
		pit = p["pit"].drop_nans().to_numpy()
		hist = np.histogram(pit, bins=10, range=(0.0, 1.0))[0] / pit.size
		pit_rows.append({"run": name, **{f"d{i + 1}": float(h) for i, h in enumerate(hist)}})

	tagged = shape.groups(preds[reference].select("item", "person", "x"), posts[reference].mean)
	grouped = shape.compare(tagged, preds)
	law, by_distance = shape.summary(grouped, list(preds))

	return {
		"ppc": pl.DataFrame(rows),
		"pit": pl.DataFrame(pit_rows),
		"cv_law": law,
		"cv_by_distance": by_distance.with_columns(pl.col("distance").cast(pl.String)),
	}


if __name__ == "__main__":
	typer.run(main)

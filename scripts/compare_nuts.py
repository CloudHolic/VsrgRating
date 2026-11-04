"""Compare a ZOI-Beta MAP fit with the NUTS posterior on the same data, one response at a time.

uv run scripts/compare_nuts.py +experiment=core300_map_vs_nuts
"""

from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import typer
from arviz_stats.base import array_stats
from omegaconf import DictConfig

from vsrgrating.config import compose_config
from vsrgrating.data import locate_dataset
from vsrgrating.evaluation.fitted import load_run, locate_run
from vsrgrating.irt.zoi_beta.model import COORDS, ItemParams
from vsrgrating.readout.grids import GRIDS
from vsrgrating.readout.zoi_beta import b50
from vsrgrating.tracking import run_output, write_metrics


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("compare_nuts", overrides)
	jax.config.update("jax_enable_x64", True)

	dataset = locate_dataset(Path(cfg.dataset_root), cfg.dataset.name)
	pairs = {response: locate_pair(cfg, response, dataset.name) for response in cfg.responses}

	for response, (nuts, point) in pairs.items():
		with run_output(cfg, overrides, response) as out:
			draws = nuts_draws(nuts)
			check_halves(cfg.halves, draws.b.shape[0])

			summary = pl.read_parquet(nuts / "items_summary.parquet")
			mean_tau = item_params(load_run("nuts", nuts).item)
			map_tau = item_params(load_run("map", point).item)

			coords = coord_table(map_tau, summary)
			readout, items = readout_tables(map_tau, mean_tau, draws, GRIDS[response], cfg.halves)

			coords.write_csv(out / "coords.csv")
			readout.write_csv(out / "readout.csv")
			items.write_parquet(out / "items.parquet")

			tags = {"response": response, "dataset_hash": dataset.name}
			write_metrics(out, tags, comparison_metrics(coords, readout), {})


def locate_pair(cfg: DictConfig, response: str, dataset_hash: str) -> tuple[Path, Path]:
	"""(NUTS run, MAP run), both ZOI-Beta fits of this dataset."""
	root = Path(cfg.runs_root)
	nuts = locate_run(root, cfg.nuts, response, dataset_hash)
	point = locate_run(root, cfg.map, response, dataset_hash)

	for path, kind in ((nuts, "nuts"), (point, "map")):
		fitted = load_run(kind, path)
		if fitted.model.family != "zoi_beta" or fitted.kind != kind:
			raise ValueError(
				f"{path} is a {fitted.model.family} {fitted.kind} fit, not zoi_beta {kind}"
			)

	return nuts, point


def item_params(item: object) -> ItemParams:
	assert isinstance(item, ItemParams)
	return item


def nuts_draws(path: Path) -> ItemParams:
	"""The stored (thinned) item draws, each coordinate shaped (chains, draws, items)."""
	long = pl.read_parquet(path / "draws_items.parquet")
	wide = long.pivot(on="coord", index=["chain", "draw", "item_idx"], values="value").sort(
		"chain", "draw", "item_idx"
	)
	shape = (long["chain"].n_unique(), long["draw"].n_unique(), long["item_idx"].n_unique())

	return ItemParams(*(jnp.asarray(wide[c].to_numpy().reshape(shape)) for c in COORDS))


def check_halves(halves: list[list[int]], chains: int) -> None:
	used = [c for half in halves for c in half]
	if len(halves) != 2 or len(set(used)) != len(used) or not set(used) <= set(range(chains)):
		raise ValueError(
			f"halves must be two disjoint chain sets from 0..{chains - 1}, got {halves}"
		)


def coord_table(point: ItemParams, summary: pl.DataFrame) -> pl.DataFrame:
	"""Per coordinate: MAP's distance from the posterior mean and median in posterior sds, and
	the share of items whose MAP lies inside the 90% interval."""
	rows = []
	for coord, value in zip(COORDS, point, strict=True):
		s = summary.filter(pl.col("coord") == coord).sort("item_idx")
		v, sd = np.asarray(value), s["sd"].to_numpy()
		rows.append(
			{
				"coord": coord,
				"median_dev_mean": float(np.median(np.abs(v - s["mean"].to_numpy()) / sd)),
				"median_dev_median": float(np.median(np.abs(v - s["q50"].to_numpy()) / sd)),
				"inside_90": float(
					np.mean((s["q05"].to_numpy() <= v) & (v <= s["q95"].to_numpy()))
				),
			}
		)

	return pl.DataFrame(rows)


def readout_tables(
	point: ItemParams,
	mean_tau: ItemParams,
	draws: ItemParams,
	grid: tuple[float, ...],
	halves: list[list[int]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
	"""b50 rank agreement per y: MAP against the NUTS posterior median of b50, the two chain
	halves against each other (the noise floor), and MAP against b50 at the posterior mean."""
	chains, n_draws, n_items = draws.b.shape
	flat = ItemParams(*(jnp.asarray(v.reshape(-1)) for v in draws))
	rows, items = [], {"item_idx": np.arange(n_items)}

	for y in grid:
		at_map = b50(point, y)
		per_draw = b50(flat, y).reshape(chains, n_draws, n_items)

		median = np.median(per_draw, axis=(0, 1))
		first, second = (np.median(per_draw[list(h)], axis=(0, 1)) for h in halves)
		mcse = array_stats.mcse(per_draw, chain_axis=0, draw_axis=1, method="median")

		rows.append(
			{
				"y": y,
				"map_vs_nuts": _spearman(at_map, median),
				"half_vs_half": _spearman(first, second),
				"map_vs_mean_tau": _spearman(at_map, b50(mean_tau, y)),
				"mcse_median": float(np.median(mcse)),
			}
		)
		items |= {f"map_b50_{y:.2f}": at_map, f"nuts_b50_{y:.2f}": median}

	return pl.DataFrame(rows), pl.DataFrame(items)


def comparison_metrics(coords: pl.DataFrame, readout: pl.DataFrame) -> dict[str, float]:
	metrics = {
		f"{key}.{row['coord']}": row[key]
		for row in coords.iter_rows(named=True)
		for key in ("inside_90", "median_dev_mean")
	}
	metrics |= {
		f"{key}.{row['y']:.2f}": row[key]
		for row in readout.iter_rows(named=True)
		for key in ("map_vs_nuts", "half_vs_half")
	}

	return metrics


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
	frame = pl.DataFrame({"a": a, "b": b})
	return float(frame.select(pl.corr("a", "b", method="spearman")).item())


if __name__ == "__main__":
	typer.run(main)

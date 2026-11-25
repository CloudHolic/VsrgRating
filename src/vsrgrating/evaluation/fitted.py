"""A finished fit read back from its run directory as a point estimate in the model's
coordinates."""

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import polars as pl
from jax import Array
from omegaconf import OmegaConf

from ..irt.cnrm import model as cnrm
from ..irt.cnrm.transform import TransformSpec
from ..irt.map import Model
from ..irt.zoi_beta import model as zoi_beta


@dataclass(frozen=True)
class Fitted:
	"""A named point fit; `raw` is G's parameters for the cnrm, None for zoi_beta."""

	name: str
	model: Model
	item: zoi_beta.ItemParams | cnrm.ItemParams
	raw: Array | None
	kind: str


def load_run(name: str, path: Path, chains: tuple[int, ...] | None = None) -> Fitted:
	"""Reads one run directory; NUTS gives posterior means, over `chains` only if given."""
	cfg = OmegaConf.load(path / "config.yaml")
	family, kind = cfg.model.family, cfg.estimator.kind

	if family == "cnrm":
		items = pl.read_parquet(path / "items.parquet").sort("item_idx")
		if items["item_idx"].to_list() != list(range(items.height)):
			raise ValueError(
				f"{path}: items.parquet does not hold every item 0..{items.height - 1}"
			)

		dims = sum(c.startswith("alpha_") for c in items.columns) or 1
		names = ["alpha"] if dims == 1 else [f"alpha_{k}" for k in range(1, dims + 1)]
		alpha = items.select(names).to_numpy()
		item = cnrm.ItemParams(
			jnp.asarray(alpha[:, 0] if dims == 1 else alpha),
			jnp.asarray(items["beta"].to_numpy()),
			jnp.asarray(items["sigma"].to_numpy()),
		)

		if kind == "em":
			spec = TransformSpec("linear")
			return Fitted(name, Model("cnrm", spec), item, spec.init(), kind)

		record = json.loads((path / "transform.json").read_text(encoding="utf-8"))
		spec = TransformSpec(record["kind"], tuple(record["knots"]), record["penalty"])
		raw = jnp.asarray(record["raw"]).reshape(spec.init().shape)

		return Fitted(name, Model("cnrm", spec, dims), item, raw, kind)

	if kind == "map":
		wide = pl.read_parquet(path / "items.parquet")
	elif chains is None:
		summary = pl.read_parquet(path / "items_summary.parquet")
		wide = summary.pivot(on="coord", index="item_idx", values="mean")
	else:
		draws = pl.read_parquet(path / "draws_items.parquet").filter(pl.col("chain").is_in(chains))
		means = draws.group_by("item_idx", "coord").agg(pl.col("value").mean())
		wide = means.pivot(on="coord", index="item_idx", values="value")

	wide = wide.sort("item_idx")
	item = zoi_beta.ItemParams(*(jnp.asarray(wide[c].to_numpy()) for c in zoi_beta.COORDS))

	return Fitted(name, Model("zoi_beta"), item, None, kind)


def run_hash(path: Path) -> str:
	"""The dataset hash the fit in `path` read."""
	record = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
	return record["tags"]["dataset_hash"]


def locate_run(runs_root: Path, experiment: str, response: str, dataset_hash: str) -> Path:
	"""The finished fit `runs_root/experiment/response`, checked to have read `dataset_hash`."""
	path = runs_root / experiment / response
	if not (path / "metrics.json").exists():
		raise FileNotFoundError(f"no finished fit at {path}")

	found = run_hash(path)
	if found != dataset_hash:
		raise ValueError(f"{path} was fit on dataset {found}, not {dataset_hash}")

	return path

"""Reference item parameters that bracket a chart model on one part, without reading a chart.

uv run --env-file .env scripts/reference_items.py name=keys4to8_reference kind=floor
uv run --env-file .env scripts/reference_items.py name=keys4to8_reference kind=ceiling
"""

import time
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import torch
import typer
from omegaconf import DictConfig

from vsrgrating.config import compose_config
from vsrgrating.encoder.network import (
	CellTensors,
	ItemOutput,
	Posterior,
	cell_tensors,
	item_loglik,
	load_posterior,
	write_item_params,
)
from vsrgrating.tracking import run_output, write_metrics

PARTS = ("dev", "valid")
KINDS = ("floor", "ceiling")
# alpha_1..3, beta, log sigma: the components the floor takes medians of and the ceiling's prior
# is put on.
COMPONENTS = ("alpha_1", "alpha_2", "alpha_3", "beta", "log_sigma")
LOG_SIGMA_RANGE = (-20.0, 10.0)
MAX_ITER = 5000
# Gradient norm under which an item counts as converged; float64 noise at the optimum is below it.
CONVERGED_GRAD = 1e-2


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("reference_items", overrides)
	check_choice(cfg)
	device = torch.device("cuda")

	posterior = load_posterior(Path(cfg.posterior_run))
	items = target_items(posterior, cfg.part)
	cells = cell_tensors(posterior, items["item_idx"].to_numpy(), device)
	prior = key_mode_psi(posterior)
	floor = floor_params(items, prior)
	tags = {
		**posterior.tags,
		"kind": str(cfg.kind),
		"part": str(cfg.part),
		"device": torch.cuda.get_device_name(device),
	}

	with run_output(cfg, overrides, f"{cfg.kind}-{cfg.part}") as out:
		start = time.perf_counter()
		mu = torch.as_tensor(posterior.mu, device=device)
		cov = torch.as_tensor(posterior.cov, device=device)
		metrics: dict[str, float] = {}
		params = torch.as_tensor(floor, device=device)
		if cfg.kind == "ceiling":
			params, metrics = fit_ceiling(params, items, prior, cells, mu, cov)

		out_items = as_output(params)
		loglik = item_loglik(out_items, torch.arange(items.height, device=device), cells, mu, cov)
		write_item_params(out / f"items_{cfg.part}.parquet", items.select("item_idx"), out_items)
		metrics |= {
			"items": float(items.height),
			"total_log_expected_k": float(loglik.sum()),
			"sigma_min": float(out_items.sigma.min()),
			"runtime_s": time.perf_counter() - start,
		}
		write_metrics(out, tags, metrics, {})


def check_choice(cfg: DictConfig) -> None:
	if cfg.kind not in KINDS:
		raise ValueError(f"kind must be one of {KINDS}, got {cfg.kind!r}")
	if cfg.part not in PARTS:
		raise ValueError(f"part must be one of {PARTS}, got {cfg.part!r}")


def target_items(posterior: Posterior, part: str) -> pl.DataFrame:
	"""item_idx and keys of the part's items in item order; dev leaves out `drop_keys`."""
	cells = posterior.cells.filter(pl.col("part") == part)
	if cells["in_t"].any():
		raise ValueError(f"{part} cells are in T, but the references read them as held out")
	items = cells.select("item_idx", "keys").unique().sort("item_idx")
	if part == "dev":
		items = items.filter(~pl.col("keys").is_in(list(posterior.drop_keys)))
	return items


def key_mode_psi(posterior: Posterior) -> pl.DataFrame:
	"""The T items' parameters with their key mode and log sigma."""
	keys = posterior.cells.select("item_idx", "keys").unique()
	return posterior.psi_t.join(keys, on="item_idx").with_columns(log_sigma=pl.col("sigma").log())


def floor_params(items: pl.DataFrame, prior: pl.DataFrame) -> np.ndarray:
	"""(I, 5) float64: each item's key-mode median of every component."""
	medians = prior.group_by("keys").agg(pl.col(c).median() for c in COMPONENTS)
	table = items.join(medians, on="keys", how="left")
	if table.select(pl.col(c).null_count() for c in COMPONENTS).to_numpy().any():
		raise ValueError("a target key mode has no T items")
	return table.select(COMPONENTS).to_numpy().astype(np.float64)


def fit_ceiling(
	start: torch.Tensor,
	items: pl.DataFrame,
	prior: pl.DataFrame,
	cells: CellTensors,
	mu: torch.Tensor,
	cov: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
	"""Each item's MAP under its own cells' log E_q[k] and an independent normal prior per
	component (that key mode's T mean and variance). Items share no parameter, so one L-BFGS
	over all of them reaches every item's own optimum."""
	moments = prior.group_by("keys").agg(
		*(pl.col(c).mean().alias(f"{c}_mean") for c in COMPONENTS),
		*(pl.col(c).var().alias(f"{c}_var") for c in COMPONENTS),
	)
	table = items.join(moments, on="keys", how="left")
	device = start.device
	mean = torch.as_tensor(table.select(f"{c}_mean" for c in COMPONENTS).to_numpy(), device=device)
	var = torch.as_tensor(table.select(f"{c}_var" for c in COMPONENTS).to_numpy(), device=device)
	index = torch.arange(items.height, device=device)
	# L-BFGS flattens gradients with view(), which a gradient strided like a column slice breaks.
	flat = start.flatten().clone().requires_grad_(True)

	def objective() -> torch.Tensor:
		params = flat.view_as(start)
		loglik = item_loglik(as_output(params), index, cells, mu, cov)
		log_prior = (-0.5 * (params - mean) ** 2 / var).sum(1)
		return -(loglik + log_prior)

	optimizer = torch.optim.LBFGS(
		[flat],
		max_iter=MAX_ITER,
		tolerance_grad=1e-9,
		tolerance_change=0.0,
		history_size=50,
		line_search_fn="strong_wolfe",
	)

	def closure() -> torch.Tensor:
		optimizer.zero_grad()
		loss = objective().sum()
		loss.backward()
		return loss

	optimizer.step(closure)
	(grad,) = torch.autograd.grad(objective().sum(), flat)
	norms = grad.view_as(start).norm(dim=1)
	metrics = {
		"converged_items": float((norms <= CONVERGED_GRAD).sum()),
		"max_grad_norm": float(norms.max()),
		"iterations": float(optimizer.state[flat]["n_iter"]),
	}
	return flat.detach().view_as(start), metrics


def as_output(params: torch.Tensor) -> ItemOutput:
	return ItemOutput(params[:, :3], params[:, 3], torch.exp(params[:, 4].clamp(*LOG_SIGMA_RANGE)))


if __name__ == "__main__":
	typer.run(main)

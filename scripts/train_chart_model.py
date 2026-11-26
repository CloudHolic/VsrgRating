"""Train a chart model on the train items under a fixed person posterior.

uv run --env-file .env scripts/train_chart_model.py +experiment=keys4to8_scratch fit.lr=3e-4
uv run --env-file .env scripts/train_chart_model.py +experiment=keys4to8_stats
"""

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import torch
import typer
from omegaconf import DictConfig
from urc_converter import Chart, scale_rate

from vsrgrating.config import compose_config
from vsrgrating.data import RATE_GROUP_RATES
from vsrgrating.encoder.inputs import (
	STATS_FEATURES,
	ChartTokens,
	LayoutInput,
	chart_stats,
	layout_input,
	layouts,
	mirror,
	tokenize,
)
from vsrgrating.encoder.network import (
	CellTensors,
	ChartEncoder,
	ChartItemModel,
	EncoderConfig,
	FitConfig,
	ItemHead,
	ItemInit,
	ItemOutput,
	Posterior,
	Scaler,
	StatsModel,
	TrainData,
	VectorScaler,
	cell_tensors,
	fit,
	fit_stats,
	load_chart,
	load_items,
	load_posterior,
	predict,
	write_item_params,
)
from vsrgrating.tracking import run_output, write_metrics

PARTS = ("train", "dev")
# osu!mania narrows every window but the first by 3 ms per OD point.
OD_STEP_MS = 3.0

type Charts = list[list[LayoutInput]]
type Record = tuple[dict[str, float], dict[str, list[float]]]


@dataclass(frozen=True)
class Part:
	"""One part: its row positions in the item table, those rows, and their cells."""

	rows: np.ndarray
	table: pl.DataFrame
	cells: CellTensors

	@property
	def keys(self) -> pl.DataFrame:
		"""The column naming its items in the tables written for it."""
		return self.table.select("item_idx")


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("train_chart_model", overrides)
	label = run_label(cfg)
	check_new(cfg, label)

	device = torch.device("cuda")

	posterior = load_posterior(Path(cfg.posterior_run))
	split_dir = Path(cfg.split_root) / posterior.dataset_name / posterior.tags["dataset_hash"]
	items = load_items(posterior, split_dir, Path(cfg.charts_root), PARTS)
	charts, split_chords = read_charts(Path(cfg.charts_root), items.table)
	tokens = [tokenize(chart) for chart in charts]

	train = part_of(posterior, items.table, "train", device)
	dev = part_of(posterior, items.table, "dev", device)

	tags = {
		**posterior.tags,
		"chart_index": items.chart_index,
		"model_kind": str(cfg.model.kind),
		"device": torch.cuda.get_device_name(device),
	}

	with run_output(cfg, overrides, label) as out:
		start = time.perf_counter()

		if cfg.model.kind == "encoder":
			metrics, series = train_encoder(cfg, out, posterior, train, dev, charts, tokens)
		else:
			metrics, series = train_stats(cfg, out, posterior, train, dev, tokens)

		metrics |= {
			"collisions": float(sum(t.collisions for t in tokens)),
			"split_chords": float(split_chords),
			"runtime_s": time.perf_counter() - start,
		}
		write_metrics(out, tags, metrics, series)


def run_label(cfg: DictConfig) -> str:
	match cfg.model.kind:
		case "encoder":
			return f"lr{cfg.fit.lr:g}-seed{cfg.fit.seed}"
		case "stats":
			return "lbfgs"

	raise ValueError(f"model.kind must be encoder or stats, got {cfg.model.kind!r}")


def check_new(cfg: DictConfig, label: str) -> None:
	"""Fails before any data is read when the local run already exists."""
	out = Path(cfg.runs_root) / cfg.name / label

	if not cfg.tracking.mlflow and out.exists():
		raise FileExistsError(f"{out} already holds a run")


def read_charts(charts_root: Path, table: pl.DataFrame) -> tuple[list[Chart], int]:
	"""Every item's chart as played, and the chords the rate changes split."""
	charts, split = [], 0

	for row in table.iter_rows(named=True):
		chart, chords = load_chart(charts_root, row)
		charts.append(chart)

		split += chords

	return charts, split


def part_of(posterior: Posterior, table: pl.DataFrame, part: str, device: torch.device) -> Part:
	rows = np.flatnonzero(table["part"].to_numpy() == part)
	return Part(
		rows, table[rows], cell_tensors(posterior, table["item_idx"].to_numpy()[rows], device)
	)


def train_encoder(
	cfg: DictConfig,
	out: Path,
	posterior: Posterior,
	train: Part,
	dev: Part,
	charts: list[Chart],
	tokens: list[ChartTokens],
) -> Record:
	torch.manual_seed(cfg.fit.seed)
	device = train.cells.x.device

	inputs = [inputs_of(t) for t in tokens]
	mirrored = [inputs_of(tokenize(mirror(chart))) for chart in charts]
	scaler = Scaler.fit([x for i in train.rows for x in inputs[i]])

	config = EncoderConfig(**cfg.model.encoder)
	encoder = ChartEncoder(config)

	init = item_init(posterior)
	model = ChartItemModel(encoder, ItemHead(encoder.width, init)).to(device)

	fit_config = FitConfig(**{k: v for k, v in cfg.fit.items() if k != "keep_epochs"})
	mu = torch.as_tensor(posterior.mu, device=device)
	cov = torch.as_tensor(posterior.cov, device=device)

	train_data = TrainData(
		[inputs[i] for i in train.rows], [mirrored[i] for i in train.rows], train.cells, mu, cov
	)
	dev_data = TrainData([inputs[i] for i in dev.rows], None, dev.cells, mu, cov)

	def on_epoch(epoch: int, record: dict[str, float], dev_out: ItemOutput) -> None:
		print(
			f"epoch {epoch}: train {record['train_nll']:.5f},"
			f" dev {record['dev_loglik_per_cell']:.5f} per cell, clipped {record['clipped']:.2f}",
			flush=True,
		)

		if cfg.fit.keep_epochs:
			(out / "epochs").mkdir(exist_ok=True)
			write_item_params(out / "epochs" / f"items_dev_{epoch}.parquet", dev.keys, dev_out)

	result = fit(model, train_data, dev_data, scaler, fit_config, on_epoch)

	budget = fit_config.token_budget
	predicted = predict(model, dev_data.inputs, scaler, budget)

	write_item_params(
		out / "items_train.parquet", train.keys, predict(model, train_data.inputs, scaler, budget)
	)
	write_item_params(out / "items_dev.parquet", dev.keys, predicted)

	variants = {
		"unknown_judgment": predict(model, dev_data.inputs, scaler, budget, judgment_unknown=True),
		"mirrored": predict(model, [mirrored[i] for i in dev.rows], scaler, budget),
		"od_minus": predict(model, shifted_od(tokens, dev.rows, -1), scaler, budget),
		"od_plus": predict(model, shifted_od(tokens, dev.rows, 1), scaler, budget),
	}
	checks: dict[str, float] = {}

	for name, variant in variants.items():
		write_item_params(out / f"items_dev_{name}.parquet", dev.keys, variant)
		checks |= differences(name, variant, predicted)

	checks |= dev_rates(out, model, scaler, budget, dev, charts)
	torch.save(
		{
			"model": model.state_dict(),
			"encoder": asdict(config),
			"init": asdict(init),
			"scaler": asdict(scaler),
		},
		out / "model.pt",
	)

	best = result.history[result.best_epoch]
	metrics = {
		"parameters": float(sum(p.numel() for p in model.parameters())),
		"best_epoch": float(result.best_epoch),
		"epochs": float(len(result.history)),
		"dev_loglik": best["dev_loglik"],
		"dev_loglik_per_cell": best["dev_loglik_per_cell"],
		**checks,
	}
	series = {name: [h[name] for h in result.history] for name in result.history[0]}

	return metrics, series


def train_stats(
	cfg: DictConfig,
	out: Path,
	posterior: Posterior,
	train: Part,
	dev: Part,
	tokens: list[ChartTokens],
) -> Record:
	device = train.cells.x.device

	rows = np.stack([chart_stats(t) for t in tokens])
	scaler = VectorScaler.fit(rows[train.rows])
	x = torch.as_tensor(scaler.apply(rows), dtype=torch.float64, device=device)

	init = item_init(posterior)
	model = StatsModel(len(STATS_FEATURES), init).double().to(device)
	result = fit_stats(
		model,
		x[train.rows],
		train.cells,
		x[dev.rows],
		torch.as_tensor(posterior.mu, device=device),
		torch.as_tensor(posterior.cov, device=device),
		max_iter=cfg.stats.max_iter,
		rtol=cfg.stats.rtol,
	)

	write_item_params(out / "items_train.parquet", train.keys, result.train)
	write_item_params(out / "items_dev.parquet", dev.keys, result.dev)
	torch.save(
		{"model": model.state_dict(), "init": asdict(init), "scaler": asdict(scaler)},
		out / "model.pt",
	)

	metrics = {
		"converged": float(result.converged),
		"evaluations": float(len(result.losses)),
		"train_nll": result.losses[-1],
	}

	return metrics, {"loss": result.losses}


def item_init(posterior: Posterior) -> ItemInit:
	"""The componentwise median of the T items' fitted parameters (sigma on the log scale)."""
	psi = posterior.psi_t
	a1, a2, a3, beta = (
		float(np.median(psi[name].to_numpy())) for name in ("alpha_1", "alpha_2", "alpha_3", "beta")
	)

	return ItemInit((a1, a2, a3), beta, float(np.median(np.log(psi["sigma"].to_numpy()))))


def inputs_of(tokens: ChartTokens) -> list[LayoutInput]:
	"""Layout inputs of a chart under every layout of its lane count."""
	return [layout_input(tokens, m) for m in layouts(tokens.lanes)]


def shifted_od(tokens: list[ChartTokens], rows: np.ndarray, points: int) -> Charts:
	"""Inputs of `rows` with an osu!mania judgment `points` OD higher (narrower windows)."""
	charts = []

	for i in rows:
		t = tokens[i]
		if t.judgment is not None:
			windows = t.judgment[:, 0].copy()
			windows[1:] -= OD_STEP_MS * points

			judgment = np.column_stack([np.maximum.accumulate(windows), t.judgment[:, 1]])
			t = replace(t, judgment=judgment)

		charts.append(inputs_of(t))

	return charts


def dev_rates(
	out: Path, model: ChartItemModel, scaler: Scaler, budget: int, dev: Part, charts: list[Chart]
) -> dict[str, float]:
	"""Every dev NM beatmap at every rate group, written to dev_rates.parquet."""
	beatmaps = dev.table["beatmap_id"].to_numpy()
	keys, inputs = [], []

	for k in np.flatnonzero(dev.table["rate_group"].to_numpy() == "NM"):
		chart = charts[dev.rows[k]]

		for group, rate in RATE_GROUP_RATES.items():
			scaled = chart if rate == 1.0 else scale_rate(chart, rate).chart

			keys.append((int(beatmaps[k]), group))
			inputs.append(inputs_of(tokenize(scaled)))

	predicted = predict(model, inputs, scaler, budget)
	table = pl.DataFrame(keys, schema=["beatmap_id", "rate_group"], orient="row")
	write_item_params(out / "dev_rates.parquet", table, predicted)

	return {"dev_rates_finite": float(_finite(predicted))}


def differences(name: str, variant: ItemOutput, base: ItemOutput) -> dict[str, float]:
	"""Whether `variant` is finite, and its largest and mean absolute gap to `base` per output."""
	record = {f"{name}_finite": float(_finite(variant))}
	gaps = {
		"alpha": variant.alpha - base.alpha,
		"beta": variant.beta - base.beta,
		"log_sigma": variant.sigma.log() - base.sigma.log(),
	}

	for part, gap in gaps.items():
		record[f"{name}_{part}_max_gap"] = float(gap.abs().max())
		record[f"{name}_{part}_mean_gap"] = float(gap.abs().mean())

	return record


def _finite(out: ItemOutput) -> bool:
	return all(bool(torch.isfinite(v).all()) for v in (out.alpha, out.beta, out.sigma))


if __name__ == "__main__":
	typer.run(main)

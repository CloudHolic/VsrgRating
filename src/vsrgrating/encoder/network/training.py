"""Fine-tuning under the fixed person posterior."""

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import Tensor

from ..inputs import LayoutInput
from .batching import collate, token_batches
from .heads import ChartItemModel, ItemOutput, StatsModel
from .items import Posterior
from .objective import expected_log_k, log_expected_k
from .scaling import Scaler

# Response-count bands of an item for the sigma watch, by its cells in the posterior run.
SIGMA_BANDS = ((10, 19), (20, 99), (100, None))


@dataclass(frozen=True)
class CellTensors:
	"""The cells of a list of items, grouped by item."""

	offsets: Tensor
	x: Tensor
	g: Tensor
	log_dg: Tensor
	in_t: Tensor
	person: Tensor

	@property
	def n_cells(self) -> int:
		"""Cells of all the items."""
		return int(self.offsets[-1])

	def counts(self) -> np.ndarray:
		"""Cells of each item."""
		return np.diff(self.offsets.cpu().numpy())


@dataclass(frozen=True)
class FitConfig:
	"""Optimizer, schedule, stopping and augmentation of one fine-tuning run."""

	lr: float
	seed: int
	weight_decay: float = 0.01
	warmup: float = 0.05
	final_lr_fraction: float = 0.05
	clip: float = 1.0
	max_epochs: int = 60
	patience: int = 10
	token_budget: int = 32768
	judgment_drop: float = 0.3
	mirror: float = 0.5


@dataclass(frozen=True)
class TrainData:
	"""Charts in item order as tensors (mu (N, D), cov (N, D, D), float64)."""

	inputs: Sequence[Sequence[LayoutInput]]
	mirrored: Sequence[Sequence[LayoutInput]] | None
	cells: CellTensors
	mu: Tensor
	cov: Tensor


@dataclass(frozen=True)
class FitResult:
	"""The epoch whose dev score was best and every epoch's record."""

	best_epoch: int
	history: list[dict[str, float]]


@dataclass(frozen=True)
class StatsFitResult:
	"""The baseline's fit."""

	converged: bool
	losses: list[float]
	train: ItemOutput
	dev: ItemOutput


def cell_tensors(posterior: Posterior, item_idx: np.ndarray, device: torch.device) -> CellTensors:
	"""The posterior run's cell of `item_idx`, in that order."""
	order = pl.DataFrame({"item_idx": item_idx, "position": np.arange(item_idx.size)})
	cells = posterior.cells.join(order, on="item_idx").sort("position", "person_idx")
	counts = np.bincount(cells["position"].to_numpy(), minlength=item_idx.size)

	def tensor(name: str, dtype: torch.dtype) -> Tensor:
		return torch.as_tensor(cells[name].to_numpy(), dtype=dtype, device=device)

	return CellTensors(
		offsets=torch.as_tensor(np.concatenate([[0], np.cumsum(counts)]), device=device),
		x=tensor("x", torch.float64),
		g=tensor("g", torch.float64),
		log_dg=tensor("log_dg", torch.float64),
		in_t=tensor("in_t", torch.bool),
		person=tensor("person_idx", torch.long),
	)


def item_loglik(
	out: ItemOutput, items: Tensor, cells: CellTensors, mu: Tensor, cov: Tensor
) -> Tensor:
	"""The summed cell log likelihood of each chart."""
	starts, ends = cells.offsets[items], cells.offsets[items + 1]
	lengths = ends - starts

	chart = torch.repeat_interleave(torch.arange(items.numel(), device=items.device), lengths)
	first = torch.repeat_interleave(starts - torch.cumsum(lengths, 0) + lengths, lengths)
	rows = first + torch.arange(int(lengths.sum()), device=items.device)

	person = cells.person[rows]
	args = (
		cells.x[rows],
		cells.g[rows],
		cells.log_dg[rows],
		out.alpha.double()[chart],
		out.beta.double()[chart],
		out.sigma.double()[chart],
		mu[person],
		cov[person],
	)
	t = cells.in_t[rows]

	loglik = torch.empty(rows.numel(), dtype=torch.float64, device=items.device)
	loglik[t] = expected_log_k(*(a[t] for a in args))
	loglik[~t] = log_expected_k(*(a[~t] for a in args))

	return torch.zeros(items.numel(), dtype=torch.float64, device=items.device).index_add(
		0, chart, loglik
	)


def predict(
	model: ChartItemModel,
	charts: Sequence[Sequence[LayoutInput]],
	scaler: Scaler,
	token_budgets: int,
	judgment_unknown: bool = False,
) -> ItemOutput:
	"""Item parameters of `charts` in their order, in eval mode."""
	device = next(model.parameters()).device
	was_training = model.training
	model.eval()
	n_tokens, n_layouts = _sizes(charts)
	batches = token_batches(
		n_tokens, n_layouts, token_budgets, model.encoder.config.segment, np.random.default_rng(0)
	)

	alpha = torch.empty(len(charts), model.head.linear.out_features - 2, device=device)
	beta = torch.empty(len(charts), device=device)
	sigma = torch.empty(len(charts), device=device)

	with torch.no_grad():
		for chosen in batches:
			batch = collate([charts[i] for i in chosen], scaler, float(judgment_unknown), None)
			out = model(batch.to(device))
			index = torch.as_tensor(chosen, device=device)

			alpha[index], beta[index], sigma[index] = out.alpha, out.beta, out.sigma

	model.train(was_training)

	return ItemOutput(alpha, beta, sigma)


def fit(
	model: ChartItemModel,
	train: TrainData,
	dev: TrainData,
	scaler: Scaler,
	config: FitConfig,
	on_epoch: Callable[[int, dict[str, float], ItemOutput], None] | None = None,
) -> FitResult:
	"""Fit the model."""
	device = next(model.parameters()).device
	generator = torch.Generator().manual_seed(config.seed)

	n_tokens, n_layouts = _sizes(train.inputs)
	segment = model.encoder.config.segment
	steps = config.max_epochs * len(
		token_batches(n_tokens, n_layouts, config.token_budget, segment, np.random.default_rng(0))
	)

	optimizer = _optimizer(model, config)
	schedule = torch.optim.lr_scheduler.LambdaLR(
		optimizer, lambda step: _lr_factor(step, steps, config)
	)

	dev_items = torch.arange(len(dev.inputs), device=device)
	train_counts = train.cells.counts()
	dev_counts = dev.cells.counts()

	# Without mirrored inputs every chart is drawn as it is.
	sources = (train.inputs, train.mirrored if train.mirrored is not None else train.inputs)
	history: list[dict[str, float]] = []
	best = (-math.inf, 0, copy.deepcopy(model.state_dict()))

	for epoch in range(config.max_epochs):
		model.train()
		rng = np.random.default_rng([config.seed, epoch])
		batches = token_batches(n_tokens, n_layouts, config.token_budget, segment, rng)

		scale = train.cells.n_cells / len(batches)
		losses, clipped = [], 0

		for chosen in batches:
			flips = rng.random(chosen.size) < config.mirror
			charts = [sources[int(f)][i] for i, f in zip(chosen, flips, strict=True)]
			batch = collate(charts, scaler, config.judgment_drop, generator).to(device)

			out = model(batch)

			items = torch.as_tensor(chosen, device=device)
			loss = -item_loglik(out, items, train.cells, train.mu, train.cov).sum() / scale

			optimizer.zero_grad(set_to_none=True)
			loss.backward()

			norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
			clipped += bool(norm > config.clip)

			optimizer.step()
			schedule.step()

			losses.append(float(loss.detach()))

		dev_out = predict(model, dev.inputs, scaler, config.token_budget)
		dev_loglik = float(item_loglik(dev_out, dev_items, dev.cells, dev.mu, dev.cov).sum())
		train_out = predict(model, train.inputs, scaler, config.token_budget)

		record = {
			"epoch": epoch,
			"train_nll": float(np.mean(losses)),
			"dev_loglik": dev_loglik,
			"dev_loglik_per_cell": dev_loglik / dev.cells.n_cells,
			"clipped": clipped / len(batches),
			"lr": float(schedule.get_last_lr()[0]),
			**_sigma_watch("train", train_out.sigma, train_counts),
			**_sigma_watch("dev", dev_out.sigma, dev_counts),
		}
		history.append(record)

		if on_epoch is not None:
			on_epoch(epoch, record, dev_out)

		if dev_loglik > best[0]:
			best = (dev_loglik, epoch, copy.deepcopy(model.state_dict()))
		elif epoch - best[1] >= config.patience:
			break

	model.load_state_dict(best[2])
	return FitResult(best[1], history)


def fit_stats(
	model: StatsModel,
	train_x: Tensor,
	train_cells: CellTensors,
	dev_x: Tensor,
	mu: Tensor,
	cov: Tensor,
	max_iter: int = 2000,
	rtol: float = 1e-9,
) -> StatsFitResult:
	"""Full-batch L-BFGS on the train cells' mean negative log likelihood."""
	items = torch.arange(train_x.shape[0], device=train_x.device)
	losses: list[float] = []

	optimizer = torch.optim.LBFGS(
		model.parameters(),
		max_iter=max_iter,
		tolerance_change=rtol,
		tolerance_grad=0.0,
		line_search_fn="strong_wolfe",
	)

	def closure() -> Tensor:
		optimizer.zero_grad()

		loss = -item_loglik(model(train_x), items, train_cells, mu, cov).sum() / train_cells.n_cells
		loss.backward()
		losses.append(float(loss.detach()))

		return loss

	optimizer.step(closure)

	# L-BFGS stops early only on its tolerances; reaching max_iter means it did not converge.
	iterations = optimizer.state[next(iter(model.parameters()))]["n_iter"]
	converged = iterations < max_iter

	with torch.no_grad():
		return StatsFitResult(converged, losses, model(train_x), model(dev_x))


def _optimizer(model: ChartItemModel, config: FitConfig) -> torch.optim.AdamW:
	"""AdamW optimizer."""
	decay = [p for p in model.parameters() if p.ndim >= 2]
	plain = [p for p in model.parameters() if p.ndim < 2]

	return torch.optim.AdamW(
		[
			{"params": decay, "weight_decay": config.weight_decay},
			{"params": plain, "weight_decay": 0.0},
		],
		lr=config.lr,
	)


def _lr_factor(step: int, steps: int, config: FitConfig) -> float:
	"""Linear warmup, then cosine decay to `final_lr_fraction`."""
	warm = max(1, round(config.warmup * steps))
	if step < warm:
		return (step + 1) / warm

	progress = min(1.0, (step - warm) / max(1, steps - warm))
	floor = config.final_lr_fraction

	return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def _sizes(charts: Sequence[Sequence[LayoutInput]]) -> tuple[np.ndarray, np.ndarray]:
	n_tokens = np.array([layouts_[0].state.shape[0] for layouts_ in charts])
	n_layouts = np.array([len(layouts_) for layouts_ in charts])

	return n_tokens, n_layouts


def _sigma_watch(part: str, sigma: Tensor, counts: np.ndarray) -> dict[str, float]:
	"""Minimum and 5% quantile of sigma within each response-count band."""
	values = sigma.detach().double().cpu().numpy()
	record = {}

	for low, high in SIGMA_BANDS:
		band = (counts >= low) & (counts <= (high if high is not None else np.inf))
		name = f"{low}+" if high is None else f"{low}-{high}"

		if band.any():
			record[f"{part}_sigma_min_{name}"] = float(values[band].min())
			record[f"{part}_sigma_q05_{name}"] = float(np.quantile(values[band], 0.05))

	return record

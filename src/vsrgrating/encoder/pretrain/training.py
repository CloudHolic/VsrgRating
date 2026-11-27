"""The pretraining loop and the calibration of its task weights and clip."""

import copy
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .data import (
	Augment,
	LoadBatch,
	MLMBatch,
	PretrainCorpus,
	charts_of,
	epoch_batches,
	prepare,
	watch_samples,
)
from .model import PretrainModel
from .watch import evaluate_watch

# The watch charts' masks and prefixes come from this seed, whatever the run's.
WATCH_SEED = 20261127
# The pooled prefix gives one point per chart against every token of the token loss; a tenth of
# the token loss's pull on the shared layers keeps it from steering them.
POOL_SHARE = 0.1
CLIP_QUANTILE = 0.95


@dataclass(frozen=True)
class PretrainConfig:
	"""One pretraining run."""

	lr: float
	warmup: float
	decay: float
	clip: float
	updates: int
	mlm: bool
	load: bool
	w_mlm: float
	w_load_tok: float
	w_load_pool: float
	seed: int
	weight_decay: float = 0.01
	token_budget: int = 32768
	mirror: float = 0.5
	rate_fraction: float = 0.5
	rates: tuple[float, float] = (0.75, 1.5)
	judgment_drop: float = 0.3
	workers: int = 4
	watch_every: int = 0


@dataclass(frozen=True)
class Calibration:
	"""Task weights and clip from the gradients of the shared layers, and the step time."""

	w_mlm: float
	w_load_tok: float
	w_load_pool: float
	clip: float
	g_mlm: float
	g_load_tok: float
	g_load_pool: float
	seconds_per_step: float


@dataclass(frozen=True)
class PretrainResult:
	"""Every main-line and branch step's record, each branch's checkpoint and watch scores."""

	history: list[dict[str, float | str]]
	branches: dict[float, Path]
	watch: dict[float, dict[str, float]]


def wsd_factor(step: int, warm_steps: int, decay_start: int | None, decay_steps: int) -> float:
	"""Learning rate factor."""
	if step < warm_steps:
		return (step + 1) / warm_steps

	if decay_start is None or step < decay_start:
		return 1.0

	return max(0.0, (decay_start + decay_steps - step) / decay_steps)


def pretrain(
	model: PretrainModel,
	train: PretrainCorpus,
	watch: PretrainCorpus,
	config: PretrainConfig,
	branches: Sequence[float],
	out: Path,
	resume: Path | None = None,
	on_step: Callable[[dict[str, float | str]], None] | None = None,
) -> PretrainResult:
	"""The main line at a constant rate after warmup."""
	device = next(model.parameters()).device
	optimizer = _optimizer(model, config)
	base, step = config.updates, 0

	if resume is not None:
		state = torch.load(resume, weights_only=True)
		model.load_state_dict(state["model"])
		optimizer.load_state_dict(state["optimizer"])
		base, step = int(state["base_updates"]), int(state["step"])

	warm = max(1, round(config.warmup * base))
	valid = _valid(train, device)
	samples = watch_samples(watch, WATCH_SEED, model.encoder.config.segment, config.token_budget)
	schedule = _schedule(train, config, model, max(round(f * base) for f in branches))

	history: list[dict[str, float | str]] = []
	written: dict[float, Path] = {}
	scores: dict[float, dict[str, float]] = {}

	for frac in sorted(branches):
		total = round(frac * base)
		decay_steps = round(config.decay * total)
		decay_start = total - decay_steps

		main = range(step, decay_start)
		_run(
			model,
			optimizer,
			train,
			config,
			schedule,
			main,
			_constant(warm),
			valid,
			history,
			on_step,
			"main",
			samples,
		)
		step = decay_start

		if frac == 1.0:
			torch.save(
				{
					"model": model.state_dict(),
					"optimizer": optimizer.state_dict(),
					"step": step,
					"base_updates": base,
					"encoder": asdict(model.encoder.config),
					"scaler": train.scaler.to_dict(),
				},
				out / "stable_1.0.pt",
			)

		saved_model = copy.deepcopy(model.state_dict())
		saved_optimizer = copy.deepcopy(optimizer.state_dict())
		branch = range(decay_start, total)
		decaying = _decaying(warm, decay_start, decay_steps)
		_run(
			model,
			optimizer,
			train,
			config,
			schedule,
			branch,
			decaying,
			valid,
			history,
			on_step,
			f"branch_{frac:g}",
			None,
		)

		path = out / f"branch_{frac:g}.pt"
		torch.save(
			{
				"encoder": model.encoder.state_dict(),
				"config": asdict(model.encoder.config),
				"scaler": train.scaler.to_dict(),
				"corpus": train.corpus_hash,
				"updates": total,
			},
			path,
		)

		written[frac] = path
		scores[frac] = evaluate_watch(model, samples, valid)
		model.load_state_dict(saved_model)
		optimizer.load_state_dict(saved_optimizer)

	return PretrainResult(history, written, scores)


def calibrate(
	model: PretrainModel,
	corpus: PretrainCorpus,
	config: PretrainConfig,
	steps: int = 100,
	on_step: Callable[[dict[str, float | str]], None] | None = None,
) -> Calibration:
	"""Train through warmup."""
	device = next(model.parameters()).device
	ones = replace(
		config, w_mlm=1.0, w_load_tok=1.0, w_load_pool=1.0, clip=float("inf"), mlm=True, load=True
	)

	optimizer = _optimizer(model, ones)
	warm = max(1, round(config.warmup * config.updates))
	valid = _valid(corpus, device)
	schedule = _schedule(corpus, ones, model, warm + steps)
	_run(
		model,
		optimizer,
		corpus,
		ones,
		schedule,
		range(warm),
		_constant(warm),
		valid,
		[],
		on_step,
		"warmup",
		None,
	)

	shared = model.shared()
	everything = list(model.parameters())
	norms, products, seconds = [], [], []

	for _, (mlm_batch, load_batch) in _loader(
		corpus, ones, schedule, range(warm, warm + steps), model
	):
		start = time.perf_counter()

		assert mlm_batch is not None and load_batch is not None

		mlm_batch, load_batch = mlm_batch.to(device), load_batch.to(device)
		grads = []

		optimizer.zero_grad(set_to_none=True)
		model.mlm_loss(mlm_batch).backward()

		grads.append(_grads(everything))
		optimizer.zero_grad(set_to_none=True)

		token_loss, pool_loss = model.load_losses(load_batch, valid)
		token_loss.backward(retain_graph=True)

		grads.append(_grads(everything))
		optimizer.zero_grad(set_to_none=True)
		pool_loss.backward()
		grads.append(_grads(everything))

		index = {id(p): i for i, p in enumerate(everything)}
		norms.append([_norm([g[index[id(p)]] for p in shared]) for g in grads])
		products.append([[_dot(a, b) for b in grads] for a in grads])

		for p, total in zip(
			everything, (sum(parts) for parts in zip(*grads, strict=True)), strict=True
		):
			p.grad = total

		_set_lr(optimizer, config.lr)
		optimizer.step()
		seconds.append(time.perf_counter() - start)

	g_mlm, g_tok, g_pool = np.mean(np.array(norms), axis=0)
	w = np.array([g_tok / g_mlm, 1.0, POOL_SHARE * g_tok / g_pool])
	totals = [float(np.sqrt(max(w @ np.array(p) @ w, 0.0))) for p in products]

	return Calibration(
		w_mlm=float(w[0]),
		w_load_tok=1.0,
		w_load_pool=float(w[2]),
		clip=float(np.quantile(totals, CLIP_QUANTILE)),
		g_mlm=float(g_mlm),
		g_load_tok=float(g_tok),
		g_load_pool=float(g_pool),
		seconds_per_step=float(np.mean(seconds)),
	)


def _run(
	model: PretrainModel,
	optimizer: torch.optim.Optimizer,
	corpus: PretrainCorpus,
	config: PretrainConfig,
	schedule: Sequence[np.ndarray],
	steps: range,
	factor: Callable[[int], float],
	valid: Mapping[str, Tensor],
	history: list[dict[str, float | str]],
	on_step: Callable[[dict[str, float | str]], None] | None,
	line: str,
	samples: Sequence | None,
) -> None:
	device = next(model.parameters()).device
	model.train()

	for step, (mlm_batch, load_batch) in _loader(corpus, config, schedule, steps, model):
		start = time.perf_counter()
		optimizer.zero_grad(set_to_none=True)
		record: dict[str, float | str] = {"line": line, "step": step}

		if mlm_batch is not None:
			loss = model.mlm_loss(mlm_batch.to(device))
			(config.w_mlm * loss).backward()
			record["mlm"] = float(loss.detach())

		if load_batch is not None:
			token_loss, pool_loss = model.load_losses(load_batch.to(device), valid)
			(config.w_load_tok * token_loss + config.w_load_pool * pool_loss).backward()
			record["load_token"] = float(token_loss.detach())
			record["load_pool"] = float(pool_loss.detach())

		norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip))
		_set_lr(optimizer, config.lr * factor(step))
		optimizer.step()

		record |= {
			"grad_norm": norm,
			"clipped": float(norm > config.clip),
			"lr": config.lr * factor(step),
			"seconds": time.perf_counter() - start,
		}
		history.append(record)

		if samples is not None and config.watch_every and (step + 1) % config.watch_every == 0:
			history.append({"line": "watch", "step": step, **evaluate_watch(model, samples, valid)})

		if on_step is not None:
			on_step(record)


class _Steps(Dataset):
	"""Each step's two batches, built in a worker from plain paths."""

	def __init__(self, items: list[tuple[int, list[str]]], prepare_args: dict) -> None:
		self.items = items
		self.prepare_args = prepare_args

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, index: int) -> tuple[int, tuple[MLMBatch | None, LoadBatch | None]]:
		step, paths = self.items[index]
		return step, prepare(paths, step, **self.prepare_args)


def _loader(
	corpus: PretrainCorpus,
	config: PretrainConfig,
	schedule: Sequence[np.ndarray],
	steps: range,
	model: PretrainModel,
) -> DataLoader:
	items = [(s, charts_of(corpus, schedule[s])) for s in steps]
	args = {
		"seed": config.seed,
		"augment": Augment(config.mirror, config.rate_fraction, config.rates, config.judgment_drop),
		"segment": model.encoder.config.segment,
		"scaler": corpus.scaler,
		"load_scaler": corpus.load_scaler,
		"mlm": config.mlm,
		"load": config.load,
	}

	return DataLoader(
		_Steps(items, args),
		batch_size=None,
		shuffle=False,
		num_workers=config.workers,
		persistent_workers=False,
	)


def _schedule(
	corpus: PretrainCorpus, config: PretrainConfig, model: PretrainModel, steps: int
) -> list[np.ndarray]:
	"""The charts of steps 0..steps-1: whole passes over the corpus, each shuffled by its epoch."""
	out: list[np.ndarray] = []
	epoch = 0

	while len(out) < steps:
		out += epoch_batches(
			corpus, config.token_budget, model.encoder.config.segment, config.seed, epoch
		)
		epoch += 1

	return out[:steps]


def _constant(warm: int) -> Callable[[int], float]:
	return lambda step: wsd_factor(step, warm, None, 0)


def _decaying(warm: int, start: int, steps: int) -> Callable[[int], float]:
	return lambda step: wsd_factor(step, warm, start, steps)


def _optimizer(model: nn.Module, config: PretrainConfig) -> torch.optim.AdamW:
	decay = [p for p in model.parameters() if p.ndim >= 2]
	plain = [p for p in model.parameters() if p.ndim < 2]

	return torch.optim.AdamW(
		[
			{"params": decay, "weight_decay": config.weight_decay},
			{"params": plain, "weight_decay": 0.0},
		],
		lr=config.lr,
	)


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
	for group in optimizer.param_groups:
		group["lr"] = lr


def _valid(corpus: PretrainCorpus, device: torch.device) -> dict[str, Tensor]:
	return {
		name: torch.as_tensor(mask, device=device)
		for name, mask in corpus.load_scaler.valid.items()
	}


def _grads(params: Sequence[nn.Parameter]) -> list[Tensor]:
	return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]


def _norm(grads: Sequence[Tensor]) -> float:
	return float(torch.stack([(g.double() ** 2).sum() for g in grads]).sum().sqrt())


def _dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> float:
	return float(
		torch.stack([(x.double() * y.double()).sum() for x, y in zip(a, b, strict=True)]).sum()
	)

"""Pretraining samples."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import Tensor
from urc_converter import parse, scale_rate

from ..inputs import LayoutInput, layout_input, layouts, mirror, tokenize
from ..network import LayoutBatch, Scaler, collate, token_batches
from .heads import MLM_STATES, LoadTargets
from .load import chart_load, layout_load
from .masking import (
	mask_future_holds,
	mask_mlm,
	mlm_labels,
	mlm_spans,
	prefix_ends,
	prefix_weights,
	pretraining_view,
)
from .scaling import LoadScaler

# The purposes a step's draws are seeded by.
BATCHING, AUGMENTATION, MLM_MASK, PREFIX_END = 0, 1, 2, 3


@dataclass(frozen=True)
class PretrainCorpus:
	"""One part of a pretraining corpus."""

	charts: pl.DataFrame
	charts_root: Path
	scaler: Scaler
	load_scaler: LoadScaler
	state_frequencies: dict[str, np.ndarray]
	corpus_hash: str


@dataclass(frozen=True)
class Augment:
	"""What a step does to the charts it draws."""

	mirror: float
	rate_fraction: float
	rates: tuple[float, float]
	judgment_drop: float


@dataclass(frozen=True)
class MLMBatch:
	"""The masked pass."""

	batch: LayoutBatch
	labels: Tensor

	def to(self, device: torch.device | str) -> MLMBatch:
		"""The same batch on `device`."""
		return MLMBatch(self.batch.to(device), self.labels.to(device))


@dataclass(frozen=True)
class LoadBatch:
	"""The prefix pass."""

	batch: LayoutBatch
	prefix_end: Tensor
	weights: Tensor
	n_tokens: Tensor
	targets: LoadTargets

	def to(self, device: torch.device | str) -> LoadBatch:
		"""The same batch on `device`."""
		return LoadBatch(
			self.batch.to(device),
			self.prefix_end.to(device),
			self.weights.to(device),
			self.n_tokens.to(device),
			self.targets.to(device),
		)


def load_corpus(path: Path, charts_root: Path, part: str) -> PretrainCorpus:
	"""The `part` (train or watch) of the corpus wrote."""
	charts = pl.read_parquet(path / "charts.parquet").filter(pl.col("part") == part)
	manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

	return PretrainCorpus(
		charts=charts,
		charts_root=charts_root,
		scaler=Scaler.from_dict(torch.load(path / "scaler.pt", weights_only=True)),
		load_scaler=LoadScaler.from_dict(torch.load(path / "load_scaler.pt", weights_only=True)),
		state_frequencies={
			key: np.array(values) for key, values in manifest["state_frequencies"].items()
		},
		corpus_hash=path.name,
	)


def step_rng(seed: int, purpose: int, step: int) -> np.random.Generator:
	"""The generator of one purpose's draws at one step."""
	return np.random.default_rng([seed, purpose, step])


def epoch_batches(
	corpus: PretrainCorpus, budget: int, segment: int, seed: int, epoch: int
) -> list[np.ndarray]:
	"""The chart indices of every step of one pass over the corpus."""
	n_layouts = np.array([len(layouts(int(n))) for n in corpus.charts["lanes"]])
	rng = np.random.default_rng([seed, BATCHING, epoch])

	return token_batches(corpus.charts["n_tokens"].to_numpy(), n_layouts, budget, segment, rng)


def charts_of(corpus: PretrainCorpus, rows: np.ndarray) -> list[str]:
	"""The chart files of `rows`, as plain paths for workers."""
	return [str(corpus.charts_root / corpus.charts["urc_path"][int(r)]) for r in rows]


def prepare(
	paths: Sequence[str],
	step: int,
	seed: int,
	augment: Augment,
	segment: int,
	scaler: Scaler,
	load_scaler: LoadScaler,
	mlm: bool,
	load: bool,
) -> tuple[MLMBatch | None, LoadBatch | None]:
	"""The two passes' batches of one step; a pass that is off is not built."""
	rng = step_rng(seed, AUGMENTATION, step)
	charts = []

	for path in paths:
		chart = parse(Path(path).read_text(encoding="utf-8"))
		flip = rng.random() < augment.mirror
		scaled = rng.random() < augment.rate_fraction
		rate = round(float(np.exp(rng.uniform(*np.log(augment.rates)))), 2)
		drop = rng.random() < augment.judgment_drop

		if flip:
			chart = mirror(chart)

		if scaled and rate != 1.0:
			chart = scale_rate(chart, rate).chart

		charts.append((chart, tokenize(chart), drop))

	mlm_batch = _mlm(charts, step, seed, scaler) if mlm else None
	load_batch = _load(charts, step, seed, segment, scaler, load_scaler) if load else None

	return mlm_batch, load_batch


def watch_samples(
	corpus: PretrainCorpus, seed: int, segment: int, budget: int
) -> list[tuple[MLMBatch, LoadBatch, dict]]:
	"""Fixed batches of the watch charts as they are."""
	rows = epoch_batches(corpus, budget, segment, seed, 0)
	out = []
	none = Augment(mirror=0.0, rate_fraction=0.0, rates=(1.0, 1.0), judgment_drop=0.0)

	for step, chosen in enumerate(rows):
		paths = charts_of(corpus, chosen)
		mlm_batch, load_batch = prepare(
			paths, step, seed, none, segment, corpus.scaler, corpus.load_scaler, True, True
		)

		assert mlm_batch is not None and load_batch is not None
		extra = _watch_extra(paths, mlm_batch, corpus)
		out.append((mlm_batch, load_batch, extra))

	return out


def _mlm(charts: list[tuple], step: int, seed: int, scaler: Scaler) -> MLMBatch:
	rng = step_rng(seed, MLM_MASK, step)
	inputs, labels = [], []

	for _, tokens, drop in charts:
		masked = mlm_spans(tokens.times.size, rng)
		chart_inputs = [
			pretraining_view(mask_mlm(_dropped(layout_input(tokens, m), drop), tokens, masked))
			for m in layouts(tokens.lanes)
		]

		inputs.append(chart_inputs)
		labels += [mlm_labels(tokens, masked)] * len(chart_inputs)

	batch = collate(inputs, scaler, 0.0, None)
	return MLMBatch(batch, _pad_labels(labels, batch.state.shape))


def _load(
	charts: list[tuple],
	step: int,
	seed: int,
	segment: int,
	scaler: Scaler,
	load_scaler: LoadScaler,
) -> LoadBatch:
	rng = step_rng(seed, PREFIX_END, step)
	inputs, ends, weights, counts, finger, pair, judgment, visual, fingers = ([] for _ in range(9))

	for chart, tokens, drop in charts:
		n = tokens.times.size
		chart_ends = prefix_ends(n, segment, rng)
		chart_weights = prefix_weights(n, segment, chart_ends)

		whole = chart_load(chart, tokens)
		chart_inputs = []

		for m in layouts(tokens.lanes):
			x = _dropped(layout_input(tokens, m), drop)
			chart_inputs.append(mask_future_holds(pretraining_view(x), tokens, chart_ends, segment))
			per_layout = layout_load(tokens, m)

			finger.append(load_scaler.apply("finger", per_layout.finger))
			pair.append(load_scaler.apply("pair", per_layout.pair))
			fingers.append(per_layout.fingers)
			judgment.append(load_scaler.apply("judgment", whole.judgment))
			visual.append(load_scaler.apply("visual", whole.visual))
			ends.append(chart_ends)
			weights.append(chart_weights)
			counts.append(n)

		inputs.append(chart_inputs)

	batch = collate(inputs, scaler, 0.0, None)
	b, t = batch.state.shape[:2]
	segments = max(e.size for e in ends)

	return LoadBatch(
		batch=batch,
		prefix_end=_stack(ends, (b, segments), np.int64),
		weights=_stack(weights, (b, t), np.float32),
		n_tokens=torch.tensor(counts),
		targets=_targets(finger, pair, judgment, visual, fingers, t),
	)


def _targets(
	finger: list, pair: list, judgment: list, visual: list, fingers: list, t: int
) -> LoadTargets:
	b = len(finger)
	k = max(f.shape[1] for f in finger)
	p = max(q.shape[1] for q in pair)

	out = {
		"finger": np.zeros((b, t, k, *finger[0].shape[2:]), np.float32),
		"pair": np.zeros((b, t, p, *pair[0].shape[2:]), np.float32),
		"judgment": np.zeros((b, t, *judgment[0].shape[1:]), np.float32),
		"visual": np.zeros((b, t, *visual[0].shape[1:]), np.float32),
		"fingers": np.zeros((b, k, 2), np.int64),
		"finger_mask": np.zeros((b, k), bool),
		"pair_mask": np.zeros((b, p), bool),
	}

	for i in range(b):
		n, kf = finger[i].shape[:2]
		kp = pair[i].shape[1]

		out["finger"][i, :n, :kf] = finger[i]
		out["pair"][i, :n, :kp] = pair[i]
		out["judgment"][i, :n] = judgment[i]
		out["visual"][i, :n] = visual[i]
		out["fingers"][i, :kf] = fingers[i]
		out["finger_mask"][i, :kf] = True
		out["pair_mask"][i, :kp] = True

	return LoadTargets(**{name: torch.from_numpy(value) for name, value in out.items()})


def _watch_extra(paths: Sequence[str], mlm_batch: MLMBatch, corpus: PretrainCorpus) -> dict:
	"""Lane counts and smoothed per-lane state frequencies of each layout's unmasked cells."""
	labels = mlm_batch.labels.numpy()
	state = mlm_batch.batch.state.numpy()
	real = mlm_batch.batch.token_mask.numpy()
	lanes = mlm_batch.batch.lane_mask.numpy().sum(axis=1)
	frequencies = np.zeros((labels.shape[0], labels.shape[2], MLM_STATES), np.float32)

	for i in range(labels.shape[0]):
		prior = corpus.state_frequencies.get(str(int(lanes[i])), corpus.state_frequencies["all"])
		visible = (labels[i] < 0) & real[i][:, None]

		for lane in range(int(lanes[i])):
			counts = np.bincount(state[i][visible[:, lane], lane], minlength=MLM_STATES)
			frequencies[i, lane] = (counts + prior) / (counts.sum() + 1.0)

	past = []
	for path in paths:
		chart = parse(Path(path).read_text(encoding="utf-8"))
		tokens = tokenize(chart)
		whole = chart_load(chart, tokens, past=True)

		for m in layouts(tokens.lanes):
			per_layout = layout_load(tokens, m, past=True)
			past.append(
				{
					"finger": corpus.load_scaler.apply("finger", per_layout.finger),
					"pair": corpus.load_scaler.apply("pair", per_layout.pair),
					"judgment": corpus.load_scaler.apply("judgment", whole.judgment),
					"visual": corpus.load_scaler.apply("visual", whole.visual),
				}
			)

	return {"lanes": lanes, "frequencies": frequencies, "past": past}


def _dropped(x: LayoutInput, drop: bool) -> LayoutInput:
	return replace(x, judgment=None) if drop else x


def _pad_labels(labels: list[np.ndarray], shape: tuple[int, ...]) -> Tensor:
	out = np.full(shape, -1, np.int64)

	for i, lab in enumerate(labels):
		out[i, : lab.shape[0], : lab.shape[1]] = lab

	return torch.from_numpy(out)


def _stack(rows: list[np.ndarray], shape: tuple[int, int], dtype: type) -> Tensor:
	out = np.zeros(shape, dtype)

	for i, row in enumerate(rows):
		out[i, : row.size] = row

	return torch.from_numpy(out)

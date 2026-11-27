"""Pretrain the chart encoder on the pretraining corpus.

uv run scripts/pretrain.py +experiment=pretrain_all mode=calibrate updates=20200
uv run scripts/pretrain.py +experiment=pretrain_all mode=train updates=20200 \
	calibration=runs/pretrain_all/calibrate-L6
uv run scripts/pretrain.py +experiment=pretrain_all mode=train updates=20200 \
	calibration=runs/pretrain_all/calibrate-L6 branches=[2.0] \
	resume=runs/pretrain_all/all-L6-seed20261127/stable_1.0.pt
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from omegaconf import DictConfig

from vsrgrating.config import compose_config
from vsrgrating.encoder.network import (
	ChartEncoder,
	EncoderConfig,
	LoadHead,
	MLMHead,
	PretrainConfig,
	PretrainCorpus,
	PretrainModel,
	PretrainResult,
	Scaler,
	calibrate,
	load_corpus,
	pretrain,
)
from vsrgrating.tracking import run_output, write_metrics

LOSSES = {(True, True): "all", (False, True): "no_mlm", (True, False): "no_load"}
CALIBRATED = ("w_mlm", "w_load_tok", "w_load_pool", "clip")
PROGRESS_EVERY = 100

type Record = tuple[dict[str, float], dict[str, list[float]]]
type OnStep = Callable[[dict[str, float | str]], None]


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("pretrain", overrides)
	label = run_label(cfg)
	check_new(cfg, label)

	device = torch.device("cuda")

	train = load_corpus(Path(cfg.corpus), Path(cfg.charts_root), "train")
	watch = load_corpus(Path(cfg.corpus), Path(cfg.charts_root), "watch")
	encoder = EncoderConfig(**cfg.model.encoder)
	config = pretrain_config(cfg, encoder, train)

	torch.manual_seed(cfg.seed)
	model = new_model(encoder).to(device)

	tags = {
		"corpus_hash": train.corpus_hash,
		"device": torch.cuda.get_device_name(device),
		"mode": str(cfg.mode),
		"losses": LOSSES[(config.mlm, config.load)],
		"calibration": str(cfg.calibration),
		"resume": str(cfg.resume),
	}

	with run_output(cfg, overrides, label) as out:
		start = time.perf_counter()

		if cfg.mode == "calibrate":
			metrics, series = run_calibration(model, train, config, encoder, out)
		else:
			metrics, series = run_training(cfg, model, train, watch, config, out)

		metrics |= {
			"parameters": float(sum(p.numel() for p in model.encoder.parameters())),
			"runtime_s": time.perf_counter() - start,
		}
		write_metrics(out, tags, metrics, series)


def run_label(cfg: DictConfig) -> str:
	layers = cfg.model.encoder.layers

	match cfg.mode:
		case "calibrate":
			return f"calibrate-L{layers}"
		case "train":
			key = (bool(cfg.losses.mlm), bool(cfg.losses.load))
			if key not in LOSSES:
				raise ValueError("at least one of losses.mlm and losses.load must be on")

			label = f"{LOSSES[key]}-L{layers}-seed{cfg.seed}"
			if cfg.resume is None:
				return label

			return f"{label}-ext{max(cfg.branches):g}"

	raise ValueError(f"mode must be calibrate or train, got {cfg.mode!r}")


def check_new(cfg: DictConfig, label: str) -> None:
	"""Fails before any data is read when the local run already exists."""
	out = Path(cfg.runs_root) / cfg.name / label

	if not cfg.tracking.mlflow and out.exists():
		raise FileExistsError(f"{out} already holds a run")


def pretrain_config(
	cfg: DictConfig, encoder: EncoderConfig, corpus: PretrainCorpus
) -> PretrainConfig:
	"""The run's settings; training takes its weights and clip from a calibration run."""
	weights = dict.fromkeys(CALIBRATED, 1.0)

	if cfg.mode == "train":
		if cfg.calibration is None:
			raise ValueError("mode=train needs calibration=<calibration run>")

		weights = calibrated(Path(cfg.calibration), encoder, corpus)
		if cfg.resume is not None:
			check_resume(Path(cfg.resume), encoder, corpus.scaler)

	return PretrainConfig(
		lr=cfg.optim.lr,
		warmup=cfg.optim.warmup,
		decay=cfg.optim.decay,
		weight_decay=cfg.optim.weight_decay,
		token_budget=cfg.optim.token_budget,
		updates=cfg.updates,
		mlm=cfg.losses.mlm,
		load=cfg.losses.load,
		seed=cfg.seed,
		workers=cfg.workers,
		watch_every=cfg.watch_every or max(1, cfg.updates // 20),
		w_mlm=weights["w_mlm"],
		w_load_tok=weights["w_load_tok"],
		w_load_pool=weights["w_load_pool"],
		clip=weights["clip"],
	)


def calibrated(path: Path, encoder: EncoderConfig, corpus: PretrainCorpus) -> dict[str, float]:
	"""The weights and clip of a calibration run of the same encoder on the same corpus."""
	record = json.loads((path / "calibration.json").read_text(encoding="utf-8"))

	if record["encoder"] != asdict(encoder):
		raise ValueError(f"{path} calibrated {record['encoder']}, not {asdict(encoder)}")
	if record["corpus_hash"] != corpus.corpus_hash:
		raise ValueError(f"{path} calibrated on corpus {record['corpus_hash']}")

	return {name: float(record[name]) for name in CALIBRATED}


def check_resume(path: Path, encoder: EncoderConfig, scaler: Scaler) -> None:
	"""Fails when the main line to resume had another encoder or input scaler."""
	state = torch.load(path, weights_only=True)

	if state["encoder"] != asdict(encoder):
		raise ValueError(f"{path} holds the encoder {state['encoder']}, not {asdict(encoder)}")

	if Scaler.from_dict(state["scaler"]).digest() != scaler.digest():
		raise ValueError(f"{path} was trained with another input scaler")


def new_model(config: EncoderConfig) -> PretrainModel:
	encoder = ChartEncoder(config)
	return PretrainModel(
		encoder, MLMHead(config.d_model, config.d_lane), LoadHead(config.d_model, encoder.width)
	)


def run_calibration(
	model: PretrainModel,
	train: PretrainCorpus,
	config: PretrainConfig,
	encoder: EncoderConfig,
	out: Path,
) -> Record:
	warm = max(1, round(config.warmup * config.updates))
	result = asdict(calibrate(model, train, config, on_step=progress(warm)))

	record = {**result, "encoder": asdict(encoder), "corpus_hash": train.corpus_hash}
	(out / "calibration.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
	print(", ".join(f"{name} {value:.4g}" for name, value in result.items()), flush=True)

	return result, {}


def run_training(
	cfg: DictConfig,
	model: PretrainModel,
	train: PretrainCorpus,
	watch: PretrainCorpus,
	config: PretrainConfig,
	out: Path,
) -> Record:
	branches = tuple(float(f) for f in cfg.branches)
	resume = None if cfg.resume is None else Path(cfg.resume)

	base, start = config.updates, 0

	if resume is not None:
		state = torch.load(resume, weights_only=True)
		base, start = int(state["base_updates"]), int(state["step"])

	steps = planned_steps(base, start, branches, config.decay)
	result = pretrain(model, train, watch, config, branches, out, resume, progress(steps))

	return training_record(result)


def planned_steps(base: int, start: int, branches: Sequence[float], decay: float) -> int:
	"""The main line from `start` to the last decay start, and every branch's decay."""
	totals = [round(f * base) for f in branches]
	decays = [round(decay * total) for total in totals]

	return max(t - d for t, d in zip(totals, decays, strict=True)) - start + sum(decays)


def progress(planned: int) -> OnStep:
	"""Prints every `PROGRESS_EVERY` steps and the last: losses, elapsed and remaining time."""
	start = time.perf_counter()
	done = 0

	def on_step(record: dict[str, float | str]) -> None:
		nonlocal done
		done += 1
		if done % PROGRESS_EVERY and done != planned:
			return

		elapsed = time.perf_counter() - start
		losses = ", ".join(
			f"{name} {float(record[name]):.4f}"
			for name in ("mlm", "load_token", "load_pool")
			if name in record
		)
		print(
			f"{record['line']} step {record['step']} ({done}/{planned}): {losses},"
			f" {elapsed:.0f} s, {elapsed / done * (planned - done):.0f} s left",
			flush=True,
		)

	return on_step


def training_record(result: PretrainResult) -> Record:
	"""Each branch's watch scores and each line's clip rate and step time as metrics; every
	line's step records and the watch curve as series."""
	lines: dict[str, list[dict[str, float | str]]] = {}

	for record in result.history:
		lines.setdefault(str(record["line"]), []).append(record)

	metrics = {
		f"branch_{frac:g}.{name}": value
		for frac, scores in result.watch.items()
		for name, value in scores.items()
	}

	for line, records in lines.items():
		if line != "watch":
			metrics[f"{line}.clipped_fraction"] = float(np.mean([r["clipped"] for r in records]))
			metrics[f"{line}.seconds_per_step"] = float(np.mean([r["seconds"] for r in records]))

	series = {
		f"{line}.{name}": [float(r[name]) for r in records]
		for line, records in lines.items()
		for name in records[0]
		if name != "line"
	}

	return metrics, series


if __name__ == "__main__":
	typer.run(main)

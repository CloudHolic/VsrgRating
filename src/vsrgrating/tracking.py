"""Where a fit's outputs go: a run directory under `runs_root`, or an MLflow run."""

import json
import os
import shutil
import tempfile
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from omegaconf import DictConfig, OmegaConf

ARTIFACT_SCHEME = "mlflow-artifacts:"
METRICS = "metrics.json"


@contextmanager
def run_output(cfg: DictConfig, overrides: Sequence[str], response: str) -> Generator[Path]:
	"""A directory for one fit's outputs, starting with config.yaml and overrides.yaml.

	- local: `runs_root/name/response`, staged and renamed into place only when the fit succeeds.
	- mlflow: a temporary directory uploaded with metrics.json to the run `<name>-<response>`
	when the fit succeeds; a failed fit leaves the run FAILED.
	"""
	run = _mlflow_run if cfg.tracking.mlflow else _local_run
	with run(cfg, response) as out:
		OmegaConf.save(cfg, out / "config.yaml")
		OmegaConf.save(OmegaConf.create(list(overrides)), out / "overrides.yaml")
		yield out


def write_metrics(
	out: Path,
	tags: Mapping[str, str],
	metrics: Mapping[str, float],
	series: Mapping[str, Sequence[float]],
) -> None:
	"""metrics.json: what the fit observed. An MLflow run gets the same tags, metrics and series."""
	record = {
		"tags": dict(tags),
		"metrics": {k: float(v) for k, v in metrics.items()},
		"series": {k: [float(v) for v in values] for k, values in series.items()},
	}
	(out / METRICS).write_text(json.dumps(record, indent=2), encoding="utf-8")


@contextmanager
def _local_run(cfg: DictConfig, response: str) -> Generator[Path]:
	target = Path(cfg.runs_root) / cfg.name / response
	if target.exists():
		raise FileExistsError(f"{target} already holds a run")

	staging = target.with_name(f".staging-{response}-{os.getpid()}")
	staging.mkdir(parents=True)

	try:
		yield staging
	except BaseException:
		shutil.rmtree(staging)
		raise

	staging.rename(target)


@contextmanager
def _mlflow_run(cfg: DictConfig, response: str) -> Generator[Path]:
	client = MlflowClient(cfg.tracking.uri)
	experiment_id = _experiment_id(client, cfg.tracking.experiment)
	run = client.create_run(experiment_id, run_name=f"{cfg.name}-{response}")
	run_id = run.info.run_id

	try:
		resolved = OmegaConf.to_container(cfg, resolve=True)

		assert isinstance(resolved, dict)

		del resolved["tracking"]
		client.log_batch(run_id, params=[Param(k, str(v)) for k, v in _flatten(resolved).items()])

		with tempfile.TemporaryDirectory() as tmp:
			out = Path(tmp)
			yield out

			_upload(client, run_id, out)
	except BaseException:
		client.set_terminated(run_id, status="FAILED")
		raise

	client.set_terminated(run_id, status="FINISHED")


def _upload(client: MlflowClient, run_id: str, out: Path) -> None:
	record = json.loads((out / METRICS).read_text(encoding="utf-8"))
	now = int(time.time() * 1000)

	metrics = [Metric(k, v, now, 0) for k, v in record["metrics"].items()]
	metrics += [
		Metric(k, v, now, step)
		for k, values in record["series"].items()
		for step, v in enumerate(values)
	]
	tags = [RunTag(k, v) for k, v in record["tags"].items()]

	client.log_batch(run_id, metrics=metrics, tags=tags)
	client.log_artifacts(run_id, str(out))


def _flatten(tree: dict[Any, Any], prefix: str = "") -> dict[str, Any]:
	flat: dict[str, Any] = {}

	for key, value in tree.items():
		name = f"{prefix}{key}"

		if isinstance(value, dict):
			flat.update(_flatten(value, f"{name}."))
		else:
			flat[name] = value

	return flat


def _experiment_id(client: MlflowClient, name: str) -> str:
	# The lookup doubles as the connectivity check: it fails before any fitting starts.
	experiment = client.get_experiment_by_name(name)
	if experiment is None:
		experiment = client.get_experiment(client.create_experiment(name))

	if not experiment.artifact_location.startswith(ARTIFACT_SCHEME):
		raise RuntimeError(
			f"experiment {name!r} stores artifacts at {experiment.artifact_location!r}, "
			"not on the server; use a new experiment name"
		)

	return experiment.experiment_id

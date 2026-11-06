"""Pick the P-spline penalty on a person hold-out inside train; the test split is never read.

uv run scripts/tune_penalty.py +experiment=core300_pspline_penalty
"""

import dataclasses
import time
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import typer
from omegaconf import DictConfig

from vsrgrating.config import compose_config
from vsrgrating.data import Split, load_split, locate_dataset, split_persons
from vsrgrating.evaluation import Fitted, paired_difference, person_posterior
from vsrgrating.fitting import fit_map_split, map_settings
from vsrgrating.irt.cnrm.transform import TransformSpec
from vsrgrating.irt.map import MAPConfig, Model
from vsrgrating.tracking import run_output, write_metrics

type Splits = dict[str, tuple[Split, Split]]


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	overrides = overrides or []
	cfg = compose_config("tune_penalty", overrides)
	jax.config.update("jax_enable_x64", True)
	base = base_spec(cfg)
	settings = map_settings(cfg)

	penalties = [float(p) for p in cfg.penalties]
	if len(set(penalties)) != len(penalties):
		raise ValueError(f"penalties repeat: {penalties}")

	seed = cfg.seed
	dataset = locate_dataset(Path(cfg.dataset_root), cfg.dataset.name)
	splits = {
		r: split_persons(load_split(dataset, r, "train"), cfg.holdout_fraction, seed)
		for r in cfg.responses
	}
	check_held(splits)

	with run_output(cfg, overrides, "_".join(cfg.responses)) as out:
		tune, exponent, joint_ll = sweep(base, settings, splits, penalties)
		joint, best = joint_table(joint_ll)
		chosen = one_se(joint, best)
		print(f"held-out maximum: {best:g}, chosen penalty: {chosen:g}", flush=True)

		tune.write_csv(out / "tune.csv")
		joint.write_csv(out / "joint.csv")
		exponent.write_csv(out / "exponent.csv")
		write_metrics(out, run_tags(cfg, splits), run_metrics(splits, best, chosen), {})


def base_spec(cfg: DictConfig) -> TransformSpec:
	"""The pspline transform whose penalty the sweep replaces."""
	if cfg.model.family != "cnrm" or cfg.transform.kind != "pspline" or cfg.estimator.kind != "map":
		raise ValueError("tune_penalty fits the cnrm with a pspline transform by map")

	return TransformSpec.from_config(cfg.transform)


def check_held(splits: Splits) -> None:
	"""The joint held-out adds each person's responses, so every response holds out the same
	persons."""
	held = [h.person_idx for _, h in splits.values()]
	if any(not np.array_equal(held[0], h) for h in held[1:]):
		raise ValueError("the responses hold out different persons")


def sweep(
	base: TransformSpec, settings: MAPConfig, splits: Splits, penalties: list[float]
) -> tuple[pl.DataFrame, pl.DataFrame, dict[float, np.ndarray]]:
	"""Every penalty fitted to each response's kept persons and scored on its held persons."""
	rows, exponents, joint = [], [], {}

	for penalty in penalties:
		spec = dataclasses.replace(base, penalty=penalty)
		lls = []
		for response, (kept, held) in splits.items():
			row, exponent, ll = fit_and_score(spec, settings, kept, held)
			rows.append({"response": response, **row})
			exponents.append(exponent.with_columns(response=pl.lit(response)))
			lls.append(ll)
			print(
				f"penalty {penalty:g} {response}: held-out {row['heldout_loglik']:.1f}", flush=True
			)
		joint[penalty] = np.sum(lls, axis=0)

	return pl.DataFrame(rows), pl.concat(exponents), joint


def fit_and_score(
	spec: TransformSpec, settings: MAPConfig, kept: Split, held: Split
) -> tuple[dict[str, float], pl.DataFrame, np.ndarray]:
	"""MAP from person-mean OLS on `kept`; each held person's log marginal, and q at the knots."""
	model = Model("cnrm", spec)

	start = time.perf_counter()
	result = fit_map_split(model, settings, kept)
	runtime = time.perf_counter() - start

	params = {k: jnp.asarray(v) for k, v in result.params.items()}
	raw = params["transform"]
	fitted = Fitted(f"pspline-{spec.penalty:g}", model, model.item_params(params), raw, "map")
	ll = person_posterior(fitted, held).log_marginal

	knots = np.asarray(spec.knots)
	# with unit weight, so that a zero penalty still reports its curve's roughness
	unit = dataclasses.replace(spec, penalty=1.0)
	row = {
		"penalty": spec.penalty,
		"heldout_loglik": float(ll.sum()),
		"per_cell": float(ll.sum() / held.x.size),
		"roughness": float(-2.0 * unit.log_penalty(raw)),
		"best_loss": result.best_loss,
		"best_step": result.best_step,
		"drop_last": result.drop_last,
		"runtime_s": runtime,
	}
	exponent = pl.DataFrame(
		{
			"penalty": np.full(knots.size, spec.penalty),
			"knot_s": knots,
			"one_minus_x": np.exp(knots),
			"q": np.asarray(spec.exponent(raw, jnp.asarray(knots))),
		}
	)
	return row, exponent, ll


def joint_table(joint: dict[float, np.ndarray]) -> tuple[pl.DataFrame, float]:
	"""Per penalty, the joint held-out and its paired difference from the best; and the best."""
	best = max(joint, key=lambda p: joint[p].sum())
	rows = []

	for penalty, ll in sorted(joint.items()):
		diff, se = paired_difference(joint[best], ll)
		rows.append(
			{"penalty": penalty, "heldout_loglik": float(ll.sum()), "diff_vs_best": diff, "se": se}
		)

	return pl.DataFrame(rows), best


def one_se(joint: pl.DataFrame, best: float) -> float:
	"""From the best penalty, step to larger ones while the held-out stays within one paired SE
	of the best, and keep the last. Near the optimum the curve is flat and its maximum alone
	drifts to the smallest penalty tried; stepping only through neighbors keeps one badly
	converged fit further out from being picked."""
	chosen = best
	larger = joint.filter(pl.col("penalty") > best).select("penalty", "diff_vs_best", "se")

	for penalty, diff, se in larger.iter_rows():
		if diff < -se:
			break
		chosen = penalty

	return chosen


def run_tags(cfg: DictConfig, splits: Splits) -> dict[str, str]:
	device = jax.devices()[0]
	kept, _ = next(iter(splits.values()))

	return {
		"responses": ",".join(cfg.responses),
		"dataset_hash": kept.dataset_hash,
		"device": f"{device.platform}:{device.device_kind}",
	}


def run_metrics(splits: Splits, best: float, chosen: float) -> dict[str, float]:
	kept, held = next(iter(splits.values()))
	metrics = {
		"best_penalty": best,
		"chosen_penalty": chosen,
		"persons_kept": kept.n_persons,
		"persons_held": held.n_persons,
	}

	return metrics | {f"cells_held.{r}": h.x.size for r, (_, h) in splits.items()}


if __name__ == "__main__":
	typer.run(main)

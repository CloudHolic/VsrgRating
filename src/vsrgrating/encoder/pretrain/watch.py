"""The watch diagnostics: both losses on fixed watch batches, against simple baselines."""

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional

from ..inputs import LaneState
from .data import LoadBatch, MLMBatch
from .load import LOAD_TAUS
from .model import PretrainModel
from .scaling import LOAD_FAMILIES


def evaluate_watch(
	model: PretrainModel,
	samples: Sequence[tuple[MLMBatch, LoadBatch, dict]],
	valid: Mapping[str, Tensor],
) -> dict[str, float]:
	"""Both losses on the watch charts and the diagnostics."""
	device = next(model.parameters()).device
	was_training = model.training
	model.eval()
	sums: dict[str, float] = {}

	def add(name: str, value: float, count: float) -> None:
		sums[f"{name}.sum"] = sums.get(f"{name}.sum", 0.0) + value
		sums[f"{name}.n"] = sums.get(f"{name}.n", 0.0) + count

	with torch.no_grad():
		for mlm_batch, load_batch, extra in samples:
			mlm_batch, load_batch = mlm_batch.to(device), load_batch.to(device)
			add("loss.mlm", float(model.mlm_loss(mlm_batch)), 1)

			token_loss, pool_loss = model.load_losses(load_batch, valid)
			add("loss.load_token", float(token_loss), 1)
			add("loss.load_pool", float(pool_loss), 1)

			_mlm_diagnostics(model, mlm_batch, extra, add)
			_load_diagnostics(model, load_batch, extra, valid, add)

	model.train(was_training)

	return {
		name[:-4]: sums[name] / max(sums[name[:-4] + ".n"], 1.0)
		for name in sums
		if name.endswith(".sum")
	}


def _mlm_diagnostics(model: PretrainModel, batch: MLMBatch, extra: dict, add: Callable) -> None:
	encoded = model.encoder.encode_layout(batch.batch)
	logits = model.mlm(encoded.token_repr, encoded.lane_repr)

	labels = batch.labels
	cells = labels >= 0

	ce = functional.cross_entropy(logits.permute(0, 3, 1, 2), labels.clamp_min(0), reduction="none")
	hit = logits.argmax(dim=-1) == labels
	frequencies = torch.as_tensor(extra["frequencies"], device=labels.device)

	base_p = torch.gather(
		frequencies[:, None].expand(-1, labels.shape[1], -1, -1), -1, labels.clamp_min(0)[..., None]
	)[..., 0]
	base_ce = -torch.log(base_p.clamp_min(1e-12))
	base_hit = frequencies.argmax(dim=-1)[:, None].expand_as(labels) == labels

	filled = cells & (labels != LaneState.EMPTY)

	for i, lanes in enumerate(extra["lanes"].tolist()):
		for key in (f"{lanes}K", "all"):
			n, m = float(cells[i].sum()), float(filled[i].sum())

			add(f"mlm.{key}.ce", float(ce[i][cells[i]].sum()), n)
			add(f"mlm.{key}.accuracy", float(hit[i][cells[i]].sum()), n)
			add(f"mlm.{key}.filled_accuracy", float(hit[i][filled[i]].sum()), m)
			add(f"mlm.{key}.baseline_ce", float(base_ce[i][cells[i]].sum()), n)
			add(f"mlm.{key}.baseline_accuracy", float(base_hit[i][cells[i]].sum()), n)


def _load_diagnostics(
	model: PretrainModel, batch: LoadBatch, extra: dict, valid: Mapping[str, Tensor], add: Callable
) -> None:
	tokens, _ = model.load_outputs(batch)
	supervised = batch.weights > 0

	for family in LOAD_FAMILIES:
		target = getattr(batch.targets, family)
		pred = getattr(tokens, family)
		past = torch.zeros_like(target)

		for i, layout in enumerate(extra["past"]):
			values = torch.as_tensor(layout[family], device=target.device)
			index = (i, slice(0, values.shape[0]), *(slice(0, n) for n in values.shape[1:-2]))
			past[index] = values

		mask = valid[family]

		if family in ("finger", "pair"):
			entity = getattr(batch.targets, f"{family}_mask")[:, None, :, None]
			cells = supervised[:, :, None, None] & entity
		else:
			cells = supervised[:, :, None]

		for t, tau in enumerate(LOAD_TAUS):
			keep = cells & mask[:, t]
			n = float(keep.sum())

			for name, guess in (
				("model", pred),
				("persistence", past),
				("constant", torch.zeros_like(target)),
			):
				loss = functional.huber_loss(
					guess[..., t], target[..., t], reduction="none", delta=1.0
				)
				add(f"load.{family}.tau{tau:g}.{name}", float(loss[keep].sum()), n)

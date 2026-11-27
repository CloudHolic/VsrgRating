"""The encoder with the pretraining heads, and the losses of its two passes."""

from collections.abc import Mapping

from torch import Tensor, nn

from ..network import ChartEncoder
from .data import LoadBatch, MLMBatch
from .heads import LoadHead, LoadOutput, MLMHead, load_pool_loss, load_token_loss, mlm_loss


class PretrainModel(nn.Module):
	"""The encoder with the masked-note and load heads."""

	def __init__(self, encoder: ChartEncoder, mlm: MLMHead, load: LoadHead) -> None:
		super().__init__()

		self.encoder = encoder
		self.mlm = mlm
		self.load = load

	def mlm_loss(self, batch: MLMBatch) -> Tensor:
		"""The masked pass's loss."""
		encoded = self.encoder.encode_layout(batch.batch)
		logits = self.mlm(encoded.token_repr, encoded.lane_repr)
		layouts = batch.batch

		return mlm_loss(logits, batch.labels, layouts.chart_of_layout, layouts.n_charts)

	def load_outputs(self, batch: LoadBatch) -> tuple[LoadOutput, LoadOutput]:
		"""The load to come at every token and from the first segment's pooled prefix."""
		encoded = self.encoder.encode_layout(batch.batch, batch.prefix_end)
		fingers, index = batch.targets.fingers, batch.batch.pair_index
		tokens = self.load(encoded.token_repr, fingers, index)
		pooled = self.encoder.pool_prefix(encoded, batch.prefix_end[:, 0])

		return tokens, self.load.pooled(pooled, fingers, index)

	def load_losses(self, batch: LoadBatch, valid: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
		"""The prefix pass's token and pooled losses."""
		tokens, pooled = self.load_outputs(batch)
		layouts = batch.batch

		token_loss = load_token_loss(
			tokens,
			batch.targets,
			valid,
			batch.weights,
			batch.n_tokens,
			layouts.chart_of_layout,
			layouts.n_charts,
		)

		pool_loss = load_pool_loss(
			pooled,
			batch.targets,
			batch.prefix_end[:, 0],
			valid,
			layouts.chart_of_layout,
			layouts.n_charts,
		)

		return token_loss, pool_loss

	def shared(self) -> list[nn.Parameter]:
		"""The encoder parameters both passes train: all but the pooling."""
		pooling = {id(p) for p in (self.encoder.queries, *self.encoder.pool_key.parameters())}
		pooling |= {id(p) for p in self.encoder.pool_value.parameters()}

		return [p for p in self.encoder.parameters() if id(p) not in pooling]

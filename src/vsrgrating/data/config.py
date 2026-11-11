"""Dataset build configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
	"""Everything that determines a dataset build, given the database contents."""

	name: str
	n_items: int
	min_item_responses: int
	min_person_responses: int
	min_item_train_responses: int
	n_persons: int | None
	test_fraction: float | None
	seed: int
	mapset_items: int = 0
	dan_courses: bool = False
	key_items: dict[int, int | None] | None = None

	def __post_init__(self) -> None:
		if (self.n_items is None) == (self.key_items is None):
			raise ValueError("give exactly one of n_items and key_items")

		if self.n_items is not None and self.n_items < 1:
			raise ValueError(f"n_items must be positive, got {self.n_items}")

		if self.key_items is not None:
			plain = {int(k): v for k, v in self.key_items.items()}
			if not plain or any(v is not None and v < 1 for v in plain.values()):
				raise ValueError(f"key_items needs positive counts or null, got {plain}")

			object.__setattr__(self, "key_items", plain)

		for field in ("min_item_responses", "min_person_responses", "min_item_train_responses"):
			if getattr(self, field) < 1:
				raise ValueError(f"{field} must be positive, got {getattr(self, field)}")

		if self.n_persons is not None and self.n_persons < 1:
			raise ValueError(f"n_persons must be positive or null, got {self.n_persons}")

		if self.test_fraction is not None and not 0.0 < self.test_fraction < 1.0:
			raise ValueError(f"test_fraction must be in (0, 1) or null, got {self.test_fraction}")

		if self.mapset_items < 0:
			raise ValueError(f"mapset_items must be non-negative, got {self.mapset_items}")

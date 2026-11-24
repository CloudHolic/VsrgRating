"""Split a dataset's items by mapset against a chart index snapshot; print the split's directory.

uv run scripts/split_mapsets.py chart_index=2026-11-24
"""

from pathlib import Path
from typing import Annotated

import typer

from vsrgrating.config import compose_config
from vsrgrating.data import build_mapset_split, locate_dataset


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	cfg = compose_config("split_mapsets", overrides or [])
	print(
		build_mapset_split(
			dataset_dir=locate_dataset(Path(cfg.dataset_root), cfg.dataset),
			charts_root=Path(cfg.charts_root),
			chart_index=cfg.chart_index,
			out_root=Path(cfg.out_root),
			seed=cfg.seed,
		)
	)


if __name__ == "__main__":
	typer.run(main)

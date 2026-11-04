"""Build a dataset from the osu! database and print its directory.

uv run --env-file .env scripts/build_dataset.py (dataset=core300)
"""

from pathlib import Path
from typing import Annotated

import typer

from vsrgrating.config import compose_config
from vsrgrating.data import DatasetConfig, build


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	cfg = compose_config("build_dataset", overrides or [])
	config = DatasetConfig(**cfg.dataset)
	print(build(config, cfg.dsn, Path(cfg.out_root)))


if __name__ == "__main__":
	typer.run(main)

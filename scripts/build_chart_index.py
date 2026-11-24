"""Convert every source chart the cache lacks and write a dated chart index; print its directory.

uv run --env-file .env scripts/build_chart_index.py
uv run --env-file .env scripts/build_chart_index.py name=2026-11-24 'sources=[osu_external,etterna]'
"""

from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from vsrgrating.charts import build_index
from vsrgrating.config import compose_config


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	cfg = compose_config("build_chart_index", overrides or [])
	sources = list(cfg.sources)
	print(
		build_index(
			sources=sources,
			dsn=cfg.dsn if "osu_archive" in sources else "",
			archive=Path(cfg.archive),
			external_root=Path(cfg.external_root),
			out_root=Path(cfg.out_root),
			name=cfg.name or date.today().isoformat(),
		)
	)


if __name__ == "__main__":
	typer.run(main)

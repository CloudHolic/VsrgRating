"""Apply the schema, then load osu! dump archives into PostgreSQL.

uv run --env-file .env scripts/database.py schema_only=true
uv run --env-file .env scripts/database.py dump_dir=/path/to/dumps (dry_run=true)
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import psycopg
import typer

from vsrgrating import db
from vsrgrating.config import compose_config


def main(overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
	cfg = compose_config("database", overrides or [])
	if not cfg.schema_only and cfg.dump_dir is None:
		raise SystemExit("pass dump_dir=..., or schema_only=true to apply the DDL alone")

	if not cfg.dry_run:
		apply_schema(cfg.dsn, cfg.schema)
	if cfg.schema_only:
		return

	paths = collect_dumps(Path(cfg.dump_dir))
	report_pools(paths)

	if cfg.dry_run:
		return

	require_pool_flags(paths)
	load_dumps(cfg.dsn, paths)


def apply_schema(dsn: str, schema: str) -> None:
	"""Apply schema.sql and views.sql to one schema."""
	with psycopg.connect(dsn) as conn:
		db.apply_all(conn, schema=schema)

	print(f"applied schema.sql and views.sql to {schema!r}")


def collect_dumps(dump_dir: Path) -> list[Path]:
	"""Every dump archive under `dump_dir`, oldest snapshot first."""
	# Sorted so the newest snapshot is loaded last and wins the upsert:
	# beatmap play counts change between dumps.
	paths = sorted(dump_dir.rglob("*.tar.bz2"))
	if not paths:
		raise SystemExit(f"no archives under {dump_dir}")

	return paths


def report_pools(paths: Sequence[Path]) -> None:
	"""The pool flags each archive's filename implies."""
	for path in paths:
		in_top, in_random = db.pool_flags(path)
		print(f"{path.name}: in_top={in_top} in_random={in_random}")


def require_pool_flags(paths: Sequence[Path]) -> None:
	"""Refuse a load where no archive can be attributed to a sampling pool."""
	if not any(any(db.pool_flags(p)) for p in paths):
		raise SystemExit(
			"no archive filename contains 'top' or 'random'; every row would carry both "
			"pool flags false and the random-pool filter would silently select nothing"
		)


def load_dumps(dsn: str, paths: Sequence[Path]) -> None:
	"""Load every archive, recording what each member contributed."""
	with psycopg.connect(dsn) as conn:
		for path in paths:
			print(f"\n{path.name}")
			results = db.load_dump(conn, path)
			db.record(conn, results)

			for r in results:
				print(
					f"  {r.member}: read {r.rows_read:,} "
					f"-> copied {r.rows_copied:,} -> written {r.rows_written:,}"
				)


if __name__ == "__main__":
	typer.run(main)

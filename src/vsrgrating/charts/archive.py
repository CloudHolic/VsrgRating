"""Reads chart files out of the osu! `osu_files` archive."""

import tarfile
from collections.abc import Iterator
from pathlib import Path


def iter_osu(path: Path, beatmap_ids: set[int]) -> Iterator[tuple[int, bytes]]:
	"""Each requested beatmap's `.osu` bytes, in archive order, from one streaming pass."""
	seen: set[int] = set()

	with tarfile.open(path, "r|bz2") as archive:
		for member in archive:
			name = Path(member.name).name
			stem = name.removesuffix(".osu")
			if not member.isfile() or stem == name or not stem.isdigit():
				continue

			beatmap_id = int(stem)
			if beatmap_id not in beatmap_ids:
				continue
			if beatmap_id in seen:
				raise ValueError(f"{name} appears twice in {path}")
			seen.add(beatmap_id)

			file = archive.extractfile(member)
			assert file is not None
			yield beatmap_id, file.read()

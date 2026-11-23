"""Fixed train/test assignment of beatmapsets, stable as the source database grows."""

import hashlib

TEST_SALT = "vsrgrating-test-2026-11-20"
TEST_MODULUS = 10_000
TEST_THRESHOLD = 1_000
TEST_FRACTION = TEST_THRESHOLD / TEST_MODULUS

# The 7K dan course mapsets, held out so the final evaluation can check the course order.
TEST_PINNED_MAPSETS: frozenset[int] = frozenset({450069, 450649, 451788, 895138, 930218, 1220647})


def is_test_mapset(beatmapset_id: int) -> bool:
	"""Whether a beatmapset belongs to the final test set, by a salted hash of its id."""
	if isinstance(beatmapset_id, bool) or not isinstance(beatmapset_id, int) or beatmapset_id <= 0:
		raise ValueError(f"beatmapset_id must be a positive int, got {beatmapset_id!r}")

	if beatmapset_id in TEST_PINNED_MAPSETS:
		return True

	digest = hashlib.blake2b(f"{TEST_SALT}:{beatmapset_id}".encode(), digest_size=8).digest()

	return int.from_bytes(digest, "big") % TEST_MODULUS < TEST_THRESHOLD

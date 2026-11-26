"""Which hand and finger plays each lane, decided by the number of lanes alone."""

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations

import numpy as np


class Hand(IntEnum):
	"""BOTH is a lane either hand plays, such as a centre thumb."""

	LEFT = 0
	RIGHT = 1
	BOTH = 2


class Finger(IntEnum):
	"""Anatomical order, thumb first."""

	THUMB = 0
	INDEX = 1
	MIDDLE = 2
	RING = 3
	PINKY = 4


@dataclass(frozen=True)
class LaneMap:
	"""One layout: the hand and finger of every lane, in lane order."""

	hand: tuple[Hand, ...]
	finger: tuple[Finger, ...]

	def mirrored(self) -> LaneMap:
		"""The layout read right to left, hands swapped."""
		swap = {Hand.LEFT: Hand.RIGHT, Hand.RIGHT: Hand.LEFT, Hand.BOTH: Hand.BOTH}
		return LaneMap(
			tuple(swap[hand] for hand in reversed(self.hand)), tuple(reversed(self.finger))
		)


# 4 to 8 lanes: the layouts players use. Each lane is hand (L, R, B) and finger (t, i, m, r, p).
_TABLES = {
	4: ("Lm Li Ri Rm",),
	5: ("Lr Lm Bi Rm Rr", "Lr Lm Li Ri Rm", "Lm Li Ri Rm Rr", "Lm Li Bt Ri Rm"),
	6: ("Lr Lm Li Ri Rm Rr",),
	7: ("Lr Lm Li Bt Ri Rm Rr",),
	8: (
		"Lp Lr Lm Li Ri Rm Rr Rp",
		"Lr Lm Li Lt Rt Ri Rm Rr",
		"Lr Lm Li Bt Ri Rm Rr Rp",
		"Lp Lr Lm Li Bt Ri Rm Rr",
	),
}
# Elsewhere fingers are taken in this order, both hands at once;
# past ten lanes each finger in the same order takes a second lane.
_PRIORITY = (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.THUMB, Finger.PINKY)
_HANDS = {"L": Hand.LEFT, "R": Hand.RIGHT, "B": Hand.BOTH}
_FINGERS = {
	"t": Finger.THUMB,
	"i": Finger.INDEX,
	"m": Finger.MIDDLE,
	"r": Finger.RING,
	"p": Finger.PINKY,
}


def layouts(lanes: int) -> tuple[LaneMap, ...]:
	"""Every layout of a chart with `lanes` lanes, special lanes included."""
	if lanes in _TABLES:
		return tuple(_parse(layout) for layout in _TABLES[lanes])
	if lanes == 1:
		return (LaneMap((Hand.BOTH,), (Finger.INDEX,)),)
	if lanes < 1 or lanes > 2 * 2 * len(_PRIORITY) or (lanes > 10 and lanes % 2):
		raise ValueError(f"no layout is defined for {lanes} lanes")

	if lanes <= 10:
		shared, odd = _PRIORITY[: lanes // 2], _PRIORITY[lanes // 2] if lanes % 2 else None
		counts = dict.fromkeys(shared, 1)
		if odd is None:
			return (_place(counts, counts),)
		return (_place(counts | {odd: 1}, counts), _place(counts, counts | {odd: 1}))

	counts = dict.fromkeys(_PRIORITY, 1) | dict.fromkeys(_PRIORITY[: (lanes - 10) // 2], 2)

	return (_place(counts, counts),)


def pairs(lane_map: LaneMap) -> np.ndarray:
	"""(hand, finger a, finger b) for every pair of fingers within one hand, a < b;
	a BOTH lane's finger belongs to either hand."""
	rows = []
	for hand in (Hand.LEFT, Hand.RIGHT):
		fingers = sorted(
			{
				f
				for h, f in zip(lane_map.hand, lane_map.finger, strict=True)
				if h in (hand, Hand.BOTH)
			}
		)
		rows += [(hand, a, b) for a, b in combinations(fingers, 2)]

	return np.array(rows, dtype=np.int8).reshape(-1, 3)


def _parse(layout: str) -> LaneMap:
	lanes = layout.split()
	return LaneMap(
		tuple(_HANDS[lane[0]] for lane in lanes), tuple(_FINGERS[lane[1]] for lane in lanes)
	)


def _place(left: dict[Finger, int], right: dict[Finger, int]) -> LaneMap:
	"""Lanes from the left edge."""
	order = sorted(left, reverse=True)
	left_lanes = [(Hand.LEFT, f) for f in order for _ in range(left[f])]
	right_lanes = [(Hand.RIGHT, f) for f in sorted(right) for _ in range(right[f])]
	hand, finger = zip(*(left_lanes + right_lanes), strict=True)
	return LaneMap(tuple(hand), tuple(finger))

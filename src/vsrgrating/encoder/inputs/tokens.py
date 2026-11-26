"""A chart as tokens: 1 token per integer millisecond holding any chart event, with each lane's
state there and the scroll speed around it."""

from collections import Counter
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
from urc_converter import Chart, Layout, Note, NoteType

_EVENTS = (NoteType.N, NoteType.LS, NoteType.LE, NoteType.M)


class LaneState(IntEnum):
	"""A lane at a token. MASKED is for pretraining only and never comes out of `tokenize`."""

	EMPTY = 0
	TAP = 1
	HOLD_START = 2
	HOLDING = 3
	HOLD_END = 4
	MINE = 5
	MASKED = 6


# Several events at one (ms, lane) show the first of these.
_PRIORITY = (
	(NoteType.LS, LaneState.HOLD_START),
	(NoteType.N, LaneState.TAP),
	(NoteType.LE, LaneState.HOLD_END),
	(NoteType.M, LaneState.MINE),
)


@dataclass(frozen=True)
class ChartTokens:
	"""Per-token arrays (T tokens, L lanes) and the chart-level facts the features need."""

	times: np.ndarray
	state: np.ndarray
	holds: np.ndarray
	speed: np.ndarray
	scroll: np.ndarray
	speed_min: np.ndarray
	speed_max: np.ndarray
	speed_changes: np.ndarray
	lanes: int
	special_lanes: tuple[int, ...]
	judgment: np.ndarray | None
	collisions: int


def tokenize(chart: Chart) -> ChartTokens:
	"""The chart's tokens."""
	if not chart.timing:
		raise ValueError("the chart has no timing points")
	if chart.timing[0].bpm <= 0:
		raise ValueError(f"the first BPM must be positive, got {chart.timing[0].bpm}")

	notes = [note for note in chart.notes if note.type in _EVENTS]
	if not notes:
		raise ValueError("the chart has no events")

	lanes = chart.layout.total_lanes
	times = np.unique(np.array([note.timestamp_ms for note in notes], dtype=np.int64))
	holds = _holds(notes)
	state, collisions = _states(notes, times, holds, lanes)
	speed, scroll, speed_min, speed_max, speed_changes = _speed(chart, times)

	judgment = None
	if chart.judgment is not None:
		judgment = np.column_stack([chart.judgment.windows, chart.judgment.rates]).astype(float)

	return ChartTokens(
		times=times,
		state=state,
		holds=holds,
		speed=speed,
		scroll=scroll,
		speed_min=speed_min,
		speed_max=speed_max,
		speed_changes=speed_changes,
		lanes=lanes,
		special_lanes=tuple(chart.layout.special_lanes or ()),
		judgment=judgment,
		collisions=collisions,
	)


def mirror(chart: Chart) -> Chart:
	"""A new chart mirrored as a game's MIRROR option does."""
	lanes = _mirrored_lanes(chart.layout)
	notes = [Note(note.timestamp_ms, lanes[note.lane], note.type) for note in chart.notes]

	return replace(chart, notes=notes)


def count_split_chords(original: Chart, scaled: Chart) -> int:
	"""How many same-ms groups of events in `original` no longer share one ms in `scaled`."""
	pairs: list[tuple[int, int]] = []
	for lane in range(original.layout.total_lanes):
		before = _lane_events(original, lane)
		after = _lane_events(scaled, lane)
		if len(before) != len(after):
			raise ValueError(
				f"lane {lane} has {len(before)} events before scaling, {len(after)} after"
			)

		pairs += [(a.timestamp_ms, b.timestamp_ms) for a, b in zip(before, after, strict=True)]

	scaled_times: dict[int, set[int]] = {}
	counts: dict[int, int] = {}

	for before_ms, after_ms in pairs:
		scaled_times.setdefault(before_ms, set()).add(after_ms)
		counts[before_ms] = counts.get(before_ms, 0) + 1

	return sum(1 for ms, found in scaled_times.items() if counts[ms] > 1 and len(found) > 1)


def _mirrored_lanes(layout: Layout) -> list[int]:
	special = sorted(layout.special_lanes or ())
	keys = [lane for lane in range(layout.total_lanes) if lane not in special]
	lanes = list(range(layout.total_lanes))

	for group in (keys, special):
		for lane, target in zip(group, reversed(group), strict=True):
			lanes[lane] = target

	return lanes


def _holds(notes: list[Note]) -> np.ndarray:
	"""(lane, LS ms, LE ms) in the parser's note order, which pairs each LS with its LE."""
	open_ls: dict[int, int] = {}
	holds: list[tuple[int, int, int]] = []

	for note in notes:
		if note.type is NoteType.LS:
			open_ls[note.lane] = note.timestamp_ms
		elif note.type is NoteType.LE:
			holds.append((note.lane, open_ls.pop(note.lane), note.timestamp_ms))

	return np.array(holds, dtype=np.int64).reshape(-1, 3)


def _states(
	notes: list[Note], times: np.ndarray, holds: np.ndarray, lanes: int
) -> tuple[np.ndarray, int]:
	state = np.full((times.size, lanes), LaneState.EMPTY, dtype=np.int8)
	for lane, start, end in holds:
		inside = slice(np.searchsorted(times, start, "right"), np.searchsorted(times, end, "left"))
		state[inside, lane] = LaneState.HOLDING

	events: dict[tuple[int, int], set[NoteType]] = {}
	for note in notes:
		events.setdefault((note.timestamp_ms, note.lane), set()).add(note.type)

	for (ms, lane), found in events.items():
		row = int(np.searchsorted(times, ms))
		state[row, lane] = next(value for kind, value in _PRIORITY if kind in found)

	per_cell = Counter((note.timestamp_ms, note.lane) for note in notes)
	return state, sum(1 for n in per_cell.values() if n > 1)


def _speed(
	chart: Chart, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Scroll speed and position at each token, and the speed's range and change count since the
	previous token."""
	points = chart.timing
	starts = np.array([point.timestamp_ms for point in points], dtype=np.int64)
	values = np.array([_sv(point.multiplier) * point.bpm for point in points]) / points[0].bpm

	reached = np.concatenate([[0.0], np.cumsum(values[:-1] * np.diff(starts))])
	active = np.clip(np.searchsorted(starts, times, "right") - 1, 0, values.size - 1)
	position = reached[active] + values[active] * (times - starts[active])
	changed = np.concatenate([[0], np.cumsum(values[1:] != values[:-1])])

	before = np.concatenate([active[:1], active[:-1]])
	low, high = values[active].copy(), values[active].copy()

	for i in np.flatnonzero(active > before):
		span = values[before[i] : active[i] + 1]
		low[i], high[i] = span.min(), span.max()

	return values[active], position - position[0], low, high, changed[active] - changed[before]


def _lane_events(chart: Chart, lane: int) -> list[Note]:
	# Within one ms an LE precedes the LS it makes room for, as the parser requires.
	order = {NoteType.LE: 0, NoteType.N: 1, NoteType.LS: 2, NoteType.M: 3}
	found = [note for note in chart.notes if note.lane == lane and note.type in _EVENTS]
	return sorted(found, key=lambda note: (note.timestamp_ms, order[note.type]))


def _sv(multiplier: float | None) -> float:
	return 1.0 if multiplier is None else multiplier

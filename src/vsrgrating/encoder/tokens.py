"""A chart as tokens: 1 token per integer millisecond holding any chart event, with each lane's
state there and the timing grid under it."""

from collections import Counter
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
from urc_converter import Chart, Layout, Note, NoteType

# Beat subdivisions, coarsest first; a token takes the first one within SNAP_TOLERANCE_MS.
SNAP_DENOMS = (1, 2, 3, 4, 6, 8, 12, 16)
SNAP_OTHER = len(SNAP_DENOMS)
SNAP_TOLERANCE_MS = 2.0

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
	bpm: np.ndarray
	sv: np.ndarray
	scroll: np.ndarray
	sv_min: np.ndarray
	sv_max: np.ndarray
	sv_changes: np.ndarray
	phase: np.ndarray
	snap: np.ndarray
	lanes: int
	special_lanes: tuple[int, ...]
	judgment: np.ndarray | None
	collisions: int


def tokenize(chart: Chart) -> ChartTokens:
	"""The chart's tokens."""
	if not chart.timing:
		raise ValueError("the chart has no timing points")

	notes = [note for note in chart.notes if note.type in _EVENTS]
	if not notes:
		raise ValueError("the chart has no events")

	lanes = chart.layout.total_lanes
	times = np.unique(np.array([note.timestamp_ms for note in notes], dtype=np.int64))
	holds = _holds(notes)
	state, collisions = _states(notes, times, holds, lanes)
	bpm, sv, phase, snap = _grid(chart, times)
	scroll, sv_min, sv_max, sv_changes = _scroll(chart, times)

	judgment = None
	if chart.judgment is not None:
		judgment = np.column_stack([chart.judgment.windows, chart.judgment.rates]).astype(float)

	return ChartTokens(
		times=times,
		state=state,
		holds=holds,
		bpm=bpm,
		sv=sv,
		scroll=scroll,
		sv_min=sv_min,
		sv_max=sv_max,
		sv_changes=sv_changes,
		phase=phase,
		snap=snap,
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


def _grid(chart: Chart, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""BPM, scroll multiplier, beat phase and snap of the timing point active at each token."""
	points = chart.timing
	starts = np.array([point.timestamp_ms for point in points], dtype=np.int64)
	anchors = np.empty(len(points), dtype=np.int64)
	previous: tuple[float, int, int, float] | None = None
	anchor = starts[0]

	for i, point in enumerate(points):
		current = (point.bpm, point.meter.beats, point.meter.note_value, _sv(point.multiplier))
		# An SV-only entry keeps the grid.
		if previous is None or current[:3] != previous[:3] or current == previous:
			anchor = point.timestamp_ms
		anchors[i] = anchor
		previous = current

	active = np.clip(np.searchsorted(starts, times, "right") - 1, 0, len(points) - 1)
	bpm = np.array([points[i].bpm for i in active], dtype=float)
	sv = np.array([_sv(points[i].multiplier) for i in active], dtype=float)
	beats = (times - anchors[active]) * bpm / 60000.0
	phase = np.mod(beats, 1.0)

	beat_ms = 60000.0 / bpm
	snap = np.full(times.size, SNAP_OTHER, dtype=np.int8)

	for index in reversed(range(len(SNAP_DENOMS))):
		d = SNAP_DENOMS[index]
		error_ms = np.abs(phase * d - np.round(phase * d)) / d * beat_ms
		snap[error_ms <= SNAP_TOLERANCE_MS] = index

	return bpm, sv, phase, snap


def _scroll(
	chart: Chart, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Scroll position at each token, and the multiplier's range and change count since the
	previous token."""
	starts = np.array([point.timestamp_ms for point in chart.timing], dtype=np.int64)
	values = np.array([_sv(point.multiplier) for point in chart.timing], dtype=float)
	last = np.append(starts[1:] != starts[:-1], True)
	starts, values = starts[last], values[last]

	reached = np.concatenate([[0.0], np.cumsum(values[:-1] * np.diff(starts))])
	active = np.clip(np.searchsorted(starts, times, "right") - 1, 0, values.size - 1)
	position = reached[active] + values[active] * (times - starts[active])
	changed = np.concatenate([[0], np.cumsum(values[1:] != values[:-1])])

	before = np.concatenate([active[:1], active[:-1]])
	sv_min, sv_max = values[active].copy(), values[active].copy()
	for i in np.flatnonzero(active > before):
		span = values[before[i] : active[i] + 1]
		sv_min[i], sv_max[i] = span.min(), span.max()

	return position - position[0], sv_min, sv_max, changed[active] - changed[before]


def _lane_events(chart: Chart, lane: int) -> list[Note]:
	# Within one ms an LE precedes the LS it makes room for, as the parser requires.
	order = {NoteType.LE: 0, NoteType.N: 1, NoteType.LS: 2, NoteType.M: 3}
	found = [note for note in chart.notes if note.lane == lane and note.type in _EVENTS]
	return sorted(found, key=lambda note: (note.timestamp_ms, order[note.type]))


def _sv(multiplier: float | None) -> float:
	return 1.0 if multiplier is None else multiplier

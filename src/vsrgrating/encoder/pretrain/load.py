"""The load a chart puts on the player from each moment on."""

from dataclasses import dataclass

import numpy as np
from urc_converter import Chart, NoteType

from ..inputs import ChartTokens, Finger, Hand, LaneMap, LaneState, pairs

LOAD_TAUS = (0.25, 1.0, 4.0, 16.0, 64.0)
FINGER_STREAMS = ("press", "release", "hold")
PAIR_STREAMS = ("outward", "inward", "together", "first_bound", "second_bound")
JUDGMENT_STREAMS = ("press_notes", "release_notes", "press_instants", "release_instants")
VISUAL_STREAMS = (
	"spatial_density",
	"speed_change",
	"speedup",
	"slowdown",
	"sign_flip",
	"stop",
	"reverse",
)
OCCUPANCY_STREAMS = ("hold", "stop", "reverse")

_HITS = (LaneState.TAP, LaneState.HOLD_START)
_VISIBLE = (NoteType.N, NoteType.LS, NoteType.LE, NoteType.M)
# |speed| is clamped here before the log, as the input features do.
_LOG_FLOOR, _LOG_CEIL = 1e-3, 1e3


@dataclass(frozen=True)
class LayoutLoad:
	"""Load under one hand and finger assignment."""

	finger: np.ndarray
	fingers: np.ndarray
	pair: np.ndarray


@dataclass(frozen=True)
class ChartLoad:
	"""Load that no assignment changes."""

	judgment: np.ndarray
	visual: np.ndarray


def layout_load(tokens: ChartTokens, lane_map: LaneMap, past: bool = False) -> LayoutLoad:
	"""The finger and pair load at every token."""
	seconds = tokens.times / 1000.0
	press = np.isin(tokens.state, _HITS)
	release = _release_counts(tokens)
	members = _fingers(lane_map)

	finger_press = np.stack([press[:, own].sum(axis=1) for own in members.values()], axis=1)
	finger_hit = finger_press > 0
	finger_release = np.stack([release[:, own].sum(axis=1) for own in members.values()], axis=1)

	held = np.stack([_held_between(tokens, own) for own in members.values()], axis=1)
	bound = np.stack([_bound_at(tokens, own) for own in members.values()], axis=1)

	index = pairs(lane_map)
	keys = list(members)
	last = _last_before(seconds, finger_hit)
	pair_events = []

	for hand, a, b in index:
		ka = keys.index(_pair_member(members, Hand(hand), Finger(a)))
		kb = keys.index(_pair_member(members, Hand(hand), Finger(b)))
		hit_a, hit_b = finger_hit[:, ka], finger_hit[:, kb]

		pair_events.append(
			np.stack(
				[
					_transition_weights(seconds, hit_a, hit_b, last[:, ka], last[:, kb]),
					_transition_weights(seconds, hit_b, hit_a, last[:, kb], last[:, ka]),
					_each_tau(hit_a & hit_b),
					_each_tau(bound[:, ka] & hit_b & ~hit_a),
					_each_tau(bound[:, kb] & hit_a & ~hit_b),
				],
				axis=1,
			)
		)

	# One pass over time for every event stream at once.
	k, p = len(keys), len(index)
	events = np.concatenate(
		[_each_tau(finger_press), _each_tau(finger_release), *pair_events], axis=1
	)
	rates = _discrete(seconds, events, past)
	held_load = _continuous(seconds, held.astype(float), past)

	finger = np.stack([rates[:, :k], rates[:, k : 2 * k], held_load], axis=2)
	pair = rates[:, 2 * k :].reshape(seconds.size, p, len(PAIR_STREAMS), len(LOAD_TAUS))
	fingers = np.array([[int(h), int(f)] for h, f in keys], dtype=np.int8).reshape(-1, 2)

	return LayoutLoad(finger=finger, fingers=fingers, pair=pair)


def chart_load(chart: Chart, tokens: ChartTokens, past: bool = False) -> ChartLoad:
	"""The judgment and on-screen load at every token, after it or with `past` up to it."""
	seconds = tokens.times / 1000.0
	press = np.isin(tokens.state, _HITS).sum(axis=1)
	release = _release_counts(tokens).sum(axis=1)
	counts = np.stack([press, release, press > 0, release > 0], axis=1)
	judgment = _discrete(seconds, _each_tau(counts), past)

	return ChartLoad(judgment=judgment, visual=_visual(chart, tokens, past))


def _visual(chart: Chart, tokens: ChartTokens, past: bool) -> np.ndarray:
	"""(T, 7, 5) on a timeline of the token times and the speed changes between the first
	and last token."""
	first, last = int(tokens.times[0]), int(tokens.times[-1])
	starts, speeds = _speeds(chart)
	inside = (starts >= first) & (starts <= last)
	grid = np.unique(np.concatenate([tokens.times, starts[inside]]))
	seconds = grid / 1000.0
	rows = np.searchsorted(grid, tokens.times)

	active = np.clip(np.searchsorted(starts, grid, "right") - 1, 0, speeds.size - 1)
	v = speeds[active]
	# Speed on each interval between grid points is the speed at its start.
	between = v[:-1]
	log_speed = np.log(np.clip(np.abs(between), _LOG_FLOOR, _LOG_CEIL))

	changes = np.zeros(grid.size)
	flips = np.zeros(grid.size)

	for start, before, after in zip(starts[1:], speeds[:-1], speeds[1:], strict=True):
		if first <= start <= last and after != before:
			row = int(np.searchsorted(grid, start))
			changes[row] += abs(_log_abs(after) - _log_abs(before))
			flips[row] += float(np.sign(after) != np.sign(before))

	objects = np.zeros(grid.size)
	visible = [note.timestamp_ms for note in chart.notes if note.type in _VISIBLE]
	np.add.at(objects, np.searchsorted(grid, visible), 1.0)
	scroll = np.concatenate([[0.0], np.cumsum(np.abs(between) * np.diff(seconds))])

	density = _discrete(scroll, _each_tau(objects[:, None]), past)
	timed = _discrete(seconds, _each_tau(np.stack([changes, flips], axis=1)), past)
	levels = np.stack(
		[np.maximum(log_speed, 0.0), np.maximum(-log_speed, 0.0), between == 0, between < 0],
		axis=1,
	)
	held = _continuous(seconds, levels.astype(float), past)
	# In VISUAL_STREAMS order.
	visual = np.concatenate([density, timed[:, :1], held[:, :2], timed[:, 1:], held[:, 2:]], axis=1)

	return visual[rows]


def _discrete(axis: np.ndarray, weights: np.ndarray, past: bool) -> np.ndarray:
	"""(N, E, 5): (1/tau) sum w exp(-|axis_i - axis_k| / tau) over the events strictly after each
	point, or with `past` at or before it."""
	taus = np.array(LOAD_TAUS)
	w = weights / taus
	out = np.zeros(w.shape)
	decay = np.exp(-np.diff(axis)[:, None, None] / taus)

	if past:
		out[0] = w[0]
		for k in range(1, axis.size):
			out[k] = decay[k - 1] * out[k - 1] + w[k]
	else:
		for k in range(axis.size - 2, -1, -1):
			out[k] = decay[k] * (out[k + 1] + w[k + 1])

	return out


def _continuous(seconds: np.ndarray, level: np.ndarray, past: bool) -> np.ndarray:
	"""(N, E, 5): (1/tau) integral g(u) exp(-|u - t_k| / tau) du from each point to the last one,
	or with `past` from the first to it."""
	taus = np.array(LOAD_TAUS)
	decay = np.exp(-np.diff(seconds)[:, None, None] / taus)
	g = level[..., None]
	out = np.zeros((seconds.size, level.shape[1], taus.size))

	if past:
		for k in range(1, seconds.size):
			out[k] = decay[k - 1] * out[k - 1] + (1 - decay[k - 1]) * g[k - 1]
	else:
		for k in range(seconds.size - 2, -1, -1):
			out[k] = decay[k] * out[k + 1] + (1 - decay[k]) * g[k]

	return out


def _transition_weights(
	seconds: np.ndarray,
	hit_from: np.ndarray,
	hit_to: np.ndarray,
	last_from: np.ndarray,
	last_to: np.ndarray,
) -> np.ndarray:
	"""(N, 5): hits of `to` whose previous hit of the pair was `from`'s, each weighted by
	exp(-gap / tau) so a move after a long rest counts little."""
	taus = np.array(LOAD_TAUS)
	moved = hit_to & ~hit_from & (last_to < last_from)
	gap = np.where(moved, seconds - np.where(np.isfinite(last_from), last_from, 0.0), np.inf)

	return np.where(moved[:, None], np.exp(-gap[:, None] / taus), 0.0)


def _each_tau(weights: np.ndarray) -> np.ndarray:
	"""(N, ..., 5): the same event weights for every memory length."""
	w = weights.astype(float)
	return np.broadcast_to(w[..., None], (*w.shape, len(LOAD_TAUS)))


def _release_counts(tokens: ChartTokens) -> np.ndarray:
	"""(T, L): hold ends at each token, from the holds since a release can share its cell."""
	counts = np.zeros(tokens.state.shape)

	if tokens.holds.size:
		rows = np.searchsorted(tokens.times, tokens.holds[:, 2])
		np.add.at(counts, (rows, tokens.holds[:, 0]), 1.0)

	return counts


def _held_between(tokens: ChartTokens, own: list[int]) -> np.ndarray:
	"""(T - 1,): whether one of `own` is held over each interval between tokens."""
	held = np.zeros(tokens.times.size - 1, dtype=bool)

	for lane, start, end in tokens.holds:
		if lane in own:
			a, b = np.searchsorted(tokens.times, [start, end])
			held[a:b] = True

	return held


def _bound_at(tokens: ChartTokens, own: list[int]) -> np.ndarray:
	"""(T,): whether one of `own` is held strictly across each token."""
	bound = np.zeros(tokens.times.size, dtype=bool)

	for lane, start, end in tokens.holds:
		if lane in own:
			bound |= (tokens.times > start) & (tokens.times < end)

	return bound


def _last_before(seconds: np.ndarray, hit: np.ndarray) -> np.ndarray:
	"""(T, K): the time of each column's last hit strictly before each token, -inf if none."""
	last = np.full(hit.shape, -np.inf)

	for k in range(hit.shape[1]):
		at = seconds[hit[:, k]]
		rows = np.searchsorted(at, seconds, "left") - 1
		last[:, k] = np.where(rows >= 0, at[np.maximum(rows, 0)] if at.size else 0.0, -np.inf)

	return last


def _speeds(chart: Chart) -> tuple[np.ndarray, np.ndarray]:
	"""Timing point starts and their speed relative to the first BPM."""
	by_ms: dict[int, float] = {}
	first_bpm = chart.timing[0].bpm

	for point in chart.timing:
		multiplier = 1.0 if point.multiplier is None else point.multiplier
		by_ms[point.timestamp_ms] = multiplier * point.bpm / first_bpm

	starts = np.array(sorted(by_ms), dtype=np.int64)
	return starts, np.array([by_ms[int(s)] for s in starts])


def _log_abs(speed: float) -> float:
	return float(np.log(np.clip(abs(speed), _LOG_FLOOR, _LOG_CEIL)))


def _fingers(lane_map: LaneMap) -> dict[tuple[Hand, Finger], list[int]]:
	members: dict[tuple[Hand, Finger], list[int]] = {}
	for lane, key in enumerate(zip(lane_map.hand, lane_map.finger, strict=True)):
		members.setdefault(key, []).append(lane)

	return members


def _pair_member(
	members: dict[tuple[Hand, Finger], list[int]], hand: Hand, finger: Finger
) -> tuple[Hand, Finger]:
	return (hand, finger) if (hand, finger) in members else (Hand.BOTH, finger)

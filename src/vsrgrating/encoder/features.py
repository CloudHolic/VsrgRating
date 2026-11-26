"""One layout's model input: the tokens of a chart with lane, finger, finger-pair and two-hand
features computed for that hand and finger assignment."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .layouts import Finger, Hand, LaneMap, pairs
from .tokens import ChartTokens, LaneState

TOKEN_FEATURES = (
	"log_dt",
	"position",
	"speed_sign",
	"speed_log",
	"scroll_dx",
	"speed_min_sign",
	"speed_min_log",
	"speed_max_sign",
	"speed_max_log",
	"speed_changes",
)
LANE_FEATURES = ("lane_dt", "hold_length", "hold_elapsed")
FINGER_FEATURES = ("finger_dt", "bound", "hits_while_bound")
PAIR_FEATURES = ("last_gap", "last_order", "other_bound")
META_FEATURES = (
	"lanes",
	"special_lanes",
	"duration",
	"hand_lanes_min",
	"hand_lanes_max",
	"both_lanes",
	"uses_thumb",
	"uses_pinky",
)
# |speed| and judgment windows are clamped here before the log.
LOG_FLOOR, LOG_CEIL = 1e-3, 1e3

_HITS = (LaneState.TAP, LaneState.HOLD_START)


class Known(IntEnum):
	"""Whether a derived value is there: NONE when it is undefined (no earlier hit)."""

	KNOWN = 0
	NONE = 1
	MASKED = 2


class Flag(IntEnum):
	"""A yes-or-no feature that pretraining can mask."""

	NO = 0
	YES = 1
	MASKED = 2


@dataclass(frozen=True)
class LayoutInput:
	"""T tokens, L lanes, P finger pairs; `judgment` is (J, 2): log window ms and score rate."""

	state: np.ndarray
	token: np.ndarray
	token_dt_known: np.ndarray
	lane: np.ndarray
	lane_known: np.ndarray
	finger: np.ndarray
	finger_known: np.ndarray
	pair: np.ndarray
	pair_known: np.ndarray
	pair_index: np.ndarray
	hands: np.ndarray
	lane_hand: np.ndarray
	lane_finger: np.ndarray
	lane_position: np.ndarray
	meta: np.ndarray
	judgment: np.ndarray | None


def layout_input(tokens: ChartTokens, lane_map: LaneMap) -> LayoutInput:
	"""The model input of `tokens` under one hand and finger assignment."""
	lanes = tokens.state.shape[1]
	if len(lane_map.hand) != lanes:
		raise ValueError(f"the layout has {len(lane_map.hand)} lanes, the chart {lanes}")

	times = tokens.times.astype(float)
	hit = np.isin(tokens.state, _HITS)
	holding = tokens.state == LaneState.HOLDING
	members = _fingers(lane_map)

	finger_hit = np.stack([hit[:, lanes_].any(axis=1) for lanes_ in members.values()], axis=1)
	finger_last = _last_hit(times, finger_hit, strict=True)
	finger_last_or_now = _last_hit(times, finger_hit, strict=False)

	lane, lane_known = _lane_features(tokens, times, hit)
	finger, finger_known = _finger_features(times, hit, holding, lane_map, members, finger_last)
	index = pairs(lane_map)
	pair, pair_known = _pair_features(holding, members, finger_hit, finger_last_or_now, index)

	token_dt_known = np.full(times.size, Known.KNOWN, dtype=np.int8)
	token_dt_known[0] = Known.NONE

	return LayoutInput(
		state=tokens.state,
		token=_token_features(tokens, times),
		token_dt_known=token_dt_known,
		lane=lane,
		lane_known=lane_known,
		finger=finger,
		finger_known=finger_known,
		pair=pair,
		pair_known=pair_known,
		pair_index=index,
		hands=_hands(hit, lane_map),
		lane_hand=np.array(lane_map.hand, dtype=np.int8),
		lane_finger=np.array(lane_map.finger, dtype=np.int8),
		lane_position=_lane_position(lanes, tokens.special_lanes),
		meta=_meta(tokens, times, lane_map),
		judgment=_judgment(tokens.judgment),
	)


def _token_features(tokens: ChartTokens, times: np.ndarray) -> np.ndarray:
	span = times[-1] - times[0]
	dt = np.diff(times, prepend=times[0])
	dx = np.diff(tokens.scroll, prepend=tokens.scroll[0])
	columns = [
		np.log1p(dt),
		(times - times[0]) / span if span > 0 else np.zeros_like(times),
		np.sign(tokens.speed),
		_log(np.abs(tokens.speed)),
		np.sign(dx) * np.log1p(np.abs(dx)),
		np.sign(tokens.speed_min),
		_log(np.abs(tokens.speed_min)),
		np.sign(tokens.speed_max),
		_log(np.abs(tokens.speed_max)),
		np.log1p(tokens.speed_changes),
	]

	return np.stack(columns, axis=1).astype(np.float32)


def _lane_features(
	tokens: ChartTokens, times: np.ndarray, hit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
	n, lanes = hit.shape
	values = np.zeros((n, lanes, len(LANE_FEATURES)), dtype=np.float32)
	known = np.full(values.shape, Known.NONE, dtype=np.int8)

	last = _last_hit(times, hit, strict=True)
	seen = np.isfinite(last)

	values[..., 0] = np.where(seen, np.log1p(times[:, None] - np.where(seen, last, 0.0)), 0.0)
	known[..., 0] = np.where(seen, Known.KNOWN, Known.NONE)

	for lane, start, end in tokens.holds:
		first, last_row = np.searchsorted(tokens.times, [start, end])
		length = np.log1p(end - start)

		for row in (first, last_row):
			values[row, lane, 1], known[row, lane, 1] = length, Known.KNOWN

		inside = slice(first + 1, last_row)
		values[inside, lane, 2] = np.log1p(times[inside] - start)
		known[inside, lane, 2] = Known.KNOWN

	return values, known


def _finger_features(
	times: np.ndarray,
	hit: np.ndarray,
	holding: np.ndarray,
	lane_map: LaneMap,
	members: dict[tuple[Hand, Finger], list[int]],
	finger_last: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	n, lanes = hit.shape
	per_finger = np.zeros((n, len(members), len(FINGER_FEATURES)), dtype=np.float32)
	known = np.full(per_finger.shape, Known.NONE, dtype=np.int8)

	seen = np.isfinite(finger_last)
	per_finger[..., 0] = np.where(
		seen, np.log1p(times[:, None] - np.where(seen, finger_last, 0.0)), 0.0
	)
	known[..., 0] = np.where(seen, Known.KNOWN, Known.NONE)

	for k, ((hand, _), own) in enumerate(members.items()):
		bound = holding[:, own].any(axis=1)
		per_finger[:, k, 1] = bound
		known[:, k, 1] = Known.KNOWN

		# A BOTH finger belongs to either hand, so every other lane counts as its hand's.
		hand_lanes = [
			lane
			for lane in range(lanes)
			if lane not in own and (hand is Hand.BOTH or lane_map.hand[lane] in (hand, Hand.BOTH))
		]
		so_far = np.cumsum(hit[:, hand_lanes].sum(axis=1))
		started = _bound_since(hit, own)
		rows = np.flatnonzero(bound)
		per_finger[rows, k, 2] = np.log1p(so_far[rows] - so_far[started[rows]])
		known[rows, k, 2] = Known.KNOWN

	column = np.array([list(members).index(_finger_key(lane_map, lane)) for lane in range(lanes)])
	return per_finger[:, column], known[:, column]


def _pair_features(
	holding: np.ndarray,
	members: dict[tuple[Hand, Finger], list[int]],
	finger_hit: np.ndarray,
	finger_last_or_now: np.ndarray,
	index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	n = finger_hit.shape[0]
	keys = list(members)
	values = np.zeros((n, len(index), len(PAIR_FEATURES)), dtype=np.float32)
	known = np.full(values.shape, Known.NONE, dtype=np.int8)
	bound = np.stack([holding[:, own].any(axis=1) for own in members.values()], axis=1)

	for p, (hand, a, b) in enumerate(index):
		ka = keys.index(_pair_member(members, Hand(hand), Finger(a)))
		kb = keys.index(_pair_member(members, Hand(hand), Finger(b)))
		ta, tb = finger_last_or_now[:, ka], finger_last_or_now[:, kb]
		both = np.isfinite(ta) & np.isfinite(tb)

		gap = np.zeros(n)
		gap[both] = ta[both] - tb[both]

		values[:, p, 0] = np.log1p(np.abs(gap))
		values[:, p, 1] = np.sign(gap)
		known[:, p, 0] = known[:, p, 1] = np.where(both, Known.KNOWN, Known.NONE)

		hit_a, hit_b = finger_hit[:, ka], finger_hit[:, kb]
		one = hit_a ^ hit_b

		values[:, p, 2] = np.where(one, np.where(hit_a, bound[:, kb], bound[:, ka]), 0.0)
		known[:, p, 2] = np.where(one, Known.KNOWN, Known.NONE)

	return values, known


def _hands(hit: np.ndarray, lane_map: LaneMap) -> np.ndarray:
	"""(mirror, same fingers): the left hand's lanes hit are the mirror of the right hand's, and
	each mirrored pair is played by the same finger. A BOTH lane is in either hand's set."""
	lanes = hit.shape[1]
	hand = np.array(lane_map.hand)
	finger = np.array(lane_map.finger)
	mirror = lanes - 1 - np.arange(lanes)

	left = hit & np.isin(hand, (Hand.LEFT, Hand.BOTH))
	right = hit & np.isin(hand, (Hand.RIGHT, Hand.BOTH))

	mirrored = left.any(axis=1) & (left[:, mirror] == right).all(axis=1)
	same = (finger == finger[mirror]) | ~left

	flags = np.zeros((hit.shape[0], 2), dtype=np.int8)
	flags[:, 0] = np.where(mirrored, Flag.YES, Flag.NO)
	flags[:, 1] = np.where(mirrored & same.all(axis=1), Flag.YES, Flag.NO)

	return flags


def _lane_position(lanes: int, special: tuple[int, ...]) -> np.ndarray:
	position = np.zeros((lanes, 3), dtype=np.float32)
	if lanes > 1:
		position[:, 0] = 2 * np.arange(lanes) / (lanes - 1) - 1
	if lanes % 2:
		position[lanes // 2, 1] = 1
	position[list(special), 2] = 1

	return position


def _meta(tokens: ChartTokens, times: np.ndarray, lane_map: LaneMap) -> np.ndarray:
	hand = lane_map.hand
	own = (hand.count(Hand.LEFT), hand.count(Hand.RIGHT))
	values = [
		tokens.lanes,
		len(tokens.special_lanes),
		np.log1p((times[-1] - times[0]) / 1000),
		min(own),
		max(own),
		hand.count(Hand.BOTH),
		Finger.THUMB in lane_map.finger,
		Finger.PINKY in lane_map.finger,
	]

	return np.array(values, dtype=np.float32)


def _judgment(judgment: np.ndarray | None) -> np.ndarray | None:
	if judgment is None:
		return None

	return np.column_stack([_log(judgment[:, 0]), judgment[:, 1]]).astype(np.float32)


def _fingers(lane_map: LaneMap) -> dict[tuple[Hand, Finger], list[int]]:
	"""Each finger (its hand, or BOTH for a shared lane) and the lanes it plays."""
	members: dict[tuple[Hand, Finger], list[int]] = {}
	for lane in range(len(lane_map.hand)):
		members.setdefault(_finger_key(lane_map, lane), []).append(lane)

	return members


def _finger_key(lane_map: LaneMap, lane: int) -> tuple[Hand, Finger]:
	return lane_map.hand[lane], lane_map.finger[lane]


def _pair_member(
	members: dict[tuple[Hand, Finger], list[int]], hand: Hand, finger: Finger
) -> tuple[Hand, Finger]:
	return (hand, finger) if (hand, finger) in members else (Hand.BOTH, finger)


def _last_hit(times: np.ndarray, hit: np.ndarray, strict: bool) -> np.ndarray:
	"""(T, K): the time of the last hit of each column before each token
	(or at it unless `strict`); inf when there is none."""
	last = np.full(hit.shape, np.inf)
	for k in range(hit.shape[1]):
		at = times[hit[:, k]]
		if at.size == 0:
			continue

		rows = np.searchsorted(at, times, "left" if strict else "right") - 1
		last[:, k] = np.where(rows >= 0, at[np.maximum(rows, 0)], np.inf)

	return last


def _bound_since(hit: np.ndarray, own: list[int]) -> np.ndarray:
	"""For each token, the row of the last press on these lanes."""
	rows = np.where(hit[:, own].any(axis=1), np.arange(hit.shape[0]), -1)
	return np.maximum.accumulate(np.maximum(rows, 0))


def _log(values: np.ndarray) -> np.ndarray:
	return np.log(np.clip(values, LOG_FLOOR, LOG_CEIL))

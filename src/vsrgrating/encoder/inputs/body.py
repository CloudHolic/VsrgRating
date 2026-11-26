"""What a chart asks of the body under one hand and finger assignment."""

from dataclasses import dataclass

import numpy as np

from .layouts import Finger, Hand, LaneMap, pairs
from .schema import FINGER_FEATURES, LANE_FEATURES, PAIR_FEATURES, Flag, Known
from .tokens import ChartTokens, LaneState

_HITS = (LaneState.TAP, LaneState.HOLD_START)


@dataclass(frozen=True)
class Body:
	"""The LayoutInput arrays this module computes, under the same names."""

	lane: np.ndarray
	lane_known: np.ndarray
	finger: np.ndarray
	finger_known: np.ndarray
	pair: np.ndarray
	pair_known: np.ndarray
	pair_index: np.ndarray
	hands: np.ndarray


def body(tokens: ChartTokens, lane_map: LaneMap) -> Body:
	"""The lane, finger, pair and two-hand features of `tokens` under `lane_map`."""
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

	return Body(
		lane=lane,
		lane_known=lane_known,
		finger=finger,
		finger_known=finger_known,
		pair=pair,
		pair_known=pair_known,
		pair_index=index,
		hands=_hands(hit, lane_map),
	)


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

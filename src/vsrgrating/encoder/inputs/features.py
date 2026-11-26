"""One layout's model input: the tokens of a chart with lane, finger, finger-pair and two-hand
features computed for that hand and finger assignment."""

import numpy as np

from .body import body
from .layouts import Finger, Hand, LaneMap
from .schema import Known, LayoutInput
from .tokens import ChartTokens

# |speed| and judgment windows are clamped here before the log.
LOG_FLOOR, LOG_CEIL = 1e-3, 1e3


def layout_input(tokens: ChartTokens, lane_map: LaneMap) -> LayoutInput:
	"""The model input of `tokens` under one hand and finger assignment."""
	lanes = tokens.state.shape[1]
	if len(lane_map.hand) != lanes:
		raise ValueError(f"the layout has {len(lane_map.hand)} lanes, the chart {lanes}")

	times = tokens.times.astype(float)
	parts = body(tokens, lane_map)

	token_dt_known = np.full(times.size, Known.KNOWN, dtype=np.int8)
	token_dt_known[0] = Known.NONE

	return LayoutInput(
		state=tokens.state,
		token=_token_features(tokens, times),
		token_dt_known=token_dt_known,
		lane=parts.lane,
		lane_known=parts.lane_known,
		finger=parts.finger,
		finger_known=parts.finger_known,
		pair=parts.pair,
		pair_known=parts.pair_known,
		pair_index=parts.pair_index,
		hands=parts.hands,
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


def _log(values: np.ndarray) -> np.ndarray:
	return np.log(np.clip(values, LOG_FLOOR, LOG_CEIL))

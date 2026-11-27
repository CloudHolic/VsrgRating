"""What pretraining hides from the encoder."""

from dataclasses import replace

import numpy as np

from .layouts import Finger, Hand
from .schema import Flag, Known, LayoutInput
from .tokens import ChartTokens, LaneState

MLM_RATE = 0.3
MLM_SPAN = (4, 16)

_HITS = (LaneState.TAP, LaneState.HOLD_START)
_POSITION = 1
_DURATION = 2
_LANE_DT, _HOLD_LENGTH, _HOLD_ELAPSED = 0, 1, 2
_FINGER_DT, _HITS_WHILE_BOUND = 0, 2
_PAIR_GAP, _PAIR_ORDER = 0, 1


def mlm_spans(n_tokens: int, rng: np.random.Generator) -> np.ndarray:
	"""(T,) bool: spans of 4 to 16 tokens, at least 1 token apart."""
	if n_tokens < 2:
		raise ValueError(f"masking needs at least 2 tokens, got {n_tokens}")

	target = max(1, round(MLM_RATE * n_tokens))
	masked = np.zeros(n_tokens, dtype=bool)

	while masked.sum() < target:
		length = min(int(rng.integers(MLM_SPAN[0], MLM_SPAN[1] + 1)), target - int(masked.sum()))
		starts = _free_starts(masked, length)
		if starts.size == 0:
			break

		start = int(rng.choice(starts))
		masked[start : start + length] = True

	return masked


def mlm_labels(tokens: ChartTokens, masked: np.ndarray) -> np.ndarray:
	"""(T, L) int8: each masked token's lane states, -1 elsewhere."""
	labels = np.full(tokens.state.shape, -1, dtype=np.int8)
	labels[masked] = tokens.state[masked]
	return labels


def mask_mlm(x: LayoutInput, tokens: ChartTokens, masked: np.ndarray) -> LayoutInput:
	"""`x` with the lane states of the masked tokens and every derived value that depends on them
	masked."""
	state = x.state.copy()
	lane, lane_known = x.lane.copy(), x.lane_known.copy()
	finger, finger_known = x.finger.copy(), x.finger_known.copy()
	pair, pair_known = x.pair.copy(), x.pair_known.copy()
	hands = x.hands.copy()

	n, lanes = state.shape
	visible_hit = np.isin(tokens.state, _HITS) & ~masked[:, None]
	holding = tokens.state == LaneState.HOLDING
	fingers = _finger_lanes(x)

	state[masked] = LaneState.MASKED
	for values, known in ((lane, lane_known), (finger, finger_known), (pair, pair_known)):
		_mask(values, known, masked)
	hands[masked] = Flag.MASKED

	for b in _span_ends(masked):
		# Gaps to the previous hit reach back into the span until a visible hit comes.
		for lane_ in range(lanes):
			end = _first(visible_hit[:, lane_], b)
			_mask(lane, lane_known, slice(b + 1, end + 1), lane_, _LANE_DT)

		for own in fingers:
			end = _first(visible_hit[:, own].any(axis=1), b)
			for lane_ in own:
				_mask(finger, finger_known, slice(b + 1, end + 1), lane_, _FINGER_DT)

		for p, (hand, a, c) in enumerate(x.pair_index):
			ka = _pair_finger(x, fingers, Hand(hand), Finger(a))
			kc = _pair_finger(x, fingers, Hand(hand), Finger(c))

			first_a = _first(visible_hit[:, fingers[ka]].any(axis=1), b)
			first_c = _first(visible_hit[:, fingers[kc]].any(axis=1), b)
			rows = slice(b + 1, max(first_a, first_c))

			for column in (_PAIR_GAP, _PAIR_ORDER):
				_mask(pair, pair_known, rows, p, column)

		# An LE + N right after the span shows as a plain tap; mask every tap's hold lengths there
		# so a hold ending in the collision cannot be told from no hold.
		if b + 1 < n:
			for lane_ in np.flatnonzero(tokens.state[b + 1] == LaneState.TAP).tolist():
				_mask(lane, lane_known, b + 1, lane_, _HOLD_LENGTH)

		# Hits while bound count from the press that bound the finger, which may be hidden.
		for own in fingers:
			bound = holding[:, own].any(axis=1)
			since = _bound_since(visible_hit, own)

			rows = np.flatnonzero(bound & (since <= b) & ~masked)
			rows = rows[rows > b]

			for lane_ in own:
				_mask(finger, finger_known, rows, lane_, _HITS_WHILE_BOUND)

	starts = {(int(lane_), int(row)) for lane_, row in _hold_rows(tokens)}
	for lane_, start, end in tokens.holds:
		first, last = (int(row) for row in np.searchsorted(tokens.times, [start, end]))
		for a, b in _spans(masked):
			if first < a <= last:
				_mask(lane, lane_known, first, lane_, _HOLD_LENGTH)
			if first <= b < last:
				_mask(lane, lane_known, slice(b + 1, last), lane_, _HOLD_ELAPSED)
				# A new hold starting where this one ends owns that row's hold length.
				if (int(lane_), last) not in starts:
					_mask(lane, lane_known, last, lane_, _HOLD_LENGTH)

	return replace(
		x,
		state=state,
		lane=lane,
		lane_known=lane_known,
		finger=finger,
		finger_known=finger_known,
		pair=pair,
		pair_known=pair_known,
		hands=hands,
	)


def pretraining_view(x: LayoutInput) -> LayoutInput:
	"""`x` without the token's place in the song and the song's length,
	which tell how much of it is left."""
	token, token_known = x.token.copy(), x.token_known.copy()
	meta, meta_known = x.meta.copy(), x.meta_known.copy()

	token[:, _POSITION], token_known[:, _POSITION] = 0.0, Known.MASKED
	meta[_DURATION], meta_known[_DURATION] = 0.0, Known.MASKED

	return replace(x, token=token, token_known=token_known, meta=meta, meta_known=meta_known)


def prefix_ends(n_tokens: int, segment: int, rng: np.random.Generator) -> np.ndarray:
	"""(S,) int64: the last prefix note of each encoder segment."""
	ends = np.zeros(_segments(n_tokens, segment).size, dtype=np.int64)

	for s, size in enumerate(_segments(n_tokens, segment)):
		if size >= 2 and rng.random() >= 0.5:
			ends[s] = rng.integers(0, size - 1)

	return ends


def prefix_weights(n_tokens: int, segment: int, ends: np.ndarray) -> np.ndarray:
	"""(T,) float64: 1 / P(token is supervised) for the tokens at or after their segment's prefix
	end."""
	weights = np.zeros(n_tokens)
	start = 0

	for size, end in zip(_segments(n_tokens, segment), ends, strict=True):
		k = np.arange(size)
		chance = np.ones(1) if size == 1 else np.minimum(1.0, 0.5 + 0.5 * (k + 1) / (size - 1))
		weights[start : start + size] = np.where(k >= end, 1.0 / chance, 0.0)
		start += size

	return weights


def mask_future_holds(
	x: LayoutInput, tokens: ChartTokens, ends: np.ndarray, segment: int
) -> LayoutInput:
	"""`x` with the hold length at every hold start masked when the hold ends after its segment's
	prefix."""
	lane, lane_known = x.lane.copy(), x.lane_known.copy()
	sizes = _segments(tokens.times.size, segment)
	bounds = np.concatenate([[0], np.cumsum(sizes)])
	cut = tokens.times[bounds[:-1] + ends]

	for lane_, start, end in tokens.holds:
		row = int(np.searchsorted(tokens.times, start))
		s = int(np.searchsorted(bounds, row, side="right")) - 1
		if end > cut[s]:
			_mask(lane, lane_known, row, lane_, _HOLD_LENGTH)

	return replace(x, lane=lane, lane_known=lane_known)


def _segments(n_tokens: int, segment: int) -> np.ndarray:
	chunk = segment - 1
	return np.array([min(chunk, n_tokens - s) for s in range(0, n_tokens, chunk)], dtype=np.int64)


def _free_starts(masked: np.ndarray, length: int) -> np.ndarray:
	"""Starts where `length` tokens fit with a free token on each side."""
	n = masked.size
	blocked = masked.copy()
	blocked[1:] |= masked[:-1]
	blocked[:-1] |= masked[1:]

	window = np.convolve(blocked, np.ones(length, dtype=int), mode="valid")
	starts = np.flatnonzero(window == 0)

	# Keep at least one unmasked token in the chart.
	return starts if masked.sum() + length < n else np.empty(0, dtype=np.int64)


def _spans(masked: np.ndarray) -> list[tuple[int, int]]:
	edges = np.diff(np.concatenate([[0], masked.astype(np.int8), [0]]))
	return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True))


def _span_ends(masked: np.ndarray) -> list[int]:
	return [int(b) for _, b in _spans(masked)]


def _first(visible: np.ndarray, b: int) -> int:
	"""The first row after `b` where `visible` is set, or the row count when none is."""
	after = np.flatnonzero(visible[b + 1 :])
	return b + 1 + int(after[0]) if after.size else visible.size


def _hold_rows(tokens: ChartTokens) -> list[tuple[int, int]]:
	"""(lane, row) of every hold start."""
	rows = np.searchsorted(tokens.times, tokens.holds[:, 1])
	return list(zip(tokens.holds[:, 0].tolist(), rows.tolist(), strict=True))


def _finger_lanes(x: LayoutInput) -> list[list[int]]:
	keys: dict[tuple[int, int], list[int]] = {}
	for lane_, key in enumerate(zip(x.lane_hand.tolist(), x.lane_finger.tolist(), strict=True)):
		keys.setdefault(key, []).append(lane_)

	return list(keys.values())


def _pair_finger(x: LayoutInput, fingers: list[list[int]], hand: Hand, finger: Finger) -> int:
	"""The finger group of a pair member."""
	keys = [(int(x.lane_hand[own[0]]), int(x.lane_finger[own[0]])) for own in fingers]

	for wanted in ((hand, finger), (Hand.BOTH, finger)):
		if wanted in keys:
			return keys.index(wanted)

	raise ValueError(f"no lane plays {hand.name} {finger.name}")


def _bound_since(hit: np.ndarray, own: list[int]) -> np.ndarray:
	"""For each token, the row of the last visible press on these lanes (-1 before any)."""
	rows = np.where(hit[:, own].any(axis=1), np.arange(hit.shape[0]), -1)
	return np.maximum.accumulate(rows)


def _mask(
	values: np.ndarray, known: np.ndarray, rows: int | slice | np.ndarray, *cell: int
) -> None:
	index: tuple[int | slice | np.ndarray, ...] = (rows, *cell)
	values[index] = 0.0
	known[index] = Known.MASKED

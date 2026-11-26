"""The model input of one layout pass."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

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


def token_known(x: LayoutInput) -> np.ndarray:
	"""The Known state of every token column: only the first (time since the previous token)
	can be undefined."""
	known = np.full(x.token.shape, Known.KNOWN, np.int8)
	known[:, 0] = x.token_dt_known
	return known

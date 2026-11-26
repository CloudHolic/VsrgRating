"""Summary statistics of a chart for the linear baseline: what simple counting can reach."""

import numpy as np

from .tokens import ChartTokens, LaneState

BASELINE_LANES = (4, 5, 6, 7, 8)
STATS_FEATURES = (
	*(f"lanes_{lanes}" for lanes in BASELINE_LANES),
	"log_hits",
	"log_duration",
	"hits_per_second",
	"peak_1s",
	"p90_5s",
	"hold_ratio",
	"chord_size",
	"gap_q10",
	"gap_q50",
	"gap_q90",
	"no_gap",
	"judgment_count",
	"judgment_min",
	"judgment_max",
	"no_judgment",
)

_HITS = (LaneState.TAP, LaneState.HOLD_START)


def chart_stats(tokens: ChartTokens) -> np.ndarray:
	"""The STATS_FEATURES of one chart; the baseline only covers BASELINE_LANES."""
	if tokens.lanes not in BASELINE_LANES:
		raise ValueError(f"the baseline covers {BASELINE_LANES} lanes, got {tokens.lanes}")

	hit = np.isin(tokens.state, _HITS)
	if not hit.any():
		raise ValueError("the chart has no hits")

	per_token = hit.sum(axis=1)
	hit_times = np.repeat(tokens.times, per_token).astype(float)
	duration = (tokens.times[-1] - tokens.times[0]) / 1000
	gaps = np.concatenate([np.diff(tokens.times[hit[:, lane]]) for lane in range(tokens.lanes)])
	presses = (tokens.state == LaneState.HOLD_START).sum()

	values = [
		*(float(tokens.lanes == lanes) for lanes in BASELINE_LANES),
		np.log1p(hit_times.size),
		np.log1p(duration),
		hit_times.size / max(duration, 1.0),
		_window_counts(hit_times, 1000.0).max(),
		np.quantile(_window_counts(hit_times, 5000.0), 0.9) / 5,
		presses / hit_times.size,
		per_token[per_token > 0].mean(),
		*(np.quantile(np.log1p(gaps), [0.1, 0.5, 0.9]) if gaps.size else (0.0, 0.0, 0.0)),
		float(gaps.size == 0),
		*_judgment(tokens.judgment),
	]

	return np.array(values, dtype=float)


def _window_counts(hit_times: np.ndarray, width: float) -> np.ndarray:
	"""Hits in [t, t + width) for every hit time t."""
	return np.searchsorted(hit_times, hit_times + width, "left") - np.arange(hit_times.size)


def _judgment(judgment: np.ndarray | None) -> tuple[float, float, float, float]:
	if judgment is None:
		return 0.0, 0.0, 0.0, 1.0
	windows = np.log(np.clip(judgment[:, 0], 1e-3, None))
	return float(len(judgment)), float(windows.min()), float(windows.max()), 0.0

"""The y each response variable is read on."""

ACC_GRID: tuple[float, ...] = (0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98)
SCORE_GRID: tuple[float, ...] = (0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.95, 0.98)
GRIDS: dict[str, tuple[float, ...]] = {"acc": ACC_GRID, "score": SCORE_GRID}

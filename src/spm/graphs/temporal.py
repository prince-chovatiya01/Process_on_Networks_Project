"""
Temporal event sequences and sliding-window slicing abstractions.
"""

from typing import Iterator, Tuple
import pandas as pd


def iter_time_windows(
    events_df: pd.DataFrame,
    window_size: int,
    step_size: int,
    time_col: str = "timestamp"
) -> Iterator[Tuple[int, int, pd.DataFrame]]:
    """
    Generate sliding temporal windows [t_start, t_start + window_size) across events.
    Yields (window_start, window_end, slice_df).
    """
    if events_df.empty:
        return
    t_min = events_df[time_col].min()
    t_max = events_df[time_col].max()

    curr_start = t_min
    while curr_start <= t_max:
        curr_end = curr_start + window_size
        mask = (events_df[time_col] >= curr_start) & (events_df[time_col] < curr_end)
        window_df = events_df[mask]
        yield curr_start, curr_end, window_df
        curr_start += step_size

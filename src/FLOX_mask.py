import numpy as np
import pandas as pd


def FLOX_mask(doy_day_frac, utc_datetime_str,
                    doy_range=None, date_range=None, time_ranges=None):

    """
    Build a boolean time mask for FLOX processing.
    doy_range : (start_doy, end_doy)
    date_range : (start_date, end_date)
        Example: ("2026-04-14", "2026-04-16")
    time_ranges : list of ("HH:MM", "HH:MM")
        Example: [("12:00","12:05"), ("14:00","16:30")]
    """

    mask = np.ones(len(doy_day_frac), dtype=bool)

    # -------------------------
    # DOY FILTER
    # -------------------------
    if doy_range is not None:
        start_doy, end_doy = doy_range
        mask &= (doy_day_frac >= start_doy) & (doy_day_frac <= end_doy)

    # -------------------------
    # DATE FILTER
    # -------------------------
    dt_time = pd.to_datetime(utc_datetime_str)

    if date_range is not None:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1])
        mask &= (dt_time >= start_dt) & (dt_time <= end_dt)

    # -------------------------
    # TIME RANGES (HH:MM intervals)
    # -------------------------
    if time_ranges is not None:

        minutes = dt_time.hour * 60 + dt_time.minute
        mask_time = np.zeros(len(minutes), dtype=bool)

        for start, end in time_ranges:
            h1, m1 = map(int, start.split(":"))
            h2, m2 = map(int, end.split(":"))

            start_min = h1 * 60 + m1
            end_min = h2 * 60 + m2

            mask_time |= (minutes >= start_min) & (minutes <= end_min)

        mask &= mask_time

    return mask

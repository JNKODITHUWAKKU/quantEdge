"""Calendar, timestamp and checksum utilities."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_EXCHANGE_CALENDAR = "XNYS"
DATE_INDEX_NAME = "Date"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_calendar(
    start: str, end: str, name: str = DEFAULT_EXCHANGE_CALENDAR
) -> pd.DatetimeIndex:
    import exchange_calendars as xcals

    calendar = xcals.get_calendar(name, start=start, end=end)
    # The requested boundary can itself be a weekend or holiday. Selecting from
    # the constructed schedule avoids treating that legitimate boundary as OOB.
    sessions = calendar.sessions
    # Strip timezone info if present so session indices match naive local dates from CSVs.
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    sessions = sessions[(sessions >= start_ts) & (sessions <= end_ts)]
    if len(sessions) == 0:
        raise ValueError("The requested interval contains no exchange sessions.")
    return pd.DatetimeIndex(sessions, name=DATE_INDEX_NAME)

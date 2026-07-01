"""Versioned ("vintaged") data store for MIST-Transformer.

Epidemiological signals are *revised over time*: the value reported for a given
``reference_date`` (the week/day an observation describes) changes as later
``issue_date`` revisions arrive. Training or back-testing a forecaster on the
*final* revised values leaks information that was not actually available at the
time a real forecast would have been made.

This module stores every (source, signal, location, reference_date, issue_date,
value) tuple as immutable Parquet and serves an *as-of* view through DuckDB:
:meth:`VersionedStore.get_vintage` only ever returns rows whose ``issue_date`` is
on or before the requested ``forecast_date``.

Storage layout
---------------
The store is an append-only directory of Parquet files (``data/store/*.parquet``).
Each :meth:`ingest` call writes a new file, so appends never rewrite existing
data. Reads glob all files via DuckDB's ``read_parquet`` with predicate pushdown
on ``issue_date``.
"""

from __future__ import annotations

import os
import uuid
from datetime import date
from typing import Optional, Union

import duckdb
import pandas as pd

# Canonical schema. Order matters for the Parquet/DuckDB column layout.
SCHEMA_COLUMNS = ["source", "signal", "location", "reference_date", "issue_date", "value"]
_DATE_COLUMNS = ["reference_date", "issue_date"]

DateLike = Union[str, date, pd.Timestamp]


class VersionedStore:
    """Append-only, vintage-aware store backed by Parquet + DuckDB.

    Parameters
    ----------
    store_dir:
        Directory holding the Parquet files. Created if missing.
    """

    def __init__(self, store_dir: Union[str, os.PathLike] = "data/store") -> None:
        self.store_dir = str(store_dir)
        os.makedirs(self.store_dir, exist_ok=True)
        # An in-memory DuckDB connection is enough: the data lives in Parquet and
        # is read on demand, so nothing is lost when the connection closes.
        self._conn = duckdb.connect(database=":memory:")

    # ------------------------------------------------------------------ helpers
    @property
    def _glob(self) -> str:
        """Glob pattern matching every Parquet file in the store."""
        return os.path.join(self.store_dir, "*.parquet")

    def _has_data(self) -> bool:
        try:
            return any(f.endswith(".parquet") for f in os.listdir(self.store_dir))
        except FileNotFoundError:
            return False

    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        """Validate columns and coerce date/value dtypes (returns a copy)."""
        missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns {missing}; "
                f"expected schema {SCHEMA_COLUMNS}"
            )
        out = df[SCHEMA_COLUMNS].copy()
        for col in _DATE_COLUMNS:
            out[col] = pd.to_datetime(out[col]).dt.normalize()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        for col in ("source", "signal", "location"):
            out[col] = out[col].astype("string")
        return out

    # ------------------------------------------------------------------ writes
    def ingest(self, df: pd.DataFrame) -> str:
        """Append observations to the store.

        ``df`` must contain the columns in :data:`SCHEMA_COLUMNS`. Dates are
        coerced to datetime. Returns the path of the Parquet file written.
        """
        clean = self._validate(df)
        if clean.empty:
            raise ValueError("Refusing to ingest an empty DataFrame")
        path = os.path.join(self.store_dir, f"obs-{uuid.uuid4().hex}.parquet")
        clean.to_parquet(path, index=False)
        return path

    # ------------------------------------------------------------------- reads
    def get_vintage(
        self,
        signal: str,
        location: str,
        forecast_date: DateLike,
        source: Optional[str] = None,
        latest_only: bool = True,
    ) -> pd.DataFrame:
        """Return the data that was *knowable* as of ``forecast_date``.

        Only rows with ``issue_date <= forecast_date`` are returned — this is the
        core anti-leakage guarantee.

        Parameters
        ----------
        signal, location:
            Filters on the corresponding columns.
        forecast_date:
            The "as-of" cutoff. No row issued after this date is ever returned.
        source:
            Optional additional filter on the ``source`` column.
        latest_only:
            When ``True`` (default), return the proper vintage view: for each
            ``reference_date`` keep only the most recent revision that was issued
            on or before ``forecast_date``. When ``False``, return every matching
            revision (useful for studying revision behaviour).

        Returns
        -------
        pandas.DataFrame
            Columns ``SCHEMA_COLUMNS`` sorted by ``reference_date`` (then
            ``issue_date``). Empty (with correct columns) if nothing matches.
        """
        cutoff = pd.to_datetime(forecast_date).normalize().date()

        if not self._has_data():
            return pd.DataFrame(columns=SCHEMA_COLUMNS)

        filters = ["signal = ?", "location = ?", "issue_date <= ?"]
        params: list = [signal, location, cutoff]
        if source is not None:
            filters.append("source = ?")
            params.append(source)
        where = " AND ".join(filters)

        if latest_only:
            # Keep the newest issue per reference_date that respects the cutoff.
            query = f"""
                SELECT source, signal, location, reference_date, issue_date, value
                FROM read_parquet('{self._glob}')
                WHERE {where}
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY source, signal, location, reference_date
                    ORDER BY issue_date DESC
                ) = 1
                ORDER BY reference_date, issue_date
            """
        else:
            query = f"""
                SELECT source, signal, location, reference_date, issue_date, value
                FROM read_parquet('{self._glob}')
                WHERE {where}
                ORDER BY reference_date, issue_date
            """

        result = self._conn.execute(query, params).fetchdf()
        for col in _DATE_COLUMNS:
            if col in result.columns:
                result[col] = pd.to_datetime(result[col])
        return result

    # ------------------------------------------------------- revision triangle
    def get_triangle(
        self,
        signal: str,
        location: str,
        forecast_date: DateLike,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return the full *revision triangle* knowable as of ``forecast_date``.

        Unlike :meth:`get_vintage` (which collapses each ``reference_date`` to a
        single as-of value), this returns **every** revision issued on or before
        ``forecast_date`` — the substrate the revision-aware forecaster needs to
        see *how* recent weeks are still settling. It is exactly
        ``get_vintage(..., latest_only=False)`` with an added integer
        ``issue_lag_weeks`` column ``= (issue_date - reference_date) / 7 days``
        (>= 0 for genuine backfill), and it inherits the same anti-leakage
        guarantee: no row with ``issue_date > forecast_date`` is ever returned.
        """
        tri = self.get_vintage(signal, location, forecast_date, source=source,
                               latest_only=False)
        if tri.empty:
            tri["issue_lag_weeks"] = pd.Series(dtype="int64")
            return tri
        lag = (tri["issue_date"] - tri["reference_date"]).dt.days // 7
        tri = tri.assign(issue_lag_weeks=lag.astype("int64"))
        return tri

    @staticmethod
    def to_lag_matrix(triangle: pd.DataFrame, max_lag: int, last_n: Optional[int] = None):
        """Pivot a triangle into a dense ``(reference_week x issue_lag)`` matrix.

        Rows are the (optionally last ``last_n``) reference weeks in ascending
        order; columns are integer issue lags ``0..max_lag``. Cell ``[w, k]`` is
        the value of week ``w`` as issued ``k`` weeks after its reference date, or
        ``NaN`` where that revision has not been issued yet as of the origin (the
        upper-right of the triangle — precisely the not-yet-known cells a
        revision-blind model wrongly treats as final). Returns
        ``(ref_weeks, matrix)`` where ``ref_weeks`` is a ``DatetimeIndex`` and
        ``matrix`` is an ``(R, max_lag+1)`` float array.
        """
        import numpy as np

        if triangle.empty:
            return pd.DatetimeIndex([]), np.empty((0, max_lag + 1), dtype=float)
        t = triangle[(triangle["issue_lag_weeks"] >= 0)
                     & (triangle["issue_lag_weeks"] <= max_lag)]
        wide = t.pivot_table(index="reference_date", columns="issue_lag_weeks",
                             values="value", aggfunc="last")
        wide = wide.reindex(columns=range(max_lag + 1))
        wide = wide.sort_index()
        if last_n is not None:
            wide = wide.iloc[-last_n:]
        return pd.DatetimeIndex(wide.index), wide.to_numpy(dtype=float)

    # ------------------------------------------------------------- maintenance
    def list_signals(self) -> pd.DataFrame:
        """Return distinct (source, signal, location) tuples currently stored."""
        if not self._has_data():
            return pd.DataFrame(columns=["source", "signal", "location"])
        return self._conn.execute(
            f"""SELECT DISTINCT source, signal, location
                FROM read_parquet('{self._glob}')
                ORDER BY source, signal, location"""
        ).fetchdf()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "VersionedStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

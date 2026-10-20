"""Offline, descriptive EDA for the project's original Alpaca parquet files.

No files are downloaded and no missing observations are reconstructed. Original
parquet files are never changed. Call ``run_raw_eda(project_root)`` from a notebook.
NYSE calendar behavior: https://pandas-market-calendars.readthedocs.io/en/latest/usage.html
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


NY_TIMEZONE = "America/New_York"
DEFAULT_START = "2020-01-01T00:00:00Z"
DEFAULT_END = "2026-10-16T00:00:00Z"
EXPECTED_PRICE_SYMBOLS = ("AAPL", "JNJ", "SPY")
COLORS = {"AAPL": "#2367a4", "JNJ": "#c87525", "SPY": "#687582"}
OWNED_FIGURES = ("monthly_news", "daily_news_distribution", "normalized_prices", "return_distributions")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def run_raw_eda(
    project_root,
    *,
    requested_start=DEFAULT_START,
    requested_end=DEFAULT_END,
    verified_news_coverage=None,
):
    """Inspect raw files, save a quality report and available figures, return a dict.

    Expected files are data/raw/news/{AAPL,JNJ}.parquet and data/raw/prices.parquet.
    Missing files produce an explicit waiting/partial status, not invented data.

    ``requested_start/end`` describe the collector's intended bounds only. The
    original code used timezone-naive dates; their interpretation as UTC here is
    an explicit, configurable assumption, not evidence of complete API coverage.

    By default, daily news summaries use only windows fully inside each ticker's
    observed timestamp range. Even those windows are NOT proven to have complete
    collection. If independently verified, pass e.g.
    ``verified_news_coverage={"AAPL": {"start": "...Z", "end": "...Z"}}``;
    these bounds are interpreted as [start, end), and must be timezone-aware.
    """
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        return {"status": "invalid_project_root", "project_root": str(root)}
    paths = {
        "AAPL": root / "data/raw/news/AAPL.parquet",
        "JNJ": root / "data/raw/news/JNJ.parquet",
        "prices": root / "data/raw/prices.parquet",
    }
    report = {
        "status": "waiting_for_data",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "files": {name: {"path": str(path), "exists": path.is_file()}
                  for name, path in paths.items()},
        "missing_files": [str(path) for path in paths.values() if not path.is_file()],
        "errors": [],
        "warnings": [],
        "unusable_inputs": [],
        "figures": [],
        "news": {},
        "prices": {},
        "assumptions": {
            "requested_start": requested_start,
            "requested_end": requested_end,
            "requested_bounds_are_verified": False,
            "original_naive_request_dates": "Assumed UTC; confirm with collector metadata.",
            "naive_raw_timestamps": "Assumed UTC when present; flagged in quality checks.",
            "news_window": "previous NYSE close < publication time <= current NYSE close",
            "news_deduplication": "None; suspected duplicates are flagged and retained.",
            "price_timestamp": "A daily-bar date label, never the market-close cutoff.",
            "scope": "Descriptive EDA only; no sentiment, target labels, or models.",
        },
    }
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    # A new run must not leave older plots looking like its current results.
    for name in OWNED_FIGURES:
        (figures_dir / f"eda_raw_{name}.png").unlink(missing_ok=True)

    def finish(status):
        report["status"] = status
        output = results_dir / "eda_raw_quality.json"
        report["quality_report"] = str(output)
        clean = _json_safe(report)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
        return clean

    if not any(path.is_file() for path in paths.values()):
        report["message"] = "Waiting for original parquet files; no raw-data EDA was run."
        return finish("waiting_for_data")

    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        report["errors"].append({"dependency": str(exc), "action": "Use the course Python environment."})
        return finish("dependency_missing")

    try:
        import pandas_market_calendars as mcal
    except ImportError:
        mcal = None
        report["errors"].append({
            "dependency": "pandas_market_calendars",
            "action": "Install this dependency to check trading dates and actual closing times.",
        })
    report["dependencies"] = {
        "pandas": pd.__version__, "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pandas_market_calendars": getattr(mcal, "__version__", None),
    }

    def read_parquet(name):
        path = paths[name]
        if not path.is_file():
            return None
        try:
            frame = pd.read_parquet(path)
            # Alpaca also supports an indexed (symbol, timestamp) price export.
            if name == "prices" and {"symbol", "timestamp"}.issubset(set(frame.index.names)):
                frame = frame.reset_index()
            report["files"][name]["read_status"] = "ok"
            report["files"][name]["bytes"] = path.stat().st_size
            with path.open("rb") as source:
                report["files"][name]["sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
            return frame
        except ImportError as exc:
            report["files"][name]["read_status"] = "dependency_missing"
            report["errors"].append({"file": str(path), "error": str(exc),
                                     "action": "Install pyarrow or fastparquet to read parquet files."})
        except Exception as exc:
            report["files"][name]["read_status"] = "read_failed"
            report["errors"].append({"file": str(path), "error": str(exc)})
        return None

    def base_quality(frame):
        return {
            "rows": len(frame), "columns": list(frame.columns),
            "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
            "null_counts": {name: int(count) for name, count in frame.isna().sum().items()},
            "null_fractions": {name: float(count / len(frame)) if len(frame) else None
                               for name, count in frame.isna().sum().items()},
        }

    def parse_timestamps(values):
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
        naive_count = 0
        for value in values[parsed.notna()]:
            try:
                naive_count += pd.Timestamp(value).tzinfo is None
            except (ValueError, TypeError):
                pass
        return parsed, {
            "missing_timestamp_rows": int(values.isna().sum()),
            "invalid_timestamp_rows": int((values.notna() & parsed.isna()).sum()),
            "naive_timestamp_rows_assumed_utc": int(naive_count),
            "observed_start_utc": parsed.min() if parsed.notna().any() else None,
            "observed_end_utc": parsed.max() if parsed.notna().any() else None,
        }

    def save_figure(fig, name):
        figures_dir.mkdir(parents=True, exist_ok=True)
        path = figures_dir / f"eda_raw_{name}.png"
        temporary = path.with_suffix(".png.tmp")
        fig.savefig(temporary, format="png", dpi=170, bbox_inches="tight")
        temporary.replace(path)
        plt.close(fig)
        report["figures"].append(str(path))

    def schedule_for(start, end):
        if mcal is None:
            return None
        # Padding supplies the previous close, and the next close after news.
        start_date = (start.tz_convert(NY_TIMEZONE) - pd.Timedelta(days=14)).date()
        end_date = (end.tz_convert(NY_TIMEZONE) + pd.Timedelta(days=14)).date()
        try:
            schedule = mcal.get_calendar("NYSE").schedule(start_date=start_date, end_date=end_date)
            schedule.index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
            return schedule
        except Exception as exc:
            report["errors"].append({"calendar_error": str(exc)})
            return None

    news_frames = {}
    for ticker in ("AAPL", "JNJ"):
        frame = read_parquet(ticker)
        if frame is None:
            continue
        quality = base_quality(frame)
        report["news"][ticker] = quality
        required = {"ticker", "created_at", "headline", "source", "symbols"}
        quality["missing_columns"] = sorted(required - set(frame.columns))
        if quality["missing_columns"]:
            report["unusable_inputs"].append({"input": ticker, "reason": "missing expected columns"})
        if not {"ticker", "created_at"}.issubset(frame.columns):
            report["warnings"].append(f"{ticker}: cannot analyze time coverage without ticker and created_at.")
            continue
        frame = frame.copy()
        frame["_time"], time_quality = parse_timestamps(frame["created_at"])
        quality.update(time_quality)
        if "updated_at" in frame:
            frame["_updated_time"], revision_quality = parse_timestamps(frame["updated_at"])
            quality["updated_timestamp_checks"] = revision_quality
        else:
            quality["headline_revision_check"] = "unavailable: updated_at was not saved"
            report["warnings"].append(
                f"{ticker}: publication time alone cannot establish that the retrieved headline existed at that time."
            )
        matches = frame["ticker"].astype("string").eq(ticker).fillna(False)
        quality["rows_with_unexpected_or_missing_ticker"] = int((~matches).sum())
        if "headline" in frame:
            headline = frame["headline"].astype("string").fillna("")
            quality["blank_headline_rows"] = int(headline.str.strip().eq("").sum())
            frame["_headline_key"] = headline.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
            keys = ["ticker", "_time", "_headline_key"]
            if "source" in frame:
                keys.append("source")
            quality["suspected_duplicate_rows"] = int(frame.duplicated(keys, keep=False).sum())
            quality["suspected_duplicate_excess_records"] = int(frame.duplicated(keys).sum())
            quality["duplicate_rule"] = "Same ticker, parsed publication time, normalized headline, and source if available."
        id_column = next((name for name in ("id", "article_id") if name in frame), None)
        quality["article_id_column"] = id_column
        quality["reliable_unique_article_count_available"] = bool(
            id_column and len(frame) and frame[id_column].notna().all()
        )
        if id_column:
            ids = frame.loc[frame[id_column].notna(), ["ticker", id_column]]
            quality["duplicate_id_rows_within_ticker"] = int(ids.duplicated(keep=False).sum())
            quality["unique_non_null_ticker_article_id_pairs"] = len(ids.drop_duplicates())
        if "source" in frame:
            quality["source_counts"] = frame["source"].astype("string").fillna("[missing]").value_counts().to_dict()
        if "symbols" in frame:
            def symbols_list(value):
                if isinstance(value, (list, tuple, np.ndarray)):
                    return [str(item) for item in value]
                return [value] if isinstance(value, str) else []
            symbols = frame["symbols"].map(symbols_list)
            quality["multi_symbol_news_rows"] = int(symbols.map(len).gt(1).sum())
            quality["ticker_absent_from_symbols_rows"] = int(symbols.map(lambda values: ticker not in values).sum())
        valid = frame.loc[matches & frame["_time"].notna()].copy()
        quality["records_used_for_time_summaries"] = len(valid)
        if valid.empty:
            report["unusable_inputs"].append({"input": ticker, "reason": "no usable timestamped news rows"})
            continue
        news_frames[ticker] = valid
        local = valid["_time"].dt.tz_convert(NY_TIMEZONE)
        quality["counts_by_local_year"] = local.dt.year.value_counts().sort_index().to_dict()

    if news_frames:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for ticker, frame in news_frames.items():
            local = frame["_time"].dt.tz_convert(NY_TIMEZONE).dt.tz_localize(None)
            monthly = local.dt.to_period("M").value_counts().sort_index()
            months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
            monthly = monthly.reindex(months, fill_value=0)
            report["news"][ticker]["counts_by_local_month"] = {str(key): int(value) for key, value in monthly.items()}
            report["news"][ticker]["edge_months_may_be_partial"] = [str(months[0]), str(months[-1])]
            ax.plot(monthly.index.to_timestamp(), monthly.values, label=ticker, color=COLORS[ticker])
        ax.set(title="Collected news coverage varies across stocks and months",
               xlabel="Publication month (New York time)", ylabel="Collected article–ticker records")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.text(0.01, 0.01, "Duplicates retained. Edge months may be partial; plotted counts do not prove API completeness.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        save_figure(fig, "monthly_news")

    daily_counts = {}
    for ticker, frame in news_frames.items():
        times = frame["_time"]
        coverage = (verified_news_coverage or {}).get(ticker)
        lower, upper = times.min(), times.max()
        basis = "observed_interval_not_proven_complete"
        if coverage is not None:
            try:
                lower = pd.Timestamp(coverage["start"])
                upper = pd.Timestamp(coverage["end"])
                if lower.tzinfo is None or upper.tzinfo is None or lower >= upper:
                    raise ValueError("Coverage bounds must be timezone-aware and start < end.")
                lower, upper = lower.tz_convert("UTC"), upper.tz_convert("UTC")
                basis = "externally_verified_coverage_asserted_by_caller"
            except (KeyError, TypeError, ValueError) as exc:
                report["warnings"].append(f"{ticker}: invalid verified coverage; using observed interval ({exc}).")
                lower, upper = times.min(), times.max()
        schedule = schedule_for(min(times.min(), lower), max(times.max(), upper))
        if schedule is None or schedule.empty:
            continue
        closes = pd.DatetimeIndex(schedule["market_close"])
        positions = closes.searchsorted(pd.DatetimeIndex(times), side="left")
        assigned = positions < len(closes)
        counts = pd.Series(0, index=schedule.index, dtype="int64")
        if assigned.any():
            assigned_dates = schedule.index.take(positions[assigned])
            observed_counts = pd.Series(assigned_dates).value_counts()
            counts.loc[observed_counts.index] = observed_counts.values
        if "_updated_time" in frame:
            eligible_closes = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
            eligible_closes.loc[assigned] = closes.take(positions[assigned]).to_numpy()
            known_revision = frame["_updated_time"].notna() & eligible_closes.notna()
            report["news"][ticker]["headline_revision_check"] = {
                "rows_with_known_update_time_and_assigned_close": int(known_revision.sum()),
                "updated_after_first_eligible_close_rows": int(
                    (known_revision & frame["_updated_time"].gt(eligible_closes)).sum()
                ),
                "interpretation": "Later updates may change headline wording. Historical headline availability remains unverified.",
            }
        previous_closes = schedule["market_close"].shift()
        # First/last partial windows are omitted, rather than counted as news-free.
        eligible = previous_closes.ge(lower) & schedule["market_close"].lt(upper)
        interior = counts.loc[eligible]
        quality = report["news"][ticker]
        quality["trading_window_summary"] = {
            "calendar": "NYSE", "basis": basis,
            "coverage_lower_utc": lower, "coverage_upper_utc_exclusive": upper,
            "news_records_assigned_to_a_close": int(assigned.sum()),
            "news_records_unassigned_after_schedule": int((~assigned).sum()),
            "eligible_interior_windows": len(interior),
            "records_in_eligible_interior_windows": int(interior.sum()),
            "records_excluded_from_interior_summaries": int(len(frame) - interior.sum()),
            "windows_with_zero_collected_records": int(interior.eq(0).sum()),
            "zero_record_window_fraction": float(interior.eq(0).mean()) if len(interior) else None,
            "daily_record_count_summary": interior.describe().to_dict(),
            "interpretation": "Zero collected records are not proof that no news existed.",
        }
        if len(interior):
            daily_counts[ticker] = interior

    if daily_counts:
        fig, axes = plt.subplots(1, len(daily_counts), figsize=(10, 4.6), squeeze=False)
        for ax, (ticker, counts) in zip(axes[0], daily_counts.items()):
            max_count = int(counts.max())
            edges = np.unique(np.r_[0, np.geomspace(1, max(max_count + 1, 2), 24)])
            ax.hist(counts, bins=edges, color=COLORS[ticker], edgecolor="white")
            ax.set_xscale("symlog", linthresh=1)
            ax.set(title=f"{ticker}: {counts.eq(0).mean():.1%} have zero collected records",
                   xlabel="Records per close-to-close window (symlog)", ylabel="NYSE trading windows")
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Daily news counts differ across stocks")
        fig.text(0.01, 0.01, "Only full windows inside stated intervals; observed-range coverage is not proven complete. Duplicates retained.", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        save_figure(fig, "daily_news_distribution")

    prices = read_parquet("prices")
    if prices is not None:
        quality = base_quality(prices)
        report["prices"] = quality
        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        quality["missing_columns"] = sorted(required - set(prices.columns))
        if quality["missing_columns"]:
            report["unusable_inputs"].append({"input": "prices", "reason": "missing expected columns"})
        if {"symbol", "timestamp", "close"}.issubset(prices.columns):
            prices = prices.copy()
            prices["_time"], time_quality = parse_timestamps(prices["timestamp"])
            quality.update(time_quality)
            prices["_date"] = prices["_time"].dt.tz_convert(NY_TIMEZONE).dt.tz_localize(None).dt.normalize()
            prices["symbol"] = prices["symbol"].astype("string")
            observed_symbols = sorted(prices["symbol"].dropna().unique().tolist())
            quality["expected_symbols"] = list(EXPECTED_PRICE_SYMBOLS)
            quality["missing_expected_symbols"] = sorted(set(EXPECTED_PRICE_SYMBOLS) - set(observed_symbols))
            quality["unexpected_symbols"] = sorted(set(observed_symbols) - set(EXPECTED_PRICE_SYMBOLS))
            if quality["missing_expected_symbols"]:
                report["unusable_inputs"].append({
                    "input": "prices", "reason": "missing expected symbols",
                    "symbols": quality["missing_expected_symbols"],
                })
            duplicate = prices.duplicated(["symbol", "_date"], keep=False)
            quality["duplicate_symbol_date_rows"] = int(duplicate.sum())
            quality["numeric_checks"] = {}
            for name in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
                if name not in prices:
                    continue
                numeric = pd.to_numeric(prices[name], errors="coerce")
                finite = pd.Series(np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)), index=prices.index)
                quality["numeric_checks"][name] = {
                    "non_numeric_non_null_rows": int((prices[name].notna() & numeric.isna()).sum()),
                    "non_finite_or_missing_rows": int((~finite).sum()),
                    "invalid_sign_rows": int((numeric.le(0) if name in ("open", "high", "low", "close", "vwap") else numeric.lt(0)).sum()),
                }
                prices[name] = numeric.where(finite)
            bad_ohlc = pd.Series(False, index=prices.index)
            if {"open", "high", "low", "close"}.issubset(prices.columns):
                bad_ohlc = (prices["low"].gt(prices[["open", "close"]].min(axis=1))
                            | prices["high"].lt(prices[["open", "close"]].max(axis=1))
                            | prices["low"].gt(prices["high"]))
            quality["inconsistent_ohlc_rows"] = int(bad_ohlc.sum())
            valid_mask = (prices["_time"].notna() & prices["symbol"].notna()
                          & prices["close"].gt(0) & ~duplicate & ~bad_ohlc)
            valid = prices.loc[valid_mask].copy()
            if valid.empty:
                report["unusable_inputs"].append({"input": "prices", "reason": "no usable unique positive closes"})
            quality["rows_failing_initial_price_plot_checks"] = int((~valid_mask).sum())
            quality["duplicate_policy"] = "All ambiguous duplicate symbol/date rows excluded from plots, never from raw files."
            quality["by_symbol"] = {}
            series = {}
            returns = {}
            for symbol, group in valid.groupby("symbol"):
                group = group.sort_values("_date")
                by_symbol = {"valid_unique_rows": len(group),
                             "observed_start_date": group["_date"].min(),
                             "observed_end_date": group["_date"].max()}
                quality["by_symbol"][str(symbol)] = by_symbol
                schedule = schedule_for(group["_time"].min(), group["_time"].max())
                if schedule is None:
                    by_symbol["calendar_check"] = "unavailable; return distribution skipped"
                    continue
                expected = schedule.index[(schedule.index >= group["_date"].min())
                                          & (schedule.index <= group["_date"].max())]
                dates = pd.DatetimeIndex(group["_date"])
                by_symbol["non_session_date_rows"] = int((~dates.isin(expected)).sum())
                missing = expected.difference(dates)
                by_symbol["missing_sessions_between_observed_endpoints"] = len(missing)
                by_symbol["missing_session_dates"] = [value.date().isoformat() for value in missing]
                # Check missing beginning/end dates as well as gaps between observations.
                try:
                    request_lower = pd.Timestamp(requested_start)
                    request_upper = pd.Timestamp(requested_end)
                    if request_lower.tzinfo is None or request_upper.tzinfo is None or request_lower >= request_upper:
                        raise ValueError("Requested bounds must be timezone-aware and start < end.")
                    request_schedule = schedule_for(request_lower, request_upper)
                    if request_schedule is not None:
                        bar_labels = request_schedule.index.tz_localize(NY_TIMEZONE).tz_convert("UTC")
                        requested_dates = request_schedule.index[(bar_labels >= request_lower) & (bar_labels < request_upper)]
                        request_missing = requested_dates.difference(dates)
                        by_symbol["requested_window_coverage"] = {
                            "basis": "configured request bounds; original naive dates assumed UTC",
                            "expected_sessions": len(requested_dates),
                            "missing_sessions": len(request_missing),
                            "missing_before_first_observation": int((request_missing < dates.min()).sum()),
                            "missing_after_last_observation": int((request_missing > dates.max()).sum()),
                            "missing_dates": [value.date().isoformat() for value in request_missing],
                        }
                except (TypeError, ValueError) as exc:
                    report["warnings"].append(f"Requested-window price coverage was not checked: {exc}")
                close = group.set_index("_date")["close"].reindex(expected)
                if close.notna().any():
                    series[str(symbol)] = close
                    # Reindexing before pct_change prevents a missing trading day
                    # from silently producing a multi-day return.
                    returns[str(symbol)] = close.pct_change(fill_method=None).dropna() * 100
                    by_symbol["daily_return_percent_summary"] = returns[str(symbol)].describe().to_dict()
            quality["rows_used_in_price_plots"] = sum(int(close.notna().sum()) for close in series.values())
            quality["rows_excluded_from_price_plots"] = int(len(prices) - quality["rows_used_in_price_plots"])
            if not series:
                report["unusable_inputs"].append({"input": "prices", "reason": "no valid calendar-aligned closing prices"})
            if series:
                common_dates = pd.concat(series, axis=1).dropna().index
                if len(common_dates):
                    common_start = common_dates[0]
                    quality["normalized_price_common_start"] = common_start
                    fig, ax = plt.subplots(figsize=(10, 4.8))
                    for symbol, close in series.items():
                        shown = close.loc[common_start:]
                        ax.plot(shown.index, shown / close.loc[common_start] * 100,
                                label=symbol, color=COLORS.get(symbol))
                    ax.set(title=f"Price paths indexed to 100 on {common_start.date()}",
                           xlabel="NYSE trading date", ylabel="Reported close / common-date close × 100")
                    ax.legend()
                    ax.grid(alpha=0.2)
                    fig.text(0.01, 0.01, "Collector requests Adjustment.ALL; parquet values alone cannot verify adjustment provenance. Gaps are not filled.", fontsize=8)
                    fig.tight_layout(rect=(0, 0.05, 1, 1))
                    save_figure(fig, "normalized_prices")
                else:
                    report["warnings"].append("Indexed-price comparison skipped: the symbols have no common valid trading date.")
            usable_returns = {name: values for name, values in returns.items() if len(values)}
            if usable_returns:
                fig, axes = plt.subplots(1, len(usable_returns), figsize=(11, 4.5), squeeze=False)
                for ax, (symbol, values) in zip(axes[0], usable_returns.items()):
                    ax.hist(values, bins=45, color=COLORS.get(symbol, "#4477AA"), edgecolor="white")
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set(title=symbol, xlabel="Close-to-close price change (%)", ylabel="Trading days")
                    ax.grid(axis="y", alpha=0.2)
                fig.suptitle("Daily price changes include both ordinary and extreme moves")
                fig.text(0.01, 0.01, "Only adjacent NYSE sessions with valid closes; no filling across missing sessions. All observed periods shown descriptively.", fontsize=8)
                fig.tight_layout(rect=(0, 0.05, 1, 0.93))
                save_figure(fig, "return_distributions")
        else:
            report["warnings"].append("Prices lack symbol, timestamp, or close; price figures were skipped.")

    loaded = [name for name, info in report["files"].items() if info.get("read_status") == "ok"]
    report["message"] = ("Descriptive EDA completed for available files. Missing inputs and coverage uncertainties remain explicit."
                         if loaded else "Raw files exist but could not be read; no raw-data figures were generated.")
    if not loaded:
        return finish("input_or_dependency_error")
    if not report["figures"]:
        report["message"] = "Files were read, but no usable observations were available for figures. See quality checks and dependency errors."
        return finish("no_usable_data" if not report["errors"] else "partial")
    if report["errors"] or report["missing_files"] or report["unusable_inputs"]:
        return finish("partial")
    return finish("completed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary = run_raw_eda(args.project_root)
    print(json.dumps({key: summary.get(key) for key in ("status", "message", "missing_files", "figures", "quality_report")}, indent=2))

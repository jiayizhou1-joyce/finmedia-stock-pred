"""Opt-in collection of supplemental Yahoo chart prices, with no credentials.

Yahoo's public chart endpoint is undocumented and may change or deny access.
This module neither bypasses access restrictions nor claims redistribution rights.
Importing it or running the CLI without --collect makes no network requests.
All outputs are separate from the team's original data/raw files.
"""

from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timezone
from email.utils import parsedate_to_datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import requests


VERSION = "1.0.0"
NY = ZoneInfo("America/New_York")
ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
OUTPUT_COLUMNS = ["symbol", "date", "timestamp", "open", "high", "low", "close", "volume", "adjusted_close"]
ALLOWED_SYMBOLS = {"AAPL", "JNJ", "SPY", "AMZN"}
HEADERS = {"User-Agent": "finmedia-stock-pred/1.0 (academic data collection)", "Accept": "application/json"}
LIMITATIONS = [
    "The public Yahoo chart endpoint is undocumented and may change or restrict access.",
    "Quote OHLC and volume are retained as supplied; no additional adjustment is applied by this collector.",
    "Yahoo-provided quote OHLC must not be assumed to be completely unadjusted, or equivalent to Alpaca Adjustment.ALL.",
    "adjusted_close is Yahoo's supplied series; its methodology and historical revisions are not independently verified.",
    "The original daily timestamp is a session observation label, not the market-close cutoff.",
    "Provider metadata can describe retrieval-time market conditions; it is not used as historical model input.",
    "A nonempty successful response does not prove complete historical coverage or confer redistribution rights.",
]


class PublicPriceError(RuntimeError):
    """An explicit transport, access, validation, or cache failure."""


@dataclass(frozen=True)
class PublicPriceConfig:
    symbols: tuple = ("AAPL", "JNJ", "SPY", "AMZN")
    start: str = "2020-01-01"
    end: str = "2026-10-16"
    max_attempts: int = 3
    initial_retry_wait: float = 1.0
    max_retry_wait: float = 30.0

    def __post_init__(self):
        symbols = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in self.symbols))
        if not symbols or not set(symbols).issubset(ALLOWED_SYMBOLS):
            raise ValueError("Choose symbols from AAPL, JNJ, SPY, and AMZN.")
        start, end = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if start >= end:
            raise ValueError("The start date must precede the exclusive end date.")
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5.")
        if not 0 <= self.initial_retry_wait <= self.max_retry_wait <= 30:
            raise ValueError("Retry waits must satisfy 0 <= initial <= maximum <= 30 seconds.")
        object.__setattr__(self, "symbols", symbols)

    def parameters(self):
        # These are New York calendar-date boundaries, not UTC calendar dates.
        period1 = int(datetime.combine(date.fromisoformat(self.start), datetime_time(), NY).timestamp())
        period2 = int(datetime.combine(date.fromisoformat(self.end), datetime_time(), NY).timestamp())
        return {"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _versions():
    result = {"pandas": pd.__version__, "requests": requests.__version__}
    for package in ("pyarrow", "fastparquet"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _identity(symbol, config):
    return {"collector_version": VERSION, "symbol": symbol, "endpoint": ENDPOINT.format(symbol=symbol),
            "parameters": config.parameters(), "start_session_date_inclusive": config.start,
            "end_session_date_exclusive": config.end, "session_timezone": "America/New_York",
            "versions": _versions()}


def _paths(base, symbol):
    return {"raw": base / "raw" / f"{symbol}.json",
            "parquet": base / "cache" / f"{symbol}.parquet",
            "manifest": base / "cache" / f"{symbol}.manifest.json"}


def _schema(frame):
    return [{"name": name, "dtype": str(dtype)} for name, dtype in frame.dtypes.items()]


def _read_cache(base, symbol, config):
    paths = _paths(base, symbol)
    if not all(path.is_file() for path in paths.values()):
        return None, None, "file_or_manifest_missing"
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("identity") != _identity(symbol, config):
            return None, None, "parameters_or_versions_changed"
        if (manifest.get("raw_json_sha256") != _sha256(paths["raw"])
                or manifest.get("parquet_sha256") != _sha256(paths["parquet"])):
            return None, None, "file_hash_mismatch"
        frame = pd.read_parquet(paths["parquet"])
        if (list(frame.columns) != OUTPUT_COLUMNS or manifest.get("schema") != _schema(frame)
                or manifest.get("row_count") != len(frame) or frame.empty):
            return None, None, "schema_or_row_count_mismatch"
        if (not frame["symbol"].eq(symbol).all() or frame["date"].duplicated().any()
                or not frame["date"].ge(pd.Timestamp(config.start)).all()
                or not frame["date"].lt(pd.Timestamp(config.end)).all()
                or frame["adjusted_close"].isna().any() or not frame["adjusted_close"].gt(0).all()):
            return None, None, "invalid_cached_observations"
        return frame, manifest, "validated_cache"
    except Exception:
        return None, None, "unreadable_cache"


def _retry_after(response, now):
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except (ValueError, TypeError):
        try:
            when = parsedate_to_datetime(header)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - now).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


def _archive_response(base, symbol, config, response):
    now = datetime.now(timezone.utc)
    body = response.content
    if not isinstance(body, bytes):
        raise PublicPriceError("The HTTP client did not provide the original response bytes.")
    token = f"{now:%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    path = base / "raw/responses" / symbol / f"{token}.body"
    _atomic_bytes(path, body)
    metadata = {"retrieved_at_utc": now.isoformat(), "endpoint": ENDPOINT.format(symbol=symbol),
                "parameters": config.parameters(), "http_status": int(response.status_code),
                "content_type": response.headers.get("Content-Type"),
                "retry_after": response.headers.get("Retry-After"), "response_path": str(path),
                "response_sha256": _sha256(path), "interpretation": "Exact HTTP body; not a claim of valid price data."}
    _atomic_json(path.with_suffix(".response.json"), metadata)
    return body, metadata, now


def _fetch(base, symbol, config, session, sleep):
    url = ENDPOINT.format(symbol=symbol)
    for attempt in range(config.max_attempts):
        try:
            response = session.get(url, params=config.parameters(), headers=HEADERS.copy(),
                                   timeout=(10, 30), allow_redirects=False)
        except (requests.Timeout, requests.ConnectionError) as error:
            # TLS/certificate failures are not transient connection failures.
            if isinstance(error, requests.exceptions.SSLError) or attempt + 1 == config.max_attempts:
                raise PublicPriceError(f"{symbol}: transport failed ({type(error).__name__}); collection stopped.") from None
            sleep(min(config.initial_retry_wait * 2 ** attempt, config.max_retry_wait))
            continue
        body, metadata, now = _archive_response(base, symbol, config, response)
        status = int(response.status_code)
        if status == 200:
            return body, metadata
        if status not in {429, 500, 502, 503, 504}:
            raise PublicPriceError(f"{symbol}: HTTP {status}; no access-gate, endpoint, or identity workaround attempted.")
        wait = _retry_after(response, now)
        if wait is not None and (not math.isfinite(wait) or wait > config.max_retry_wait):
            raise PublicPriceError(f"{symbol}: server requested a longer pause than the allowed retry window; stopped without retrying early.")
        if attempt + 1 == config.max_attempts:
            raise PublicPriceError(f"{symbol}: HTTP {status}; bounded retry attempts exhausted.")
        sleep(wait if wait is not None else min(config.initial_retry_wait * 2 ** attempt, config.max_retry_wait))


def _reject_json_constant(value):
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def _numeric_array(values, name, expected_length, *, allow_null=True, integer=False):
    if not isinstance(values, list) or len(values) != expected_length:
        raise PublicPriceError(f"Malformed {name} array: its length must match timestamp exactly.")
    for value in values:
        if value is None and allow_null:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (integer and value != int(value))):
            raise PublicPriceError(f"Malformed {name}: expected finite numeric values" + (" or null." if allow_null else "."))
    return pd.Series(values, dtype="Int64" if integer else "float64")


def normalize_response(body, symbol, config):
    """Validate one response and retain quote values, without filling or adjustment."""
    try:
        payload = json.loads(body, parse_constant=_reject_json_constant)
    except (ValueError, TypeError, UnicodeDecodeError):
        raise PublicPriceError(f"{symbol}: response is not valid JSON; exact body was archived.") from None
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict) or chart.get("error"):
        raise PublicPriceError(f"{symbol}: Yahoo returned a chart error or unexpected response schema.")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise PublicPriceError(f"{symbol}: response must contain one nonempty chart result.")
    result = results[0]
    meta = result.get("meta")
    if not isinstance(meta, dict) or str(meta.get("symbol", "")).upper() != symbol:
        raise PublicPriceError(f"{symbol}: provider symbol metadata is missing or mismatched.")
    provider_timezone = meta.get("exchangeTimezoneName")
    if provider_timezone not in {"America/New_York", "US/Eastern"}:
        raise PublicPriceError(f"{symbol}: expected explicit New York exchange timezone metadata.")
    timestamps = result.get("timestamp")
    if not isinstance(timestamps, list) or not timestamps:
        raise PublicPriceError(f"{symbol}: empty or missing timestamps.")
    epoch = _numeric_array(timestamps, "timestamp", len(timestamps), allow_null=False, integer=True)
    parsed = pd.to_datetime(epoch, unit="s", utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    if parsed.isna().any():
        raise PublicPriceError(f"{symbol}: invalid epoch timestamps.")
    indicators = result.get("indicators")
    if not isinstance(indicators, dict):
        raise PublicPriceError(f"{symbol}: missing price indicators.")
    quotes, adjusted = indicators.get("quote"), indicators.get("adjclose")
    if not isinstance(quotes, list) or len(quotes) != 1 or not isinstance(quotes[0], dict):
        raise PublicPriceError(f"{symbol}: missing daily quote data.")
    if not isinstance(adjusted, list) or len(adjusted) != 1 or not isinstance(adjusted[0], dict):
        raise PublicPriceError(f"{symbol}: adjusted close is required; no replacement with close is permitted.")
    frame = pd.DataFrame({"symbol": pd.Series([symbol] * len(timestamps), dtype="string"), "timestamp": parsed})
    frame["date"] = (parsed.dt.tz_convert(provider_timezone).dt.tz_localize(None)
                     .dt.normalize().astype("datetime64[ns]"))
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = _numeric_array(quotes[0].get(name), name, len(timestamps), integer=name == "volume")
    frame["adjusted_close"] = _numeric_array(adjusted[0].get("adjclose"), "adjusted_close", len(timestamps), allow_null=False)
    if not frame["adjusted_close"].gt(0).all():
        raise PublicPriceError(f"{symbol}: adjusted_close must be present, finite, and positive for every returned row.")
    eligible = frame["date"].ge(pd.Timestamp(config.start)) & frame["date"].lt(pd.Timestamp(config.end))
    outside = int((~eligible).sum())
    frame = frame.loc[eligible, OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise PublicPriceError(f"{symbol}: no observations remain in the requested half-open session-date interval.")
    if frame["date"].duplicated().any():
        raise PublicPriceError(f"{symbol}: duplicate daily session dates; no arbitrary deduplication performed.")
    details = {"returned_rows": len(timestamps), "outside_session_date_interval_rows": outside,
               "observed_first_session_date": frame["date"].min().date().isoformat(),
               "observed_last_session_date": frame["date"].max().date().isoformat(),
               "null_counts": {name: int(count) for name, count in frame.isna().sum().items()},
               "negative_volume_rows": int(frame["volume"].lt(0).sum()),
               "nonpositive_quote_price_rows": int(frame[["open", "high", "low", "close"]].le(0).any(axis=1).sum()),
               "provider_metadata": meta}
    return frame, details


def _write_symbol(base, symbol, config, body, frame, response_metadata, details):
    paths = _paths(base, symbol)
    paths["parquet"].parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=paths["parquet"].parent, suffix=".parquet", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.to_parquet(temporary, index=False)
        persisted = pd.read_parquet(temporary)
        # Publish the manifest last. An interrupted pair cannot validate later.
        _atomic_bytes(paths["raw"], body)
        os.replace(temporary, paths["parquet"])
        manifest = {"identity": _identity(symbol, config), "retrieved_at_utc": response_metadata["retrieved_at_utc"],
                    "raw_json_path": str(paths["raw"]), "raw_json_sha256": _sha256(paths["raw"]),
                    "parquet_path": str(paths["parquet"]), "parquet_sha256": _sha256(paths["parquet"]),
                    "schema": _schema(persisted), "row_count": len(frame), "response": response_metadata,
                    "adjustment_basis": "Yahoo-provided quote OHLC/volume and separate Yahoo-provided adjusted_close; no collector-side price adjustment.",
                    "limitations": LIMITATIONS, **details}
        _atomic_json(paths["manifest"], manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def inspect_cache(project_root, config=None):
    """Inspect local caches only. This function never constructs an HTTP session."""
    config = config or PublicPriceConfig()
    base = Path(project_root).expanduser().resolve() / "data/supplemental/yahoo_prices"
    symbols = {}
    for symbol in config.symbols:
        frame, manifest, reason = _read_cache(base, symbol, config)
        symbols[symbol] = {"cache_status": reason, "rows": len(frame) if frame is not None else None,
                           "retrieved_at_utc": manifest.get("retrieved_at_utc") if manifest else None}
    all_valid = all(value["cache_status"] == "validated_cache" for value in symbols.values())
    dataset = base / "prices.parquet"
    dataset_manifest = base / "manifest.json"
    dataset_valid = False
    if all_valid and dataset.is_file() and dataset_manifest.is_file():
        try:
            manifest = json.loads(dataset_manifest.read_text())
            components = {symbol: _sha256(_paths(base, symbol)["manifest"]) for symbol in config.symbols}
            dataset_valid = (manifest.get("component_manifest_hashes") == components
                             and manifest.get("parquet_sha256") == _sha256(dataset))
        except (OSError, ValueError, TypeError):
            pass
    return {"status": "ready" if dataset_valid else "ready_for_assembly" if all_valid else "missing_or_invalid_cache",
            "network_calls_executed": False, "symbols": symbols, "dataset_path": str(dataset),
            "start_session_date_inclusive": config.start, "end_session_date_exclusive": config.end,
            "parameters": config.parameters(), "limitations": LIMITATIONS}


def run_collection(project_root, config=None, *, collect=False, refresh=False, session=None, sleep=time.sleep):
    """Inspect by default; collect=True explicitly permits new public HTTP requests."""
    config = config or PublicPriceConfig()
    if not collect:
        return inspect_cache(project_root, config)
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory.")
    if not (importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")):
        raise PublicPriceError("Install a parquet engine before enabling any collection requests.")
    base = root / "data/supplemental/yahoo_prices"
    report = {"status": "collecting", "symbols": {}, "network_calls_executed": False,
              "start_session_date_inclusive": config.start, "end_session_date_exclusive": config.end,
              "limitations": LIMITATIONS}
    own_session = session is None
    frames = []
    try:
        for symbol in config.symbols:
            frame, manifest, reason = _read_cache(base, symbol, config)
            if refresh or frame is None:
                if session is None:
                    session = requests.Session()
                report["network_calls_executed"] = True
                body, response_metadata = _fetch(base, symbol, config, session, sleep)
                frame, details = normalize_response(body, symbol, config)
                manifest = _write_symbol(base, symbol, config, body, frame, response_metadata, details)
                action = "fetched"
            else:
                action = "reused"
            frames.append(frame)
            report["symbols"][symbol] = {"action": action, "cache_reason": reason, "rows": len(frame),
                                         "retrieved_at_utc": manifest["retrieved_at_utc"]}
        combined = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
        dataset = base / "prices.parquet"
        handle = tempfile.NamedTemporaryFile(dir=base, suffix=".parquet", delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            combined.to_parquet(temporary, index=False)
            persisted = pd.read_parquet(temporary)
            os.replace(temporary, dataset)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {"collector_version": VERSION, "provider": "Yahoo public chart",
                    "assembled_at_utc": datetime.now(timezone.utc).isoformat(), "symbols": list(config.symbols),
                    "start_session_date_inclusive": config.start, "end_session_date_exclusive": config.end,
                    "row_count": len(combined), "schema": _schema(persisted), "parquet_sha256": _sha256(dataset),
                    "component_manifest_hashes": {symbol: _sha256(_paths(base, symbol)["manifest"]) for symbol in config.symbols},
                    "adjustment_basis": "Preserved Yahoo quote OHLC/volume; separate provider adjusted_close. No collector-side adjustment.",
                    "limitations": LIMITATIONS}
        _atomic_json(base / "manifest.json", manifest)
        report.update(status="collected_archive_completeness_unverified", rows=len(combined), dataset_path=str(dataset))
        _atomic_json(base / "latest_run.json", report)
        return report
    except PublicPriceError as error:
        report.update(status="failed_completed_symbol_caches_preserved", error=str(error))
        _atomic_json(base / "latest_run.json", report)
        raise
    finally:
        if own_session and session is not None:
            session.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--collect", action="store_true", help="Explicitly permit public HTTP requests for missing/invalid caches.")
    parser.add_argument("--refresh", action="store_true", help="With --collect, request fresh responses even if caches validate.")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "JNJ", "SPY", "AMZN"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-10-16")
    args = parser.parse_args()
    config = PublicPriceConfig(symbols=tuple(args.symbols), start=args.start, end=args.end)
    print(json.dumps(run_collection(args.project_root, config, collect=args.collect, refresh=args.refresh), indent=2))

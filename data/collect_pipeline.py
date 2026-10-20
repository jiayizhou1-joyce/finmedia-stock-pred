"""Opt-in Alpaca collection with validated local caches and no import-time API calls.

Official SDK behavior used here:
https://github.com/alpacahq/alpaca-py/blob/master/alpaca/data/historical/news.py
https://github.com/alpacahq/alpaca-py/blob/master/alpaca/common/rest.py

get_news/get_stock_bars already paginate. A supplied news limit can cap the
TOTAL SDK result, so this collector leaves it unset. API boundaries may be
inclusive; saved data are filtered to explicit half-open UTC intervals.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time

import pandas as pd


PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION = "alpaca-raw-v1"
NEWS_COLUMNS = ["ticker", "id", "created_at", "updated_at", "headline", "source", "symbols", "url", "retrieved_at"]
PRICE_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


class CollectionError(RuntimeError):
    """A collection failure whose message contains no credential values."""


def _utc(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("Collection bounds must include a time zone, for example 2020-01-01T00:00:00Z.")
    return result.tz_convert("UTC")


def _iso(value):
    return _utc(value).isoformat()


@dataclass(frozen=True)
class CollectionConfig:
    tickers: tuple = ("AAPL", "JNJ")
    benchmark: str = "SPY"
    start: str = "2020-01-01T00:00:00Z"
    end: str = "2026-10-16T00:00:00Z"
    price_feed: str = "sip"
    price_adjustment: str = "all"
    max_attempts: int = 3
    initial_retry_wait: float = 1.0
    max_retry_wait: float = 30.0

    def __post_init__(self):
        tickers = tuple(dict.fromkeys(str(value).strip().upper() for value in self.tickers))
        benchmark = str(self.benchmark).strip().upper()
        if not tickers or any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", value)
                              for value in (*tickers, benchmark)):
            raise ValueError("Use nonempty stock ticker symbols without path separators.")
        if _utc(self.start) >= _utc(self.end):
            raise ValueError("start must precede end.")
        if self.price_feed not in {"sip", "iex"}:
            raise ValueError("Choose sip or iex explicitly; there is no automatic feed fallback.")
        if self.price_adjustment not in {"raw", "split", "dividend", "all"}:
            raise ValueError("Unsupported price adjustment.")
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5.")
        if not 0 <= self.initial_retry_wait <= self.max_retry_wait <= 30:
            raise ValueError("Retry waits must satisfy 0 <= initial <= maximum <= 30 seconds.")
        object.__setattr__(self, "tickers", tickers)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "start", _iso(self.start))
        object.__setattr__(self, "end", _iso(self.end))

    @property
    def price_symbols(self):
        return tuple(dict.fromkeys((*self.tickers, self.benchmark)))


def _news_chunks(config):
    cursor, end = _utc(config.start), _utc(config.end)
    while cursor < end:
        following_year = pd.Timestamp(year=cursor.year + 1, month=1, day=1, tz="UTC")
        stop = min(following_year, end)
        yield cursor, stop
        cursor = stop


def credential_file_status(project_root):
    """Check presence only; do not read a key or load environment variables."""
    path = Path(project_root).expanduser().resolve() / ".env"
    return {"project_env_exists": path.is_file(), "credentials_loaded": False}


def plan_collection(project_root, config=None):
    """Describe requests without reading credentials, constructing clients, or using the network."""
    config = config or CollectionConfig()
    root = Path(project_root).expanduser().resolve()
    chunks = [{"start_inclusive": _iso(start), "end_exclusive": _iso(end)}
              for start, end in _news_chunks(config)]
    return {
        "status": "plan_only", "network_calls_executed": False,
        "credential_status": credential_file_status(root),
        "credential_source_when_enabled": str(root / ".env"),
        "credential_variable_names": ["API_KEY", "SECRET_KEY"],
        "news_tickers": list(config.tickers), "news_year_chunks_per_ticker": chunks,
        "price_symbols": list(config.price_symbols),
        "start_inclusive": config.start, "end_exclusive": config.end,
        "price_feed": config.price_feed, "price_adjustment": config.price_adjustment,
        "sdk_calls_if_all_caches_missing": len(config.tickers) * len(chunks) + 1,
        "pagination": "SDK automatic; no total-result limit supplied. HTTP request count is not estimated.",
        "network_when_enabled": "New requests use credentials from this project's .env only; no global environment fallback.",
        "cache_policy": "Reuse only after exact parameters, versions, schema, row count, and SHA-256 validate.",
        "completeness": "A successful request or cache validation does not establish API archive completeness.",
    }


def _versions():
    values = {"python": platform.python_version(), "pandas": pd.__version__}
    for package in ("alpaca-py", "pyarrow"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _identity(kind, parameters, versions):
    return {"kind": kind, "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION, "parameters": parameters, "versions": versions}


def _manifest_path(path):
    return path.with_suffix(".manifest.json")


def _schema(frame):
    return [{"name": name, "dtype": str(dtype)} for name, dtype in frame.dtypes.items()]


def _ranges(frame, timestamp_column):
    values = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce", format="mixed")
    return {"observed_start_utc": values.min().isoformat() if values.notna().any() else None,
            "observed_end_utc": values.max().isoformat() if values.notna().any() else None}


def _read_cache(path, identity, required_columns):
    """An invalid or interrupted cache is a cache miss, never an accepted partial file."""
    manifest_path = _manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        return None, "file_or_manifest_missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            return None, "parameters_or_versions_changed"
        if manifest.get("file_sha256") != _hash(path):
            return None, "file_hash_mismatch"
        frame = pd.read_parquet(path)
        if list(frame.columns) != list(required_columns):
            return None, "column_schema_mismatch"
        if manifest.get("schema") != _schema(frame):
            return None, "dtype_schema_mismatch"
        if manifest.get("row_count") != len(frame):
            return None, "row_count_mismatch"
        expected_status = "collected_nonempty" if len(frame) else "collected_empty"
        if manifest.get("status") != expected_status:
            return None, "uncommitted_or_wrong_status"
        timestamp_column = "timestamp" if identity["kind"] == "prices" else "created_at"
        if any(manifest.get(name) != value for name, value in _ranges(frame, timestamp_column).items()):
            return None, "observed_range_mismatch"
        return frame, "validated_cache"
    except Exception:
        return None, "unreadable_cache"


def _atomic_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write((json.dumps(content, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_cache(path, frame, identity, timestamp_column, extra=None):
    """Write parquet atomically; publish its validated manifest last as the commit marker.

    A crash between the two replacements leaves a missing/mismatched manifest,
    which _read_cache rejects on the next run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".parquet", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.to_parquet(temporary, index=False)
        persisted = pd.read_parquet(temporary)
        manifest = {
            "identity": identity, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "collected_nonempty" if len(frame) else "collected_empty",
            "row_count": len(frame), "schema": _schema(persisted),
            "file_sha256": _hash(temporary), **_ranges(persisted, timestamp_column),
            "api_archive_completeness_verified": False,
        }
        if extra:
            manifest.update(extra)
        os.replace(temporary, path)
        _atomic_json(_manifest_path(path), manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def _get(item, name, default=None):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _news_frame(response, ticker, start, end):
    data = _get(response, "data", response)
    articles = data.get("news", []) if isinstance(data, dict) else []
    if not isinstance(articles, (list, tuple)):
        raise CollectionError("News response did not contain the expected article list.")
    fetched_at = pd.Timestamp.now(tz="UTC")
    rows = [{"ticker": ticker, "id": _get(article, "id"),
             "created_at": _get(article, "created_at"), "updated_at": _get(article, "updated_at"),
             "headline": _get(article, "headline"), "source": _get(article, "source"),
             "symbols": _get(article, "symbols"), "url": _get(article, "url"),
             "retrieved_at": fetched_at} for article in articles]
    # Object construction avoids converting integer article IDs to imprecise floats
    # when some IDs are missing.
    frame = pd.DataFrame(rows, columns=NEWS_COLUMNS, dtype=object)
    for name in ("ticker", "id", "headline", "source", "url"):
        frame[name] = frame[name].astype("string")
    frame["id"] = frame["id"].str.strip().replace("", pd.NA)
    for name in ("created_at", "updated_at", "retrieved_at"):
        frame[name] = pd.to_datetime(frame[name], utc=True, errors="coerce", format="mixed").astype("datetime64[ns, UTC]")
    if frame["created_at"].isna().any():
        raise CollectionError("News response contains missing/invalid publication times; this chunk was not committed.")
    eligible = frame["created_at"].ge(start) & frame["created_at"].lt(end)
    details = {"response_rows": len(frame), "outside_half_open_window_rows": int((~eligible).sum())}
    return frame.loc[eligible].sort_values(["created_at", "id"], kind="stable").reset_index(drop=True), details


def _latest_news(frame):
    """Choose the latest known update by ticker+ID; every missing-ID row survives."""
    with_id = frame.loc[frame["id"].notna() & frame["id"].ne("")].copy()
    without_id = frame.loc[~(frame["id"].notna() & frame["id"].ne(""))].copy()
    duplicate_ids = with_id.duplicated(["ticker", "id"], keep=False)
    details = {
        "input_rows": len(frame), "missing_id_rows_retained": len(without_id),
        "records_sharing_ticker_and_id": int(duplicate_ids.sum()),
        "deduplication_key": ["ticker", "id"],
        "latest_version_rule": "Latest updated_at, then retrieved_at; missing update times sort first. Raw chunks retain versions.",
        "updated_after_publication_rows": int(frame["updated_at"].gt(frame["created_at"]).sum()),
        "point_in_time_text_verified": False,
        "point_in_time_note": "Historical API text may be revised. created_at alone does not prove wording was available at that time.",
    }
    newest = (with_id.sort_values(["updated_at", "retrieved_at"], kind="stable", na_position="first")
              .drop_duplicates(["ticker", "id"], keep="last"))
    result = (pd.concat([newest, without_id], ignore_index=True)
              .sort_values(["ticker", "created_at", "id"], kind="stable").reset_index(drop=True))
    details["duplicate_id_records_removed"] = len(frame) - len(result)
    details["output_rows"] = len(result)
    return result[NEWS_COLUMNS], details


def _price_frame(response, config):
    frame = _get(response, "df")
    if not isinstance(frame, pd.DataFrame):
        raise CollectionError("Price response did not contain a DataFrame.")
    frame = frame.copy()
    if {"symbol", "timestamp"}.issubset(set(frame.index.names)):
        frame = frame.reset_index()
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    if frame.empty:
        frame = pd.DataFrame(columns=PRICE_COLUMNS)
    elif not required.issubset(frame.columns):
        raise CollectionError("Price response lacks required bar columns; no price cache was committed.")
    for name in PRICE_COLUMNS:
        if name not in frame:
            frame[name] = pd.NA
    frame = frame[PRICE_COLUMNS]
    frame["symbol"] = frame["symbol"].astype("string")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed").astype("datetime64[ns, UTC]")
    if frame["timestamp"].isna().any():
        raise CollectionError("Price response contains invalid bar timestamps; no price cache was committed.")
    for name in PRICE_COLUMNS[2:]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    eligible = (frame["timestamp"].ge(_utc(config.start)) & frame["timestamp"].lt(_utc(config.end))
                & frame["symbol"].isin(config.price_symbols))
    details = {"response_rows": len(frame), "outside_requested_symbols_or_window_rows": int((~eligible).sum())}
    frame = frame.loc[eligible].sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
    details["duplicate_symbol_timestamp_rows_retained_for_review"] = int(frame.duplicated(["symbol", "timestamp"], keep=False).sum())
    details["row_count_by_symbol"] = {str(name): int(count) for name, count in frame.groupby("symbol").size().items()}
    details["missing_numeric_counts"] = {name: int(frame[name].isna().sum()) for name in PRICE_COLUMNS[2:]}
    return frame, details


def _http_status(error):
    try:
        value = getattr(error, "status_code", None)
    except Exception:
        value = None
    response = getattr(error, "response", None)
    if response is None:
        response = getattr(getattr(error, "_http_error", None), "response", None)
    return value if value is not None else getattr(response, "status_code", None)


def _is_connection_error(error):
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    try:
        from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout, SSLError
        return isinstance(error, (RequestsConnectionError, Timeout)) and not isinstance(error, SSLError)
    except ImportError:
        return False


def _request_with_retry(operation, config, sleep, label):
    for attempt in range(config.max_attempts):
        try:
            return operation()
        except Exception as error:
            status = _http_status(error)
            transient = status in {429, 500, 502, 503, 504} or (status is None and _is_connection_error(error))
            if not transient or attempt + 1 >= config.max_attempts:
                category = f"HTTP {status}" if status is not None else type(error).__name__
                raise CollectionError(f"{label} failed ({category}); no feed or credential fallback was attempted.") from None
            wait = min(config.initial_retry_wait * 2 ** attempt, config.max_retry_wait)
            sleep(wait)


def _project_credentials(root):
    env_path = root / ".env"
    if not env_path.is_file():
        raise CollectionError("Create the project's .env with API_KEY and SECRET_KEY before enabling collection.")
    try:
        from dotenv import dotenv_values
    except ImportError:
        raise CollectionError("python-dotenv is needed to read the project's .env.") from None
    # interpolate=False prevents a project file from silently borrowing credentials
    # from global environment variables. Nothing is exported to os.environ.
    values = dotenv_values(env_path, interpolate=False)
    key, secret = values.get("API_KEY"), values.get("SECRET_KEY")
    if not key or not secret or not key.strip() or not secret.strip():
        raise CollectionError("The project's .env must contain nonempty API_KEY and SECRET_KEY entries.")
    if any(re.search(r"\$\{[^}]+\}", value) for value in (key, secret)):
        raise CollectionError("Enter credentials directly in the project's .env; variable interpolation is disabled.")
    return key.strip(), secret.strip()


def _sdk_client(kind, root):
    key, secret = _project_credentials(root)
    try:
        if kind == "news":
            from alpaca.data.historical.news import NewsClient
            client = NewsClient(key, secret)
        else:
            from alpaca.data.historical import StockHistoricalDataClient
            client = StockHistoricalDataClient(key, secret)
    except ImportError:
        raise CollectionError("alpaca-py is needed only when live collection is enabled.") from None
    # Current SDK public constructors do not expose retries. Disable its internal
    # loop so the bounded policy above is the only retry layer. Refuse an unknown
    # layout instead of silently multiplying attempts after an SDK change.
    if not hasattr(client, "_retry"):
        raise CollectionError("This alpaca-py version has an unknown retry configuration; review before collecting.")
    client._retry = 0
    return client


def _sdk_request(kind, parameters):
    try:
        from alpaca.data.requests import NewsRequest, StockBarsRequest
    except ImportError:
        raise CollectionError("alpaca-py is needed only when live collection is enabled.") from None
    if kind == "news":
        return NewsRequest(symbols=parameters["ticker"], start=_utc(parameters["start"]).to_pydatetime(),
                           end=_utc(parameters["end"]).to_pydatetime(), sort="asc", limit=None,
                           include_content=False, exclude_contentless=False)
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.timeframe import TimeFrame
    return StockBarsRequest(symbol_or_symbols=parameters["symbols"], timeframe=TimeFrame.Day,
                            start=_utc(parameters["start"]).to_pydatetime(), end=_utc(parameters["end"]).to_pydatetime(),
                            feed=DataFeed(parameters["feed"]), adjustment=Adjustment(parameters["adjustment"]), limit=None)


def collect_dataset(
    project_root,
    config=None,
    *,
    dry_run=False,
    news_client=None,
    price_client=None,
    request_factory=None,
    sleep=time.sleep,
    progress=None,
):
    """Explicitly collect or reuse raw data. Calling this with dry_run=False is opt-in.

    Fake clients may be injected for offline tests: they receive parameter dicts
    by default and return SDK-shaped responses. For injected real SDK clients,
    pass request_factory=_sdk_request. Default real clients are constructed lazily
    only for missing/invalid caches and use this project's .env exclusively.
    """
    config = config or CollectionConfig()
    root = Path(project_root).expanduser().resolve()
    if dry_run:
        return plan_collection(root, config)
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory.")
    versions = _versions()
    events = []
    summary = {"status": "collecting", "pipeline_version": PIPELINE_VERSION,
               "versions": versions, "news": {}, "prices": {}, "events": events,
               "api_archive_completeness_verified": False,
               "credential_source_for_new_requests": {
                   "news": "injected client (not inspected)" if news_client is not None else str(root / ".env"),
                   "prices": "injected client (not inspected)" if price_client is not None else str(root / ".env"),
               },
               "point_in_time_text_verified": False}
    injected_news, injected_prices = news_client is not None, price_client is not None

    def emit(event):
        events.append(event)
        if progress:
            progress(event.copy())

    def request(kind, parameters):
        nonlocal news_client, price_client
        if request_factory:
            built = request_factory(kind, parameters.copy())
        elif (kind == "news" and injected_news) or (kind == "prices" and injected_prices):
            built = parameters.copy()
        else:
            built = _sdk_request(kind, parameters)
        if kind == "news":
            if news_client is None:
                news_client = _sdk_client("news", root)
            return _request_with_retry(lambda: news_client.get_news(built), config, sleep, "News request")
        if price_client is None:
            price_client = _sdk_client("prices", root)
        return _request_with_retry(lambda: price_client.get_stock_bars(built), config, sleep, "Price request")

    try:
        for ticker in config.tickers:
            chunks, chunk_provenance = [], []
            for start, end in _news_chunks(config):
                parameters = {"ticker": ticker, "start": _iso(start), "end": _iso(end),
                              "saved_interval": "start <= created_at < end", "sort": "asc",
                              "limit": None, "include_content": False, "exclude_contentless": False}
                identity = _identity("news_chunk", parameters, versions)
                suffix = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:10]
                path = root / "data/raw/news/chunks" / ticker / f"{start:%Y%m%d}_{end:%Y%m%d}_{suffix}.parquet"
                frame, reason = _read_cache(path, identity, NEWS_COLUMNS)
                if frame is None:
                    frame, details = _news_frame(request("news", parameters), ticker, start, end)
                    _write_cache(path, frame, identity, "created_at", details)
                    action = "fetched"
                else:
                    action = "reused"
                chunks.append(frame)
                chunk_provenance.append({"path": str(path.relative_to(root)), "file_sha256": _hash(path)})
                emit({"kind": "news_chunk", "ticker": ticker, "start": _iso(start), "end": _iso(end),
                      "action": action, "cache_reason": reason, "rows": len(frame),
                      "empty": frame.empty})
            path = root / "data/raw/news" / f"{ticker}.parquet"
            parameters = {"ticker": ticker, "start": config.start, "end": config.end,
                          "chunks": chunk_provenance, "deduplication": "latest update by ticker+nonmissing ID"}
            identity = _identity("news_ticker", parameters, versions)
            combined, reason = _read_cache(path, identity, NEWS_COLUMNS)
            if combined is None:
                combined, details = _latest_news(pd.concat(chunks, ignore_index=True))
                manifest = _write_cache(path, combined, identity, "created_at", details)
            else:
                manifest = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
            summary["news"][ticker] = {"path": str(path), "rows": len(combined),
                                       "cache_reason": reason, **_ranges(combined, "created_at"),
                                       "missing_id_rows_retained": manifest.get("missing_id_rows_retained", 0),
                                       "duplicate_id_records_removed": manifest.get("duplicate_id_records_removed", 0)}
        parameters = {"symbols": list(config.price_symbols), "start": config.start, "end": config.end,
                      "timeframe": "Day", "feed": config.price_feed, "adjustment": config.price_adjustment,
                      "limit": None, "saved_interval": "start <= bar timestamp < end"}
        identity = _identity("prices", parameters, versions)
        path = root / "data/raw/prices.parquet"
        prices, reason = _read_cache(path, identity, PRICE_COLUMNS)
        if prices is None:
            prices, details = _price_frame(request("prices", parameters), config)
            _write_cache(path, prices, identity, "timestamp", details)
            action = "fetched"
        else:
            action = "reused"
        summary["prices"] = {"path": str(path), "rows": len(prices), "feed": config.price_feed,
                             "adjustment": config.price_adjustment, **_ranges(prices, "timestamp")}
        emit({"kind": "prices", "action": action, "cache_reason": reason, "rows": len(prices), "empty": prices.empty})
        summary["status"] = "requests_finished_archive_completeness_unverified"
    except CollectionError as error:
        summary["status"] = "failed_partial_caches_preserved"
        summary["error"] = str(error)
        _atomic_json(root / "data/raw/collection_run.json", summary)
        raise
    _atomic_json(root / "data/raw/collection_run.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--collect", action="store_true", help="Explicitly enable new API calls for missing/invalid caches.")
    args = parser.parse_args()
    print(json.dumps(collect_dataset(args.project_root, dry_run=not args.collect), indent=2))

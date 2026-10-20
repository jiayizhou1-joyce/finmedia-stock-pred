"""Opt-in, local evaluation of EODHD's documented public demo news API.

No credentials, signup, network calls, or writes occur in the default plan mode.
Official endpoint/demo/pagination documentation:
https://eodhd.com/financial-apis/stock-market-financial-news-api
Redistribution/display restrictions: https://eodhd.com/financial-apis/terms-conditions

All responses and normalized data stay under the git-ignored data/ directory.
Demo access is not a publication license: obtain suitable academic permission
before sharing raw records or displaying provider data in a public artifact.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


ENDPOINT = "https://eodhd.com/api/news"
VERSION = "1.0.0"
DEMO_TICKERS = {"AAPL", "AMZN", "TSLA", "VTI"}
COLUMNS = ["ticker", "created_at", "headline", "link", "source_domain", "symbols", "tags",
           "sentiment_polarity", "sentiment_neg", "sentiment_neu", "sentiment_pos",
           "sentiment_metadata_json", "provider", "retrieved_at", "query_window_start",
           "query_window_end", "raw_response_sha256"]


@dataclass(frozen=True)
class PublicNewsConfig:
    tickers: tuple = ("AAPL", "AMZN")
    start: str = "2024-01-01"
    end: str = "2025-01-01"
    page_size: int = 1000
    max_pages_per_window: int = 10
    max_attempts: int = 2
    spacing_seconds: float = 1.1
    timeout_seconds: float = 30.0

    def __post_init__(self):
        tickers = tuple(dict.fromkeys(str(t).upper().strip() for t in self.tickers))
        if not tickers or not set(tickers).issubset(DEMO_TICKERS):
            raise ValueError("Use only documented US demo symbols: AAPL, AMZN, TSLA, VTI.")
        start, end = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if start >= end:
            raise ValueError("start must precede the exclusive end date.")
        if not 1 <= self.page_size <= 1000 or not 1 <= self.max_pages_per_window <= 10:
            raise ValueError("page_size must be 1–1000; max_pages_per_window must be 1–10.")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be 1–3.")
        if not 1 < self.spacing_seconds <= 60 or not 0 < self.timeout_seconds <= 60:
            raise ValueError("Use spacing greater than one second and a timeout no greater than 60 seconds.")
        object.__setattr__(self, "tickers", tickers)


def _windows(config):
    cursor, end = date.fromisoformat(config.start), date.fromisoformat(config.end)
    while cursor < end:
        next_month = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
        stop = min(next_month, end)
        yield cursor, stop
        cursor = stop


def plan_public_news(project_root, config=None):
    config = config or PublicNewsConfig()
    return {
        "status": "plan_only", "network_calls_executed": False,
        "provider": "EODHD public demo", "endpoint": ENDPOINT,
        "config": asdict(config),
        "monthly_windows": [{"start_inclusive": str(a), "end_exclusive": str(b)}
                            for a, b in _windows(config)],
        "output_directory": str(Path(project_root).expanduser().resolve() / "data/supplemental/eodhd_news"),
        "credentials": "Public provider-documented demo token only; no user credentials read.",
        "publication_permission": "Local evaluation only; confirm academic permission before sharing/displaying provider data.",
        "archive_completeness_verified": False,
    }


def _digest(body):
    return hashlib.sha256(body).hexdigest()


def _atomic_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _http_get(parameters, timeout):
    request = Request(ENDPOINT + "?" + urlencode(parameters), headers={
        "User-Agent": "DATA1030 local API evaluation", "Accept": "application/json",
    })
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()


def _page_cache(directory, identity):
    key = _digest(json.dumps(identity, sort_keys=True).encode())[:24]
    return directory / f"page_{key}.manifest.json"


def _read_page_cache(manifest_path, identity):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity or manifest["http_status"] != 200:
            return None
        filename = manifest["raw_filename"]
        if Path(filename).name != filename:
            return None
        raw = manifest_path.parent / filename
        body = raw.read_bytes()
        if _digest(body) != manifest["sha256"]:
            return None
        payload = json.loads(body)
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            return None
        if len(payload) != manifest["response_rows"]:
            return None
        return payload, {**manifest, "raw_path": str(raw), "cache_reused": True}
    except (OSError, ValueError, TypeError, KeyError):
        return None


class _CollectionStopped(Exception):
    def __init__(self, reason, **details):
        self.details = {"reason": reason, **details}
        super().__init__(reason)


def _frame(rows):
    import pandas as pd
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for name in ("created_at", "retrieved_at"):
        frame[name] = pd.to_datetime(frame[name], utc=True, errors="coerce", format="mixed").astype("datetime64[ns, UTC]")
    for name in ("sentiment_polarity", "sentiment_neg", "sentiment_neu", "sentiment_pos"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    for name in set(COLUMNS) - {"created_at", "retrieved_at", "symbols", "tags",
                              "sentiment_polarity", "sentiment_neg", "sentiment_neu", "sentiment_pos"}:
        frame[name] = frame[name].astype("string")
    # Only exact, nonmissing full URL + title + time matches are removed. Query
    # parameters, path case and fragments are retained; missing-key rows survive.
    key = ["ticker", "link", "headline", "created_at"]
    complete = (frame[key].notna().all(axis=1) & frame["source_domain"].notna()
                & frame["link"].str.strip().ne("") & frame["headline"].str.strip().ne(""))
    duplicate = complete & frame.duplicated(key, keep="first")
    removed = int(duplicate.sum())
    return frame.loc[~duplicate].sort_values(["ticker", "created_at"], kind="stable").reset_index(drop=True), removed


def _normalize(payload, ticker, start, end, page):
    import pandas as pd
    lower, upper = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    rows = []
    counts = {"invalid_or_naive_date_rows": 0, "out_of_window_rows": 0,
              "blank_headline_rows": 0, "missing_or_invalid_link_rows": 0,
              "requested_ticker_absent_from_symbols_rows": 0}
    for article in payload:
        try:
            timestamp = pd.Timestamp(article.get("date"))
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError("Missing/naive provider time")
            timestamp = timestamp.tz_convert("UTC")
        except (ValueError, TypeError, OverflowError):
            counts["invalid_or_naive_date_rows"] += 1
            continue
        if not lower <= timestamp < upper:
            counts["out_of_window_rows"] += 1
            continue
        headline = article.get("title") if isinstance(article.get("title"), str) else None
        counts["blank_headline_rows"] += int(not headline or not headline.strip())
        link = article.get("link") if isinstance(article.get("link"), str) else None
        try:
            parsed = urlsplit(link or "")
            domain = parsed.hostname if parsed.scheme in ("http", "https") else None
        except ValueError:
            domain = None
        counts["missing_or_invalid_link_rows"] += int(not domain)
        symbols = article.get("symbols")
        symbols = symbols if isinstance(symbols, list) and all(isinstance(x, str) for x in symbols) else []
        tags = article.get("tags")
        tags = tags if isinstance(tags, list) and all(isinstance(x, str) for x in tags) else []
        counts["requested_ticker_absent_from_symbols_rows"] += int(ticker + ".US" not in symbols)
        sentiment = article.get("sentiment")
        sentiment = sentiment if isinstance(sentiment, dict) else {}
        rows.append({
            "ticker": ticker, "created_at": timestamp, "headline": headline, "link": link,
            "source_domain": domain, "symbols": symbols, "tags": tags,
            **{f"sentiment_{name}": sentiment.get(name) for name in ("polarity", "neg", "neu", "pos")},
            "sentiment_metadata_json": json.dumps(sentiment, sort_keys=True) if sentiment else None,
            "provider": "EODHD", "retrieved_at": page["retrieved_at_utc"],
            "query_window_start": str(start), "query_window_end": str(end),
            "raw_response_sha256": page["sha256"],
        })
    return rows, counts


def collect_public_news(project_root, config=None, *, collect=False, transport=None,
                        sleep=time.sleep, monotonic=time.monotonic, progress=None):
    """Plan by default; explicit ``collect=True`` enables requests or cache reuse.

    ``transport(parameters, timeout) -> (status, headers, exact_body_bytes)`` is
    injectable for offline tests. HTTP 401/403/429 stop the run immediately;
    only connection failures and 5xx responses receive bounded retries.
    """
    config = config or PublicNewsConfig()
    if not collect:
        return plan_public_news(project_root, config)
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory.")
    transport = transport or _http_get
    directory = root / "data/supplemental/eodhd_news"
    raw_directory = directory / "raw"
    report = {
        "status": "collecting", "provider": "EODHD", "collector_version": VERSION,
        "endpoint": ENDPOINT, "config": asdict(config),
        "retrieval_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "network_requests": 0, "cache_pages_reused": 0, "pages": [], "windows": [], "issues": [],
        "archive_completeness_verified": False, "point_in_time_text_verified": False,
        "timestamp_note": "created_at is provider-reported publication date. Initial ingestion and revision times are not supplied; historical wording/sentiment may be revised.",
        "empty_window_note": "An empty response is not evidence that no relevant news existed or that the source archive is complete.",
        "sharing": "Local demo evaluation only. Obtain academic publication permission before sharing raw data or displaying provider data; do not commit data/.",
        "sentiment_note": "Provider metadata, not independently validated labels or verified point-in-time predictors.",
    }
    rows = []
    last_finished = None

    def page_request(parameters):
        nonlocal last_finished
        identity = {"collector_version": VERSION, "endpoint": ENDPOINT, "parameters": parameters}
        manifest_path = _page_cache(raw_directory, identity)
        cached = _read_page_cache(manifest_path, identity)
        if cached is not None:
            report["cache_pages_reused"] += 1
            return cached
        for attempt in range(1, config.max_attempts + 1):
            if last_finished is not None:
                sleep(max(0.0, config.spacing_seconds - (monotonic() - last_finished)))
            report["network_requests"] += 1
            try:
                status, headers, body = transport(parameters.copy(), config.timeout_seconds)
            except (URLError, ConnectionError, TimeoutError, OSError):
                last_finished = monotonic()
                if attempt < config.max_attempts:
                    sleep(min(2 ** (attempt - 1), 5))
                    continue
                raise _CollectionStopped("network_error_after_bounded_attempts", attempts=attempt)
            last_finished = monotonic()
            if not isinstance(body, bytes):
                raise TypeError("Transport must preserve response bodies as bytes.")
            retrieved = datetime.now(timezone.utc)
            filename = f"{manifest_path.stem}_{retrieved:%Y%m%dT%H%M%S%fZ}_attempt{attempt}.response"
            raw_path = raw_directory / filename
            _atomic_bytes(raw_path, body)
            metadata = {
                "identity": identity, "http_status": int(status), "sha256": _digest(body),
                "raw_filename": filename, "raw_path": str(raw_path),
                "retrieved_at_utc": retrieved.isoformat(), "cache_reused": False,
                "response_headers": {str(k): str(v) for k, v in headers.items()
                                     if str(k).lower() in {"content-type", "date", "retry-after"}},
            }
            _atomic_bytes(raw_path.with_suffix(".request.json"), _json_bytes(metadata))
            if status != 200:
                if 500 <= status < 600 and attempt < config.max_attempts:
                    sleep(min(2 ** (attempt - 1), 5))
                    continue
                raise _CollectionStopped("http_error_collection_stopped", http_status=int(status),
                                         raw_path=str(raw_path), retry_after=headers.get("Retry-After"))
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                raise _CollectionStopped("invalid_json_response", raw_path=str(raw_path))
            if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
                raise _CollectionStopped("unexpected_response_schema", raw_path=str(raw_path))
            metadata["response_rows"] = len(payload)
            _atomic_bytes(manifest_path, _json_bytes(metadata))
            return payload, metadata

    stopped = False
    for ticker in config.tickers:
        if stopped:
            break
        for start, end in _windows(config):
            window = {"ticker": ticker, "start_inclusive": str(start), "end_exclusive": str(end),
                      "status": "collecting", "response_rows": 0, "retained_rows_before_dedup": 0}
            report["windows"].append(window)
            seen_pages = set()
            for number in range(config.max_pages_per_window):
                offset = number * config.page_size
                if offset > 9000:
                    window["status"] = "capped_window"
                    report["issues"].append({**window, "reason": "offset_exceeds_documented_9000_cap"})
                    break
                parameters = {"api_token": "demo", "s": ticker + ".US", "from": str(start),
                              "to": str(end - timedelta(days=1)), "limit": config.page_size,
                              "offset": offset, "fmt": "json"}
                try:
                    payload, metadata = page_request(parameters)
                except _CollectionStopped as error:
                    window["status"] = "request_failed"
                    report["issues"].append({"ticker": ticker, "window_start": str(start),
                                             "offset": offset, **error.details})
                    stopped = True
                    break
                report["pages"].append(metadata)
                fingerprint = _digest(json.dumps(sorted(json.dumps(row, sort_keys=True) for row in payload)).encode())
                if payload and fingerprint in seen_pages:
                    window["status"] = "duplicate_page"
                    report["issues"].append({"ticker": ticker, "window_start": str(start),
                                             "offset": offset, "reason": "duplicate_page_pagination_unverified"})
                    break
                seen_pages.add(fingerprint)
                window["response_rows"] += len(payload)
                normalized, checks = _normalize(payload, ticker, start, end, metadata)
                rows.extend(normalized)
                window["retained_rows_before_dedup"] += len(normalized)
                for name, count in checks.items():
                    window[name] = window.get(name, 0) + count
                if checks["invalid_or_naive_date_rows"] or checks["out_of_window_rows"]:
                    report["issues"].append({"ticker": ticker, "window_start": str(start),
                                             "offset": offset, "reason": "invalid_or_out_of_window_dates", **checks})
                if len(payload) < config.page_size:
                    window["status"] = "pagination_finished_archive_completeness_unverified"
                    if window["retained_rows_before_dedup"] == 0:
                        window["status"] = "empty_window"
                        report["issues"].append({"ticker": ticker, "window_start": str(start),
                                                 "reason": "empty_window_not_proven_news_free"})
                    break
            else:
                window["status"] = "capped_window"
                report["issues"].append({"ticker": ticker, "window_start": str(start),
                                         "reason": "page_budget_exhausted_window_may_be_truncated"})
            if progress:
                progress(window.copy())
            if stopped:
                break
    frame, removed = _frame(rows)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "news.parquet"
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".news-", suffix=".parquet", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report.update({
        "status": "partial" if report["issues"] else "requests_finished_archive_completeness_unverified",
        "normalized_path": str(output), "normalized_sha256": _digest(output.read_bytes()),
        "normalized_rows": len(frame), "exact_duplicate_records_removed": removed,
        "deduplication": "Exact ticker + full URL + exact title + UTC publication time; incomplete-key rows retained. No query strings removed.",
        "rows_by_ticker": {str(t): int(n) for t, n in frame.groupby("ticker").size().items()},
        "retrieval_finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_bytes(directory / "collection_run.json", _json_bytes(report))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--collect", action="store_true", help="Opt in to local demo API collection.")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "AMZN"])
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-01-01", help="Exclusive UTC end date.")
    args = parser.parse_args()
    settings = PublicNewsConfig(tickers=tuple(args.tickers), start=args.start, end=args.end)
    result = collect_public_news(args.project_root, settings, collect=args.collect)
    if args.collect:
        result = {key: result[key] for key in ("status", "network_requests", "cache_pages_reused",
                                               "normalized_rows", "rows_by_ticker", "normalized_path", "issues")}
    print(json.dumps(result, indent=2))

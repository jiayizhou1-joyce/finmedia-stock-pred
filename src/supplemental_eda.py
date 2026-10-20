"""Descriptive EDA of separately collected public-interface evaluation samples.

Reads local files only. News and price provenance remain separate from the
group's original Alpaca data. API retrieval is never triggered by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

COLORS = {"AAPL": "#2367a4", "JNJ": "#c87525", "SPY": "#687582", "AMZN": "#338575"}


def _json_finite(value):
    if isinstance(value, dict):
        return {str(k): _json_finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_finite(value.item())
    return value


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_prices(prices, start, end):
    """Reindex to actual sessions before returns/labels; never fill missing prices."""
    required = {"symbol", "date", "open", "high", "low", "close", "volume", "adjusted_close"}
    if missing := required - set(prices):
        raise ValueError(f"Price columns missing: {sorted(missing)}")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.loc[frame.date.ge(pd.Timestamp(start)) & frame.date.lt(pd.Timestamp(end))]
    if frame.empty:
        raise ValueError("No prices in the requested EDA window.")
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=start, end_date=pd.Timestamp(end) - pd.Timedelta(days=1))
    schedule.index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    rows, checks = [], []
    for symbol, group in frame.groupby("symbol", sort=True):
        duplicate = group.duplicated("date", keep=False)
        numeric = group[["open", "high", "low", "close", "adjusted_close", "volume"]].apply(pd.to_numeric, errors="coerce")
        valid_numeric = np.isfinite(numeric).all(axis=1) & numeric[["open", "high", "low", "close", "adjusted_close"]].gt(0).all(axis=1) & numeric.volume.ge(0)
        valid_ohlc = (numeric.high.ge(numeric[["open", "close", "low"]].max(axis=1)) &
                      numeric.low.le(numeric[["open", "close", "high"]].min(axis=1)))
        valid = ~duplicate & valid_numeric & valid_ohlc & group.date.isin(schedule.index)
        checks.append({"symbol": symbol, "rows": len(group), "duplicate_date_rows": int(duplicate.sum()),
                       "invalid_numeric_rows": int((~valid_numeric).sum()), "invalid_ohlc_rows": int((~valid_ohlc).sum()),
                       "non_session_rows": int((~group.date.isin(schedule.index)).sum()),
                       "expected_sessions": len(schedule), "usable_sessions": int(valid.sum()),
                       "missing_or_excluded_sessions": len(schedule) - int(valid.sum()),
                       "first_observed_date": group.date.min().date().isoformat(),
                       "last_observed_date": group.date.max().date().isoformat()})
        clean = group.loc[valid].copy()
        clean[numeric.columns] = numeric.loc[valid]
        clean = clean.set_index("date").reindex(schedule.index)
        clean.index.name = "date"
        clean["symbol"] = symbol
        clean["return_pct"] = clean.adjusted_close.pct_change(fill_method=None) * 100
        clean["next_return_pct"] = clean.return_pct.shift(-1)
        clean["target_up"] = clean.next_return_pct.gt(0).astype("Int64").where(clean.next_return_pct.notna())
        rows.append(clean.reset_index())
    return pd.concat(rows, ignore_index=True), pd.DataFrame(checks), schedule


def align_news(news, schedule):
    """Assign provider timestamps to the first actual NYSE close at/after them."""
    frame = news.copy()
    if "created_at" not in frame and "date" in frame:
        frame["created_at"] = frame["date"]
    required = {"ticker", "created_at", "headline", "link"}
    if missing := required - set(frame):
        raise ValueError(f"News columns missing: {sorted(missing)}")
    frame["created_at"] = pd.to_datetime(frame.created_at, utc=True, errors="coerce", format="mixed")
    frame["midnight_utc_timestamp"] = (frame.created_at.notna() & frame.created_at.dt.hour.eq(0)
                                        & frame.created_at.dt.minute.eq(0) & frame.created_at.dt.second.eq(0)
                                        & frame.created_at.dt.microsecond.eq(0))
    frame["headline"] = frame.headline.astype("string")
    frame["blank_headline"] = frame.headline.isna() | frame.headline.str.strip().eq("").fillna(True)
    frame["blank_link"] = frame.link.isna() | frame.link.astype("string").str.strip().eq("").fillna(True)
    frame["duplicate_exact_record"] = frame.duplicated(["ticker", "created_at", "headline", "link"], keep=False)
    complete_key = frame.created_at.notna() & ~frame.blank_headline & ~frame.blank_link & frame.ticker.notna()
    frame["exclude_duplicate_copy"] = complete_key & frame.duplicated(["ticker", "created_at", "headline", "link"], keep="first")
    normalized_title = frame.headline.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    link_counts = frame.assign(_title_key=normalized_title).groupby(
        ["ticker", "created_at", "_title_key"], dropna=False)["link"].transform("nunique")
    frame["same_headline_time_across_links"] = frame.created_at.notna() & ~frame.blank_headline & link_counts.gt(1)
    frame["cutoff_date"] = pd.NaT
    valid_times = frame.created_at.notna()
    closes = pd.DatetimeIndex(schedule.market_close)
    positions = closes.searchsorted(pd.DatetimeIndex(frame.loc[valid_times, "created_at"]), side="left")
    eligible_indices = frame.index[valid_times][positions < len(closes)]
    frame.loc[eligible_indices, "cutoff_date"] = schedule.index[positions[positions < len(closes)]]
    # Company aliases are an audit flag, not a claim that all other news is irrelevant.
    aliases = {"AAPL": r"\b(?:apple|aapl)\b", "AMZN": r"\b(?:amazon|amzn|aws)\b"}
    frame["company_named_in_headline"] = False
    for ticker, pattern in aliases.items():
        selected = frame.ticker.eq(ticker)
        frame.loc[selected, "company_named_in_headline"] = frame.loc[selected, "headline"].str.contains(pattern, case=False, regex=True, na=False)
    return frame


def run_supplemental_eda(project_root, start="2020-01-01", end="2026-10-16"):
    root = Path(project_root).resolve()
    price_path = root / "data/supplemental/yahoo_prices/prices.parquet"
    news_path = root / "data/supplemental/eodhd_news/news.parquet"
    figures = root / "figures"
    results = root / "results"
    figures.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    for name in ("01_indexed_prices", "02_return_distributions", "03_rolling_volatility",
                 "04_target_balance", "05_monthly_news", "06_news_and_next_return", "07_headline_lengths"):
        (figures / f"eda_supplemental_{name}.png").unlink(missing_ok=True)
    report = {"analysis_basis": "separate_local_evaluation_sample", "start_inclusive": start,
              "end_exclusive": end, "files": {}, "figures": [],
              "limitations": ["This sample does not validate the missing original Alpaca raw files.",
                              "Yahoo chart is an undocumented public interface; adjusted series are a current retrieval, not point-in-time history.",
                              "All results are descriptive. No predictive improvement or trading profitability was measured."]}
    for path in (price_path, news_path):
        report["files"][str(path.relative_to(root))] = {"exists": path.is_file(), "sha256": _digest(path) if path.is_file() else None}
    if not price_path.is_file():
        report["status"] = "waiting_for_prices"
        return report
    input_prices = pd.read_parquet(price_path)
    report["price_schema"] = [{"column": col, "dtype": str(input_prices[col].dtype), "missing_rows": int(input_prices[col].isna().sum())} for col in input_prices]
    daily, quality, schedule = prepare_prices(input_prices, start, end)
    report["price_quality"] = quality.to_dict("records")
    price_gaps = bool(quality.missing_or_excluded_sessions.gt(0).any())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    def save(fig, name):
        path = figures / f"eda_supplemental_{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        report["figures"].append(str(path.relative_to(root)))

    pivot = daily.pivot(index="date", columns="symbol", values="adjusted_close")
    common = pivot.dropna(how="any")
    if not common.empty:
        first = common.index[0]
        indexed = pivot.loc[first:].div(pivot.loc[first]).mul(100)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for symbol in indexed:
            ax.plot(indexed.index, indexed[symbol], label=symbol, color=COLORS.get(symbol), linewidth=1.2)
        ax.set(title="Adjusted prices show different paths across the sample", ylabel=f"Adjusted close (100 on {first.date()})", xlabel="Trading date")
        ax.grid(alpha=.15)
        ax.legend(ncol=len(indexed.columns), frameon=False)
        fig.text(.12, .005, "Source: separately collected Yahoo chart data. Historical price growth does not establish predictability.", fontsize=8, color="#59636d")
        fig.tight_layout(rect=(0, .04, 1, 1))
        save(fig, "01_indexed_prices")

    summaries = []
    fig, axes = plt.subplots(1, len(pivot.columns), figsize=(12, 3.8), sharex=True, sharey=True, squeeze=False)
    all_returns = daily.return_pct.dropna()
    max_abs = float(all_returns.abs().max()) if len(all_returns) else 1
    bins = np.linspace(-max_abs - .1, max_abs + .1, 61)
    for ax, symbol in zip(axes.flat, pivot.columns):
        group = daily.loc[daily.symbol.eq(symbol)]
        values = group.return_pct.dropna()
        ax.hist(values, bins=bins, color=COLORS.get(symbol), alpha=.85)
        ax.axvline(0, color="#555", lw=.7)
        ax.set(title=symbol, xlabel="Daily adjusted return (%)")
        summaries.append({"symbol": symbol, "return_observations": len(values), "mean_daily_return_pct": values.mean(),
                          "daily_return_std_pct": values.std(), "min_daily_return_pct": values.min(), "max_daily_return_pct": values.max(),
                          "up_day_pct": values.gt(0).mean() * 100})
    axes[0, 0].set_ylabel("Trading days (count)")
    fig.suptitle("Daily returns have occasional large moves", x=.065, ha="left")
    fig.text(.065, .005, "Identical bins and scales. No outliers trimmed; returns across a missing session are excluded.", fontsize=8, color="#59636d")
    fig.tight_layout(rect=(0, .05, 1, .93))
    save(fig, "02_return_distributions")
    report["return_summary"] = pd.DataFrame(summaries).round(5).to_dict("records")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for symbol, group in daily.groupby("symbol"):
        ax.plot(group.date, group.return_pct.rolling(21, min_periods=21).std(), label=symbol, color=COLORS.get(symbol), lw=1)
    ax.set(title="The size of daily price changes varies over time", xlabel="Trading date", ylabel="21-session rolling standard deviation (%)")
    ax.grid(alpha=.15)
    ax.legend(ncol=len(pivot.columns), frameon=False)
    fig.text(.12, .005, "Source: Yahoo chart data. This describes changing volatility; it does not test a news effect.", fontsize=8, color="#59636d")
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "03_rolling_volatility")

    counts = []
    for symbol in [s for s in pivot.columns if s != "SPY"]:
        target = daily.loc[daily.symbol.eq(symbol), "target_up"].dropna()
        counts.append({"symbol": symbol, "down_or_unchanged": int(target.eq(0).sum()), "up": int(target.eq(1).sum()),
                       "labeled_stock_days": len(target), "up_pct": float(target.mean() * 100) if len(target) else None})
    report["target_balance"] = counts
    labels = pd.DataFrame(counts).set_index("symbol")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(labels))
    for offset, name, color in [(-.19, "down_or_unchanged", "#9da9b3"), (.19, "up", "#2367a4")]:
        bars = ax.bar(x + offset, labels[name], width=.36, label=name.replace("_", " ").capitalize(), color=color)
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(x, labels.index)
    ax.set(title="Check the target balance before comparing models", ylabel="Labeled stock-days (count)", xlabel="Stock")
    ax.set_ylim(0, labels[["down_or_unchanged", "up"]].max().max() * 1.22)
    ax.legend(frameon=False)
    fig.text(.12, .005, "y = 1 if next session's adjusted close > today's; y = 0 otherwise. Missing outcomes excluded.", fontsize=8, color="#59636d")
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, "04_target_balance")

    report["status"] = "partial_prices_news_unavailable" if price_gaps else "prices_complete_news_unavailable"
    aligned = None
    if news_path.is_file():
        manifest_path = news_path.parent / "collection_run.json"
        news_collection_partial = True
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            report["news_collection"] = {k: manifest.get(k) for k in ["status", "config", "normalized_rows", "issues", "retrieval_finished_at_utc"]}
            news_collection_partial = manifest.get("status") != "requests_finished_archive_completeness_unverified"
        else:
            report["limitations"].append("News collection manifest is missing; request completion cannot be checked.")
        news = pd.read_parquet(news_path)
        report["news_schema"] = [{"column": col, "dtype": str(news[col].dtype), "missing_rows": int(news[col].isna().sum())} for col in news]
        aligned = align_news(news, schedule)
        audit_dir = root / "data/processed/supplemental"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit = []
        for ticker, group in aligned.loc[~aligned.company_named_in_headline].groupby("ticker"):
            audit.append(group.sample(n=min(5, len(group)), random_state=42))
        if audit:
            pd.concat(audit).to_parquet(audit_dir / "headline_spotcheck.parquet", index=False)
        report["news_quality"] = {"rows": len(aligned), "invalid_timestamps": int(aligned.created_at.isna().sum()),
                                  "blank_headlines": int(aligned.blank_headline.sum()), "blank_links": int(aligned.blank_link.sum()),
                                  "suspected_exact_duplicate_rows": int(aligned.duplicate_exact_record.sum()),
                                  "exact_duplicate_copies_excluded": int(aligned.exclude_duplicate_copy.sum()),
                                  "midnight_utc_timestamp_rows": int(aligned.midnight_utc_timestamp.sum()),
                                  "same_headline_time_across_links_rows": int(aligned.same_headline_time_across_links.sum()),
                                  "unique_urls": int(aligned.loc[~aligned.blank_link, "link"].nunique()),
                                  "rows_with_company_named_in_headline": int(aligned.company_named_in_headline.sum()),
                                  "unassigned_close_rows": int(aligned.cutoff_date.isna().sum())}
        report["limitations"].extend([
            "EODHD demo sample is for local evaluation; publication/redistribution rights require separate academic confirmation.",
            "Provider publication timestamps do not prove the retrieved headline or sentiment existed then; revision metadata are unavailable.",
            "Ticker-linked coverage can include broad-market articles. Company-name matching is a relevance audit, not a definitive filter.",
            "Zero retrieved articles does not establish that no news existed; provider coverage is not independently verified.",
            "Provider sentiment is retained as metadata only; it is not used as a historical model feature."])
        report["limitations"].extend([
            "Exact midnight UTC timestamps are flagged as potentially date-only; they are excluded from cutoff-window pairing, not from monthly coverage counts.",
            "Repeated normalized headline/time across URLs may reflect syndication; flagged counts do not establish independent stories."])
        usable = aligned.loc[aligned.created_at.notna() & ~aligned.blank_headline & ~aligned.exclude_duplicate_copy]
        # Infer only interior complete-close windows, never call those a complete news corpus.
        if len(usable):
            lengths = usable.assign(headline_words=usable.headline.str.split().str.len())
            report["headline_word_counts"] = []
            fig, axes = plt.subplots(1, lengths.ticker.nunique(), figsize=(10, 4), sharex=True, sharey=True, squeeze=False)
            bins = np.arange(-.5, int(lengths.headline_words.max()) + 1.5, 1)
            for ax, (ticker, group) in zip(axes.flat, lengths.groupby("ticker")):
                ax.hist(group.headline_words, bins=bins, color=COLORS.get(ticker), alpha=.85)
                median = float(group.headline_words.median())
                ax.axvline(median, color="#444", lw=1, linestyle="--", label=f"Median: {median:g} words")
                ax.set(title=ticker, xlabel="Headline length (whitespace-separated words)")
                ax.legend(frameon=False)
                report["headline_word_counts"].append({"ticker": ticker, "records": len(group), "median_words": median,
                                                       "p95_words": float(group.headline_words.quantile(.95)),
                                                       "min_words": int(group.headline_words.min()), "max_words": int(group.headline_words.max())})
            axes[0, 0].set_ylabel("Ticker-news records (count)")
            fig.suptitle("Headlines are short, but length alone does not establish relevance", x=.07, ha="left")
            fig.text(.07, .005, "Same bins and scales; all observed lengths included. Source: local EODHD demo sample.", fontsize=8, color="#59636d")
            fig.tight_layout(rect=(0, .05, 1, .93))
            save(fig, "07_headline_lengths")
            periods = usable.created_at.dt.tz_localize(None).dt.to_period("M")
            monthly = usable.groupby([usable.ticker, periods]).size().unstack(0, fill_value=0).sort_index()
            report["monthly_news_counts"] = monthly.reset_index(names="month").assign(month=lambda d: d.month.astype(str)).to_dict("records")
            report["news_source_counts"] = (usable.groupby(["ticker", "source_domain"], dropna=False).size().reset_index(name="records").to_dict("records") if "source_domain" in usable else [])
            fig, ax = plt.subplots(figsize=(10, 4.2))
            monthly.index = monthly.index.astype(str)
            monthly.plot.bar(ax=ax, color=[COLORS.get(s) for s in monthly], width=.8)
            ax.set(title="Retrieved news falls sharply late in the evaluation year", xlabel="Month (provider timestamp, UTC)", ylabel="Retained ticker-news records (count)")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title=None, frameon=False)
            fig.text(.12, .005, "Source: EODHD public demo. A fall in retrieved records may reflect archive coverage; completeness is unverified.", fontsize=8, color="#59636d")
            fig.tight_layout(rect=(0, .04, 1, 1))
            save(fig, "05_monthly_news")
            news_summary = []
            paired = []
            for ticker, group in usable.groupby("ticker"):
                timed = group.loc[~group.midnight_utc_timestamp]
                if timed.empty:
                    continue
                lo, hi = timed.created_at.min(), timed.created_at.max()
                eligible = schedule.index[(schedule.market_close.shift(1).ge(lo) & schedule.market_close.le(hi)).to_numpy()]
                n = timed.groupby("cutoff_date").size().reindex(eligible, fill_value=0)
                news_summary.append({"ticker": ticker, "records": len(group), "first_provider_time": lo.isoformat(),
                                     "last_provider_time": hi.isoformat(), "interior_close_windows": len(n),
                                     "midnight_utc_rows_excluded_from_pairing": int(group.midnight_utc_timestamp.sum()),
                                     "windows_with_zero_retrieved_news": int(n.eq(0).sum()),
                                     "mean_news_per_interior_window": round(float(n.mean()), 3) if len(n) else None,
                                     "headline_company_name_pct": round(float(group.company_named_in_headline.mean() * 100), 2)})
                prices_for_ticker = daily.loc[daily.symbol.eq(ticker)].set_index("date")
                match = prices_for_ticker.join(n.rename("news_count"), how="inner")
                match["ticker"] = ticker
                paired.append(match.reset_index())
            report["news_by_stock"] = news_summary
            if paired:
                paired = pd.concat(paired, ignore_index=True)
                fig, axes = plt.subplots(1, len(news_summary), figsize=(10, 4.2), sharex=True, sharey=True, squeeze=False)
                correlations = []
                for ax, (ticker, group) in zip(axes.flat, paired.groupby("ticker")):
                    available = group.dropna(subset=["news_count", "next_return_pct"])
                    ax.scatter(available.news_count, available.next_return_pct, color=COLORS.get(ticker), s=14, alpha=.5)
                    ax.axhline(0, color="#888", lw=.8)
                    ax.set(title=ticker, xlabel="News excluding midnight UTC records (count)", ylabel="Next-session adjusted return (%)")
                    correlations.append({"ticker": ticker, "paired_stock_days": len(available),
                                         "pearson_news_count_next_return": round(float(available.news_count.corr(available.next_return_pct)), 5) if len(available)>2 else None})
                fig.suptitle("Explore news availability alongside the next session's return", x=.07, ha="left")
                fig.text(.07, .005, "Exploratory association only; midnight UTC records excluded. Article count does not test whether NLP improves prediction.", fontsize=8, color="#59636d")
                fig.tight_layout(rect=(0, .05, 1, .93))
                save(fig, "06_news_and_next_return")
                report["news_price_pairing"] = correlations
                processed = root / "data/processed/supplemental"
                processed.mkdir(parents=True, exist_ok=True)
                paired.to_parquet(processed / "paired_stock_days.parquet", index=False)
            report["status"] = "partial_descriptive_evaluation" if (price_gaps or news_collection_partial) else "descriptive_evaluation_completed"
    report_path = results / "eda_supplemental_summary.json"
    report = _json_finite(report)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {"report": report, "price_quality": quality, "daily_prices": daily, "aligned_news": aligned,
            "figure_paths": [root / p for p in report["figures"]]}

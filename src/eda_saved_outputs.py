"""Preliminary plots from tables already saved in collect_data.ipynb.

This module never calls an API and does not substitute notebook outputs for raw
news or price observations. Its figures describe the saved summaries only.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"AAPL": "#2367a4", "JNJ": "#c87525", "SPY": "#687582"}


def _saved_text(cells: list[dict], source_fragment: str) -> str:
    matches = [c for c in cells if source_fragment in "".join(c.get("source", []))]
    if len(matches) != 1:
        raise ValueError(f"Expected one source cell containing {source_fragment!r}")
    texts = [
        "".join(o.get("data", {}).get("text/plain", []))
        for o in matches[0].get("outputs", [])
        if "text/plain" in o.get("data", {})
    ]
    if len(texts) != 1:
        raise ValueError("The collection notebook does not contain the expected saved table.")
    return texts[0]


def read_saved_summaries(project_root: str | Path) -> dict:
    root = Path(project_root)
    notebook = root / "src" / "collect_data.ipynb"
    cells = json.loads(notebook.read_text())["cells"]

    total_text = _saved_text(cells, 'news_all.groupby("ticker").size()')
    totals = {
        ticker: int(count)
        for ticker, count in re.findall(r"^(\w+)\s+(\d+)\s*$", total_text, re.M)
    }
    if not totals:
        raise ValueError("No ticker counts found in the saved output.")

    annual_text = _saved_text(cells, 'groupby(["ticker", "year"])')
    lines = annual_text.splitlines()
    years = [int(y) for y in re.findall(r"\b20\d{2}\b", lines[0])]
    annual_rows = []
    for line in lines[1:]:
        parts = line.split()
        if parts and parts[0] in totals:
            if len(parts) != len(years) + 1:
                raise ValueError("Saved year table has an unexpected shape.")
            annual_rows.extend(
                {"ticker": parts[0], "year": year, "news_records": int(count)}
                for year, count in zip(years, parts[1:])
            )
    annual = pd.DataFrame(annual_rows)
    if annual.groupby("ticker")["news_records"].sum().to_dict() != totals:
        raise ValueError("Annual totals do not match the saved overall counts.")

    price_text = _saved_text(cells, 'prices.groupby("symbol")["timestamp"]')
    coverage_rows = []
    for line in price_text.splitlines():
        parts = line.split()
        if len(parts) == 6 and parts[0] in [*totals, "SPY"]:
            coverage_rows.append(
                {
                    "symbol": parts[0],
                    "first_saved_timestamp_utc": " ".join(parts[1:3]),
                    "last_saved_timestamp_utc": " ".join(parts[3:5]),
                    "price_rows": int(parts[5]),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    if coverage.empty:
        raise ValueError("No saved price coverage rows found.")

    counts = pd.Series(totals, name="news_records").sort_index().to_frame()
    counts["share_pct"] = counts["news_records"] / counts["news_records"].sum() * 100
    return {"totals": counts, "annual": annual, "price_coverage": coverage}


def run_saved_output_eda(project_root: str | Path) -> dict:
    root = Path(project_root)
    tables = read_saved_summaries(root)
    figures = root / "figures"
    results = root / "results"
    figures.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
         "figure.facecolor": "white", "savefig.facecolor": "white"}
    )
    counts = tables["totals"]
    total = int(counts["news_records"].sum())
    share = float(counts.loc["AAPL", "share_pct"])
    ratio = float(counts.loc["AAPL", "news_records"] / counts.loc["JNJ", "news_records"])

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.barh(counts.index[::-1], counts["news_records"].iloc[::-1],
                   color=[COLORS[t] for t in counts.index[::-1]], height=0.5)
    ax.set_xlim(0, counts["news_records"].max() * 1.28)
    ax.set_xlabel("Saved ticker-linked news records (count)")
    ax.set_title(f"AAPL accounts for {share:.1f}% of the saved news records", loc="left", pad=16)
    for bar, ticker in zip(bars, counts.index[::-1]):
        value = int(counts.loc[ticker, "news_records"])
        pct = float(counts.loc[ticker, "share_pct"])
        ax.text(value + 350, bar.get_y() + bar.get_height() / 2,
                f"{value:,}  ({pct:.1f}%)", va="center", fontsize=11)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.text(0.12, 0.015, "Source: saved notebook totals; unique articles and raw-file quality have not been checked.",
             fontsize=9, color="#59636d")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    concentration = figures / "eda_saved_01_news_counts.png"
    fig.savefig(concentration, dpi=170, bbox_inches="tight")
    plt.close(fig)

    pivot = tables["annual"].pivot(index="year", columns="ticker", values="news_records")
    years = pivot.index.tolist()
    x = np.arange(len(years))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, ticker in zip(axes, counts.index):
        bars = ax.bar(x, pivot[ticker], color=COLORS[ticker], width=0.7)
        bars[-1].set_hatch("///")
        bars[-1].set_alpha(0.55)
        ax.set_xticks(x, [str(y) if i < len(years) - 1 else f"{y}*" for i, y in enumerate(years)])
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(ticker, loc="left", color=COLORS[ticker], fontweight="bold")
        ax.set_xlabel("Calendar year")
        ax.set_ylabel("Saved ticker-linked news records (count)")
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.set_ylim(0, float(pivot[ticker].max()) * 1.19)
        for bar, value in zip(bars, pivot[ticker]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Saved news coverage varies by stock and year", x=0.06, ha="left", fontsize=15)
    fig.text(0.06, 0.02, "*2026 is partial: the collection request ended October 16. Panels use different y-axis scales.\n"
             "Source: saved notebook year table; counts do not establish complete API coverage.",
             fontsize=9, color="#59636d")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    annual_plot = figures / "eda_saved_02_news_by_year.png"
    fig.savefig(annual_plot, dpi=170, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "analysis_basis": "saved_notebook_summaries_only",
        "source_notebook": "src/collect_data.ipynb",
        "source_notebook_sha256": hashlib.sha256((root / "src" / "collect_data.ipynb").read_bytes()).hexdigest(),
        "notebook_executed_for_this_analysis": False,
        "raw_quality_verified": False,
        "total_ticker_news_records": total,
        "aapl_share_pct": share,
        "aapl_to_jnj_count_ratio": ratio,
        "ticker_totals": counts.reset_index(names="ticker").to_dict(orient="records"),
        "year_counts": tables["annual"].to_dict(orient="records"),
        "price_coverage": tables["price_coverage"].to_dict(orient="records"),
        "figures": [str(p.relative_to(root)) for p in [concentration, annual_plot]],
        "limitations": [
            "News rows are not established unique articles.",
            "Equal price row counts do not establish complete trading-day coverage.",
            "No raw-row missingness, duplicates, daily returns or news-price relationships were analyzed.",
            "2026 is a partial year; compare it with complete years only with appropriate exposure adjustment.",
        ],
    }
    (results / "eda_saved_summary.json").write_text(json.dumps(payload, indent=2))
    return {**tables, "metrics": payload, "figure_paths": [concentration, annual_plot]}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_saved_output_eda(root)
    print(json.dumps(result["metrics"], indent=2))

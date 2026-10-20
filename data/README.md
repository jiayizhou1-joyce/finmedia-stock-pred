# Data collection and EDA

The six Python scripts are in this folder. Run the commands below from the **project root**, where `requirements-eda.txt` is located.

These scripts build on Carmen's Alpaca collection notebook and Michael's earlier API exploration. Joyce added the collection scripts and EDA. The Yahoo/EODHD sample is stored separately from the original Alpaca data.

## 1. Set up Python

Activate the project environment if you already have one. Otherwise, create one and install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eda.txt
```

## 2. Get the Yahoo prices and EODHD headlines

These two collectors do not need private API keys. First inspect the local price cache and review the news request plan:

```bash
python data/collect_public_prices.py
python data/collect_public_news.py
```

Add `--collect` to download the data:

```bash
python data/collect_public_prices.py --collect
python data/collect_public_news.py --collect
```

| Script | Default sample | Dates: start included, end excluded |
| --- | --- | --- |
| `collect_public_prices.py` | Yahoo daily prices for AAPL, JNJ, SPY and AMZN | `2020-01-01` to `2026-10-16`, using New York session dates |
| `collect_public_news.py` | EODHD demo headlines for AAPL and AMZN | `2024-01-01` to `2025-01-01`, using UTC publication timestamps |

To choose a shorter period:

```bash
python data/collect_public_prices.py --collect --symbols AAPL AMZN SPY --start 2024-01-01 --end 2025-01-01
python data/collect_public_news.py --collect --tickers AAPL AMZN --start 2024-01-01 --end 2025-01-01
```

The Yahoo script currently accepts AAPL, JNJ, SPY and AMZN. The EODHD script accepts the demo symbols AAPL, AMZN, TSLA and VTI; provider access may change. The defaults reproduce the sample used for our EDA.

Main output files:

```text
data/supplemental/yahoo_prices/prices.parquet
data/supplemental/eodhd_news/news.parquet
data/supplemental/eodhd_news/collection_run.json
```

The collectors also save raw responses, request parameters, retrieval times and file hashes in the provider folders. Keep these files together. Matching, validated caches are reused. For a fresh Yahoo download, add `--refresh` alongside `--collect`.

Collection writes to these fixed output paths. Copy a sample elsewhere before requesting different dates if you want to keep both. The news collector can save a partial table when collection encounters problems: check `status` and `issues` in `collection_run.json` before using it. Completing the requests does not establish that the provider archive is complete.

## 3. Get the Alpaca data, if needed

If Carmen has already collected the original files, ask her for these three files and place them at the same paths locally:

```text
data/raw/news/AAPL.parquet
data/raw/news/JNJ.parquet
data/raw/prices.parquet
```

To collect them yourself, create a private `.env` file in the **project root**:

```dotenv
API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key
```

Review the request plan, then run collection:

```bash
python data/collect_pipeline.py
python data/collect_pipeline.py --collect
```

The default request covers `2020-01-01T00:00:00Z <= timestamp < 2026-10-16T00:00:00Z`. It collects AAPL/JNJ news and daily AAPL/JNJ/SPY prices, using the SIP feed and `Adjustment.ALL`. The account needs access to that feed. This command uses the settings in `CollectionConfig` inside `collect_pipeline.py`; it has no ticker or date command-line options.

The script saves the three Parquet files above, yearly news chunks, matching manifests and `data/raw/collection_run.json`. Credentials are read from the project's `.env` only when a new Alpaca request is needed.

## 4. Run the EDA

After the data is available locally:

```bash
python data/eda_saved_outputs.py
python data/supplemental_eda.py
python data/raw_eda.py
```

| Script | Input and result |
| --- | --- |
| `eda_saved_outputs.py` | Reads Carmen's saved summary tables in `src/collect_data.ipynb` and produces two news-coverage plots. The original raw files are not needed for this step. |
| `supplemental_eda.py` | Reads the Yahoo/EODHD tables and produces seven plots covering prices, returns, volatility, target balance, news coverage, headline length and news/return pairing. |
| `raw_eda.py` | Checks the original Alpaca files for missing values, duplicates, price coverage and news timing. Missing inputs are reported explicitly. |

For a shorter price-analysis period:

```bash
python data/supplemental_eda.py --start 2024-01-01 --end 2025-01-01
```

Figures are saved in `figures/`; EDA summaries are saved in `results/`. Supplemental audit samples and news/price pairings are saved in `data/processed/supplemental/`. Each EDA run replaces its own previous outputs. Missing inputs can lead to fewer figures; check the printed status and saved summary.

The EDA scripts read local files and make no API requests. You can also open `src/eda_joyce.ipynb` in VS Code to read the saved figures and explanations. That notebook is self-contained; rerunning all its cells requires the local supplemental data.

## If a command fails

- **Missing package:** activate the project environment and install `requirements-eda.txt`.
- **Missing data:** run the matching collector, or obtain the original Alpaca files from the team.
- **Alpaca authentication or feed error:** check the root `.env` and the account's SIP access.
- **HTTP 401, 403 or 429:** stop and check provider access or rate limits. Yahoo's public chart endpoint is undocumented, and demo access may be restricted.
- **Partial or empty news download:** inspect `collection_run.json`. An empty response does not prove that no news existed.

New downloads may differ from earlier retrievals because providers can revise their archives. This is collection and descriptive EDA; no model comparison is run by these scripts. Git tracks the scripts and this README, while downloads, credentials and environments remain local. Confirm provider terms before redistributing data.

# AIA

This repo exports **football (soccer) data** from the [`football-data.org`](https://www.football-data.org/) API into simple CSV files, then provides a small notebook to visually inspect / sanity-check the raw datasets.

By default it targets **competition `2014` (La Liga / Primera División)**.

## What you get

After running the exporter, you should have:
- `data/matches.csv`: match-level data (teams, score, matchday, dates, etc.)
- `data/standings.csv`: standings table rows per season (latest matchday table for that season)
- `data/teams.csv`: team metadata per season
- `data/scorers.csv`: top scorers per season
- `data/match_goals.csv` and `data/match_bookings.csv`: match events (may be empty depending on API access/plan)

API responses are cached in `data/_api_cache/` to reduce repeated calls.

## Quickstart

### Setup

Create & activate a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install requests pandas matplotlib
```

Set your API token (either in `.env` at repo root or as an environment variable):
- `FOOTBALL_DATA_API_TOKEN=...`

### Export the data

Run the one-shot exporter:

```bash
python3 code/parser.py
```

CSV files are written to `data/`.

## Explore / visualize the raw data

Open the notebook:
- `code/data_overview.ipynb`

It prints quick dataset summaries (shape, missing values) and a few simple plots (distributions / top categories).

## Configuration (optional)

The exporter is configurable via environment variables:
- `AIA_COMPETITION_ID` (default `2014`)
- `AIA_MIN_YEAR` (default `2020`)
- `AIA_MIN_SECONDS_BETWEEN_CALLS` (default `8.0`)
- `AIA_CACHE_DIR` (default `data/_api_cache`)
- `AIA_SCORERS_LIMIT` (default `50`)

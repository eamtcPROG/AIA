from __future__ import annotations

"""
Data analysis + simple graphics for ./data/raw_data.csv.

Outputs plots into ./reports/ (created if missing).

Dependencies (install if missing):
  python3 -m pip install pandas matplotlib
"""

import os
from pathlib import Path


# ---- Configuration (edit these, or override via env vars) ----
RAW_CSV = os.getenv("AIA_RAW_CSV", "data/raw_data.csv")
REPORTS_DIR = os.getenv("AIA_REPORTS_DIR", "reports")


def main() -> int:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: pandas. Install with: pip install pandas") from e

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency: matplotlib. Install with: pip install matplotlib"
        ) from e

    root = Path(__file__).resolve().parent.parent
    csv_path = (root / RAW_CSV).resolve()
    reports_dir = (root / REPORTS_DIR).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}. Run the exporter first (python3 code/main.py).")

    df = pd.read_csv(csv_path)

    # Parse dates
    if "utcDate" in df.columns:
        df["utcDate"] = pd.to_datetime(df["utcDate"], errors="coerce", utc=True)
        df["date"] = df["utcDate"].dt.date
        df["year"] = df["utcDate"].dt.year
        df["month"] = df["utcDate"].dt.to_period("M").astype(str)

    # Goals (may be NaN if match not finished / or plan doesn't unfold details)
    for col in ["score_fullTime_home", "score_fullTime_away"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"score_fullTime_home", "score_fullTime_away"} <= set(df.columns):
        df["goals_total"] = df["score_fullTime_home"].fillna(0) + df["score_fullTime_away"].fillna(0)

    # ---- Plot 1: matches per month ----
    if "month" in df.columns:
        m = df.dropna(subset=["month"]).groupby("month")["match_id"].nunique().sort_index()
        plt.figure(figsize=(12, 4))
        m.plot(kind="bar")
        plt.title("Matches per month")
        plt.xlabel("Month")
        plt.ylabel("Matches")
        plt.tight_layout()
        out = reports_dir / "matches_per_month.png"
        plt.savefig(out, dpi=160)
        plt.close()

    # ---- Plot 2: goals distribution (finished games only, best-effort) ----
    if "goals_total" in df.columns:
        g = df["goals_total"].dropna()
        plt.figure(figsize=(8, 4))
        plt.hist(g, bins=range(0, int(g.max()) + 2 if len(g) else 2), edgecolor="black")
        plt.title("Goals per match (full time)")
        plt.xlabel("Goals")
        plt.ylabel("Matches")
        plt.tight_layout()
        out = reports_dir / "goals_per_match_hist.png"
        plt.savefig(out, dpi=160)
        plt.close()

    # ---- Plot 3: top teams by match count ----
    if {"homeTeam_name", "awayTeam_name"} <= set(df.columns):
        home = df["homeTeam_name"].dropna()
        away = df["awayTeam_name"].dropna()
        team_counts = (
            pd.concat([home, away])
            .value_counts()
            .head(15)
            .sort_values(ascending=True)
        )
        plt.figure(figsize=(10, 6))
        team_counts.plot(kind="barh")
        plt.title("Top 15 teams by appearances")
        plt.xlabel("Matches")
        plt.tight_layout()
        out = reports_dir / "top_teams_by_matches.png"
        plt.savefig(out, dpi=160)
        plt.close()

    # Small console summary
    print(f"Loaded: {csv_path}")
    print(f"Rows: {len(df):,}")
    if "utcDate" in df.columns:
        print("Date range:", df["utcDate"].min(), "->", df["utcDate"].max())
    print(f"Wrote plots to: {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


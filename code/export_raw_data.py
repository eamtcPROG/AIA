from __future__ import annotations

import csv
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from football_data_api import (
    DEFAULT_TOKEN_ENV,
    FootballDataClient,
    RateLimitConfig,
    find_repo_root,
    load_dotenv_simple,
)


def _s(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def flatten_match(match: Dict[str, Any], *, competition: Dict[str, Any]) -> Dict[str, Any]:
    score = match.get("score") or {}
    return {
        "competition_id": _s(competition, "id"),
        "competition_code": _s(competition, "code"),
        "competition_name": _s(competition, "name"),
        "area_name": _s(competition, "area", "name"),
        "match_id": match.get("id"),
        "utcDate": match.get("utcDate"),
        "status": match.get("status"),
        "matchday": match.get("matchday"),
        "stage": match.get("stage"),
        "group": match.get("group"),
        "season_startDate": _s(match, "season", "startDate"),
        "season_endDate": _s(match, "season", "endDate"),
        "homeTeam_id": _s(match, "homeTeam", "id"),
        "homeTeam_name": _s(match, "homeTeam", "name"),
        "awayTeam_id": _s(match, "awayTeam", "id"),
        "awayTeam_name": _s(match, "awayTeam", "name"),
        "score_winner": score.get("winner"),
        "score_duration": score.get("duration"),
        "score_fullTime_home": _s(score, "fullTime", "home"),
        "score_fullTime_away": _s(score, "fullTime", "away"),
        "referees_json": json.dumps(match.get("referees") or [], ensure_ascii=False),
        "raw_json": json.dumps(match, ensure_ascii=False, separators=(",", ":")),
    }


def season_years(comp_payload: dict, *, min_year: int) -> list[int]:
    years: set[int] = set()
    for s in comp_payload.get("seasons") or []:
        start = (s or {}).get("startDate")  # yyyy-mm-dd
        if isinstance(start, str) and len(start) >= 4 and start[:4].isdigit():
            y = int(start[:4])
            if y >= min_year:
                years.add(y)
    if not years:
        years = set(range(min_year, date.today().year + 1))
    return sorted(years)


def run_export(
    *,
    competition_id: int,
    min_year: int = 2020,
    out: str = "data/raw_data.csv",
    cache_dir: str = "data/_api_cache",
    min_seconds_between_calls: float = 8.0,
) -> tuple[Path, int]:
    root = find_repo_root()
    load_dotenv_simple(start_dir=root)

    token = (os.getenv(DEFAULT_TOKEN_ENV) or "").strip()
    if not token:
        raise SystemExit(f"Missing {DEFAULT_TOKEN_ENV}. Put it in .env or export it.")

    client = FootballDataClient(
        token=token,
        cache_dir=(root / cache_dir),
        rate_limit=RateLimitConfig(min_seconds_between_calls=min_seconds_between_calls),
    )

    competition = client.get(f"/competitions/{competition_id}")
    years = season_years(competition, min_year=min_year)

    rows: list[dict] = []
    for y in years:
        try:
            payload = client.get(f"/competitions/{competition_id}/matches", params={"season": y})
        except requests.HTTPError as e:
            # Common case: 403 due to subscription limits for older seasons
            resp = getattr(e, "response", None)
            if getattr(resp, "status_code", None) == 403:
                continue
            raise
        matches = payload.get("matches") or []
        rows.extend(flatten_match(m, competition=competition) for m in matches)

    out_path = root / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return out_path, len(rows)


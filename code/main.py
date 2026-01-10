from __future__ import annotations

import os
import sys
from pathlib import Path


# ---- Configuration (edit these, or override via env vars) ----
# Competition to export (default: LaLiga / Primera Division)
COMPETITION_ID = int(os.getenv("AIA_COMPETITION_ID", "2014"))
# Try seasons starting from this year; older seasons may be blocked by subscription
MIN_YEAR = int(os.getenv("AIA_MIN_YEAR", "2020"))
# Output CSV path (relative to repo root)
OUT = os.getenv("AIA_OUT", "data/raw_data.csv")
# Cache directory (relative to repo root)
CACHE_DIR = os.getenv("AIA_CACHE_DIR", "data/_api_cache")
# Throttle to avoid 429 rate limits
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("AIA_MIN_SECONDS_BETWEEN_CALLS", "8.0"))


def main() -> int:
    # Ensure ./code is importable when running from repo root
    code_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(code_dir))

    from export_raw_data import run_export  # type: ignore

    out_path, row_count = run_export(
        competition_id=COMPETITION_ID,
        min_year=MIN_YEAR,
        out=OUT,
        cache_dir=CACHE_DIR,
        min_seconds_between_calls=MIN_SECONDS_BETWEEN_CALLS,
    )

    print(f"Wrote {row_count} match rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests


DEFAULT_BASE_URL = "https://api.football-data.org/v4"
DEFAULT_TOKEN_ENV = "FOOTBALL_DATA_API_TOKEN"


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Best-effort repo root discovery (looks for README.md)."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "README.md").exists():
            return p
    return here


def load_dotenv_simple(*, start_dir: Optional[Path] = None, override: bool = False) -> Optional[Path]:
    """Load key/value pairs from a local .env file into os.environ (no deps)."""
    start_dir = start_dir or Path.cwd()
    candidates = [
        start_dir / ".env",
        start_dir.parent / ".env",
        (find_repo_root(start_dir) / ".env"),
    ]
    env_path = next((p for p in candidates if p.exists()), None)
    if not env_path:
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        if override or k not in os.environ:
            os.environ[k] = v
    return env_path


@dataclass
class RateLimitConfig:
    min_seconds_between_calls: float = 8.0
    max_retries_429: int = 6


class FootballDataClient:
    def __init__(
        self,
        *,
        token: str,
        base_url: str = DEFAULT_BASE_URL,
        unfold_goals: bool = True,
        cache_dir: Optional[Path] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.rate_limit = rate_limit
        self._last_call_ts = 0.0

        self.session = requests.Session()
        headers = {"X-Auth-Token": token}
        if unfold_goals:
            headers["X-Unfold-Goals"] = "true"
        self.session.headers.update(headers)

        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, path: str, params: Optional[Dict[str, Any]]) -> str:
        if not params:
            return path
        items = sorted((str(k), str(v)) for k, v in params.items())
        return f"{path}?{urlencode(items)}"

    def _cache_paths(self, path: str, params: Optional[Dict[str, Any]]) -> Tuple[str, Optional[Path]]:
        key = self._cache_key(path, params)
        if not self.cache_dir:
            return key, None
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return key, (self.cache_dir / f"{h}.json")

    def get(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        key, cache_path = self._cache_paths(path, params)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        if self.rate_limit:
            now = time.time()
            wait = self.rate_limit.min_seconds_between_calls - (now - self._last_call_ts)
            if wait > 0:
                time.sleep(wait)

        url = f"{self.base_url}{path}"

        for attempt in range((self.rate_limit.max_retries_429 if self.rate_limit else 0) + 1):
            r = self.session.get(url, params=params, timeout=self.timeout_seconds)
            if r.status_code == 429 and self.rate_limit:
                retry_after = r.headers.get("Retry-After")
                try:
                    sleep_s = float(retry_after) if retry_after else (self.rate_limit.min_seconds_between_calls * (attempt + 1))
                except Exception:
                    sleep_s = self.rate_limit.min_seconds_between_calls * (attempt + 1)
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            data = r.json()
            if cache_path:
                cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            if self.rate_limit:
                self._last_call_ts = time.time()
            return data

        raise RuntimeError(f"Failed after retries: {key}")


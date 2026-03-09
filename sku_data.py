"""
AMAZON ADS — SKU-WISE DATA FETCHER  (Robust Edition)
======================================================
Robustness improvements over the previous version:

  1. True token-bucket rate limiter  — caps calls/second globally across threads
  2. Retry-After header respected    — waits exactly what Amazon asks, not a guess
  3. Exponential backoff + full jitter — prevents thundering-herd after 429 bursts
  4. Two-phase pipeline              — submit ALL reports first, then poll in batch,
                                       so one slow report never blocks the others
  5. Global concurrency semaphore    — hard cap on simultaneous in-flight API calls
  6. requests.Session + connection pool — reuses TCP sockets, cuts latency
  7. Partial-result return           — if some reports fail the rest still come back
  8. Token refresh is non-blocking   — other threads queue on lock, not on HTTP

Usage (standalone):
    python sku_data.py                         # last 10 days, both markets
    python sku_data.py --days 7 --market US    # last 7 days, US only
    python sku_data.py --days 3 --workers 2    # slower but gentler on rate limits

Usage (as module from app.py):
    from sku_data import fetch_sku_ads_data, fetch_sku_ads_summary
    df = fetch_sku_ads_summary("2025-01-01", "2025-01-10", market="US")
"""

import gzip
import io
import json
import logging
import os
import random
import sys
import threading
import time
import argparse
import concurrent.futures
from dataclasses import dataclass, field
from datetime import date, timedelta, datetime
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("sku_data")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

# ==============================================================================
# CONFIGURATION  — credentials come from environment variables, never hardcoded.
#
#   Set these in:
#     • Local dev    →  a .env file (loaded below) or export in your shell
#     • GitHub CI    →  repo Settings → Secrets and variables → Actions
#     • Streamlit    →  app Settings → Secrets  (st.secrets["AMZN_CLIENT_ID"] etc.)
# ==============================================================================

# Load a local .env file when running on a developer machine.
# On CI / Streamlit Cloud the env vars are injected by the platform directly,
# so python-dotenv is optional — we silently skip if it isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Profile IDs (not secret — hardcoded defaults, overridable via env)
PROFILES: dict = {
    "US": {
        "profile_id": os.environ.get("AMZN_PROFILE_ID_US", "1738642012820077"),
        "name": "ANGARA - US",
    },
    "CA": {
        "profile_id": os.environ.get("AMZN_PROFILE_ID_CA", "2297668671987824"),
        "name": "ANGARA - Canada",
    },
}

# Credentials are loaded LAZILY (only when first API call is made).
# This means importing sku_data in app.py never crashes even if the
# Streamlit Cloud secrets haven't been set yet — the error only appears
# when the user actually clicks "Fetch".
_CLIENT_ID:     Optional[str] = None
_CLIENT_SECRET: Optional[str] = None
_REFRESH_TOKEN: Optional[str] = None


def _load_credentials():
    """
    Load credentials from environment on first use.
    Checks os.environ first (GitHub Actions, .env file, shell exports),
    then falls back to st.secrets if running inside Streamlit Cloud.
    """
    global _CLIENT_ID, _CLIENT_SECRET, _REFRESH_TOKEN

    if _CLIENT_ID:          # already loaded
        return

    def _get(key: str) -> str:
        # 1. Try plain environment variable (GitHub Actions / .env / shell)
        val = os.environ.get(key, "").strip()
        if val:
            return val

        # 2. Try Streamlit secrets (only available when running in Streamlit)
        try:
            import streamlit as st
            val = st.secrets.get(key, "").strip()
            if val:
                return val
        except Exception:
            pass

        raise EnvironmentError(
            f"Required secret '{key}' is not set.\n"
            "  • Local dev      → add to .env file in project root\n"
            "  • GitHub Actions → repo Settings → Secrets → Actions\n"
            "  • Streamlit Cloud→ app Settings → Secrets (TOML format):\n"
            f'      {key} = "your-value-here"'
        )

    _CLIENT_ID     = _get("AMZN_CLIENT_ID")
    _CLIENT_SECRET = _get("AMZN_CLIENT_SECRET")
    _REFRESH_TOKEN = _get("AMZN_REFRESH_TOKEN")
    log.debug("Credentials loaded OK")


def _get_client_id()     -> str: _load_credentials(); return _CLIENT_ID      # type: ignore
def _get_client_secret() -> str: _load_credentials(); return _CLIENT_SECRET  # type: ignore
def _get_refresh_token() -> str: _load_credentials(); return _REFRESH_TOKEN  # type: ignore

# ── concurrency / rate-limiting ───────────────────────────────────────────────
#  Amazon Ads allows roughly 2 report-create calls/sec per profile.
#  We set 1.5/s globally (both profiles combined) to stay well inside the limit.
MAX_CALLS_PER_SEC  = 1.5    # token-bucket refill rate
MAX_BURST          = 3      # token-bucket capacity
MAX_CONCURRENT_API = 4      # hard ceiling on simultaneous HTTP calls (semaphore)

# ── retry / backoff ───────────────────────────────────────────────────────────
MAX_SUBMIT_RETRIES   = 7
MAX_POLL_RETRIES     = 3    # retries for network errors only while polling
MAX_DOWNLOAD_RETRIES = 4
BACKOFF_BASE         = 5    # seconds
BACKOFF_CAP          = 120  # never wait longer than 2 min per retry
JITTER_FRACTION      = 0.3  # +-30% random jitter on each backoff

# ── poll settings ─────────────────────────────────────────────────────────────
POLL_INTERVAL_SEC  = 5      # wait between polls for one report
MAX_POLL_WAIT_SEC  = 900    # 15-min hard ceiling per report

# ── inter-submission pacing ───────────────────────────────────────────────────
SUBMIT_BATCH_DELAY = 1.0    # seconds between consecutive submissions

# ── HTTP timeouts ─────────────────────────────────────────────────────────────
T_CREATE   = 30
T_POLL     = 20
T_DOWNLOAD = 180
T_TOKEN    = 30


# ==============================================================================
# SHARED HTTP SESSION  (connection pool + urllib3 retries for transient errors)
# ==============================================================================

def _build_session() -> requests.Session:
    s = requests.Session()
    # urllib3-level retries handle TCP resets / 5xx only.
    # 429 is handled at application level so we can honour Retry-After.
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=20,
    )
    s.mount("https://", adapter)
    return s


_session = _build_session()


# ==============================================================================
# TOKEN-BUCKET RATE LIMITER
# ==============================================================================

class _TokenBucket:
    """Thread-safe token-bucket: blocks callers until a token is available."""

    def __init__(self, rate: float, capacity: float):
        self._rate     = rate
        self._capacity = capacity
        self._tokens   = capacity
        self._ts       = time.monotonic()
        self._lock     = threading.Lock()

    def acquire(self, tokens: float = 1.0):
        with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._ts) * self._rate)
                self._ts = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # sleep the minimum time needed to refill
                time.sleep((tokens - self._tokens) / self._rate)


_bucket  = _TokenBucket(MAX_CALLS_PER_SEC, MAX_BURST)
_api_sem = threading.Semaphore(MAX_CONCURRENT_API)


# ==============================================================================
# GLOBAL 429 COORDINATOR
# ==============================================================================
# When ANY thread gets a 429 it tells ALL threads to pause until the
# Retry-After window expires, preventing a cascade of further 429s.

_rl_event = threading.Event()
_rl_event.set()           # set = "clear to send"
_rl_until = 0.0           # wall-clock epoch when we may resume
_rl_lock  = threading.Lock()


def _signal_rate_limit(wait_sec: float):
    with _rl_lock:
        global _rl_until
        resume = time.time() + wait_sec
        if resume > _rl_until:
            _rl_until = resume
            _rl_event.clear()
            log.warning("⏸  429 — all threads pausing %.1f s (Retry-After)", wait_sec)


def _wait_for_rate_limit():
    """Block until any active rate-limit cooldown has passed."""
    _rl_event.wait()
    # Double-check: another thread may have extended the window
    with _rl_lock:
        remaining = _rl_until - time.time()
    if remaining > 0:
        _rl_event.clear()
        time.sleep(remaining)
        _rl_event.set()


def _parse_retry_after(resp: requests.Response) -> float:
    ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    return 30.0  # safe default when header absent


# ==============================================================================
# BACKOFF HELPER
# ==============================================================================

def _backoff(attempt: int) -> float:
    """Full-jitter exponential backoff, capped at BACKOFF_CAP seconds."""
    ceiling = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt))
    jitter  = ceiling * JITTER_FRACTION * (2 * random.random() - 1)
    return max(0.5, ceiling + jitter)


# ==============================================================================
# TOKEN MANAGER
# ==============================================================================

class _TokenManager:
    _LIFETIME_SEC = 3000  # refresh 10 min before the 60-min expiry

    def __init__(self):
        self._token:      Optional[str]      = None
        self._fetched_at: Optional[datetime] = None
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if self._needs_refresh():
                self._refresh()
            return self._token  # type: ignore[return-value]

    def _needs_refresh(self) -> bool:
        if not self._token or not self._fetched_at:
            return True
        return (datetime.now() - self._fetched_at).total_seconds() > self._LIFETIME_SEC

    def _refresh(self):
        for attempt in range(4):
            try:
                r = _session.post(
                    "https://api.amazon.com/auth/o2/token",
                    data={
                        "grant_type":    "refresh_token",
                        "refresh_token": _get_refresh_token(),
                        "client_id":     _get_client_id(),
                        "client_secret": _get_client_secret(),
                    },
                    timeout=T_TOKEN,
                )
                r.raise_for_status()
                self._token      = r.json()["access_token"]
                self._fetched_at = datetime.now()
                log.debug("Token refreshed OK")
                return
            except Exception as exc:
                wait = _backoff(attempt)
                log.warning("Token refresh attempt %d failed: %s — retry in %.1f s", attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError("Token refresh failed after 4 attempts — check credentials.")


_tok = _TokenManager()


def _headers(profile_id: str) -> dict:
    try:
        token = _tok.get()
    except Exception as e:
        print(f"[FATAL] Token refresh failed: {e}")
        raise
    return {
        "Authorization":                   f"Bearer {token}",
        "Amazon-Advertising-API-ClientId": _get_client_id(),
        "Amazon-Advertising-API-Scope":    profile_id,
        "Content-Type":                    "application/json",
    }


# ==============================================================================
# CORE API CALL WRAPPER
# ==============================================================================

def _api_call(method: str, url: str, profile_id: str, **kwargs) -> requests.Response:
    """
    One HTTP call with:
      - global 429 cooldown check
      - token-bucket pacing
      - concurrency semaphore
      - automatic 401 token refresh
    Does NOT retry — let callers decide retry policy per call type.
    """
    _wait_for_rate_limit()
    _bucket.acquire()

    with _api_sem:
        hdrs = _headers(profile_id)
        kwargs.setdefault("headers", hdrs)
        resp = _session.request(method, url, **kwargs)

        if resp.status_code == 401:
            log.debug("401 on %s — refreshing token and retrying once", url)
            with _tok._lock:
                _tok._token = None   # force refresh on next get()
            kwargs["headers"] = _headers(profile_id)
            resp = _session.request(method, url, **kwargs)

        return resp


# ==============================================================================
# REPORT SUBMISSION  (with full retry logic)
# ==============================================================================

def _submit_report(profile_id: str, payload: dict) -> Optional[str]:
    """Returns report_id string, or None on permanent failure."""
    for attempt in range(MAX_SUBMIT_RETRIES):
        try:
            resp = _api_call(
                "POST",
                "https://advertising-api.amazon.com/reporting/reports",
                profile_id,
                json=payload,
                timeout=T_CREATE,
            )
        except Exception as exc:
            wait = _backoff(attempt)
            log.warning("Submit network error attempt %d: %s — retry in %.1f s", attempt + 1, exc, wait)
            print(f"[ERROR] Submit attempt {attempt+1} exception: {exc}")
            time.sleep(wait)
            continue

        code = resp.status_code

        if code in (200, 202, 425):
            rid = resp.json().get("reportId")
            log.debug("Report submitted: %s", rid)
            return rid

        if code == 429:
            # Do NOT consume an attempt — just honour the cooldown and loop
            _signal_rate_limit(_parse_retry_after(resp))
            _wait_for_rate_limit()
            continue

        if code == 400:
            log.error("Bad request (400): %s", resp.text[:300])
            print(f"[ERROR] 400 Bad Request for {payload}: {resp.text[:300]}")
            return None   # payload error — retrying won't help

        # 5xx or unexpected
        wait = _backoff(attempt)
        log.warning("Submit HTTP %d attempt %d — retry in %.1f s", code, attempt + 1, wait)
        time.sleep(wait)

    log.error("Report submission gave up after %d attempts", MAX_SUBMIT_RETRIES)
    return None


# ==============================================================================
# POLLING  (checks status until COMPLETED / FAILED / timeout)
# ==============================================================================

def _poll_until_complete(profile_id: str, report_id: str) -> Optional[str]:
    """Returns download URL or None."""
    deadline        = time.time() + MAX_POLL_WAIT_SEC
    poll_url        = f"https://advertising-api.amazon.com/reporting/reports/{report_id}"
    net_err_streak  = 0

    while time.time() < deadline:
        try:
            resp = _api_call("GET", poll_url, profile_id, timeout=T_POLL)
        except Exception as exc:
            net_err_streak += 1
            wait = _backoff(net_err_streak, )
            log.warning("Poll network error for %s (streak %d): %s — %.1f s",
                        report_id, net_err_streak, exc, wait)
            if net_err_streak > MAX_POLL_RETRIES:
                log.error("Giving up poll on %s after %d consecutive errors", report_id, net_err_streak)
                return None
            time.sleep(wait)
            continue

        code = resp.status_code

        if code == 429:
            _signal_rate_limit(_parse_retry_after(resp))
            _wait_for_rate_limit()
            continue

        if code in (500, 502, 503, 504):
            time.sleep(POLL_INTERVAL_SEC * 2)
            continue

        if code != 200:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        # Successful poll response
        net_err_streak = 0
        data   = resp.json()
        status = data.get("status", "")

        if status == "COMPLETED":
            url = data.get("url") or data.get("location")
            log.debug("Report %s COMPLETED", report_id)
            return url

        if status == "FAILED":
            log.warning("Report %s FAILED: %s", report_id, data.get("statusDetails", ""))
            return None

        # PENDING or IN_PROGRESS — wait and try again
        time.sleep(POLL_INTERVAL_SEC)

    log.error("Report %s timed out after %d s", report_id, MAX_POLL_WAIT_SEC)
    return None


# ==============================================================================
# DOWNLOAD
# ==============================================================================

def _download(url: str) -> Optional[list]:
    """Download and parse a gzip-JSON report from S3. Returns list or None."""
    for attempt in range(MAX_DOWNLOAD_RETRIES):
        try:
            resp = _session.get(url, timeout=T_DOWNLOAD)
            resp.raise_for_status()
            with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as fh:
                return json.load(fh)
        except Exception as exc:
            wait = _backoff(attempt, )
            log.warning("Download error attempt %d: %s — retry in %.1f s", attempt + 1, exc, wait)
            time.sleep(wait)
    log.error("Download failed after %d attempts", MAX_DOWNLOAD_RETRIES)
    return None


# ==============================================================================
# SKU REPORT CONFIG & ROW NORMALISATION
# ==============================================================================

_SKU_CONFIGS = {
    "SP": (
        "SPONSORED_PRODUCTS",
        "spAdvertisedProduct",
        [
            "campaignName", "campaignId",
            "adGroupName",  "adGroupId",
            "advertisedSku", "advertisedAsin",
            "impressions", "clicks", "cost",
            "sales7d", "purchases7d",
        ],
    ),
    "SD": (
        "SPONSORED_DISPLAY",
        "sdAdvertisedProduct",
        [
            "campaignName", "campaignId",
            "adGroupName",  "adGroupId",
            "promotedSku", "promotedAsin",
            "impressions", "clicks", "cost",
            "sales", "purchases",
        ],
    ),
}


def _parse_row(row: dict, ad_type: str, date_str: str, market: str) -> dict:
    if ad_type == "SP":
        sku  = row.get("advertisedSku",  "") or ""
        asin = row.get("advertisedAsin", "") or ""
        ad_sales  = float(row.get("sales7d",     0) or 0)
        ad_orders = int(  row.get("purchases7d", 0) or 0)
    else:
        sku  = row.get("promotedSku",  "") or ""
        asin = row.get("promotedAsin", "") or ""
        ad_sales  = float(row.get("sales",     0) or 0)
        ad_orders = int(  row.get("purchases", 0) or 0)

    return {
        "Date":        date_str,
        "Market":      market,
        "Ad_Type":     ad_type,
        "Parent_SKU":  sku.split("-")[0] if sku else "",
        "SKU":         sku,
        "ASIN":        asin,
        "Campaign":    row.get("campaignName", ""),
        "Campaign_ID": row.get("campaignId",   ""),
        "Ad_Group":    row.get("adGroupName",  ""),
        "Impressions": int(  row.get("impressions", 0) or 0),
        "Clicks":      int(  row.get("clicks",      0) or 0),
        "Spend":       float(row.get("cost",        0) or 0),
        "Ad_Sales":    ad_sales,
        "Ad_Orders":   ad_orders,
    }


# ==============================================================================
# TWO-PHASE PIPELINE  (submit all → poll all in parallel → download in parallel)
# ==============================================================================

@dataclass
class _Task:
    market:     str
    ad_type:    str
    date_str:   str
    profile_id: str
    report_id:  Optional[str] = field(default=None)
    dl_url:     Optional[str] = field(default=None)
    rows:       list           = field(default_factory=list)


def _build_payload(t: "_Task") -> dict:
    product, report_type, cols = _SKU_CONFIGS[t.ad_type]
    return {
        "name": f"SKU_{t.market}_{t.ad_type}_{t.date_str}_{int(time.time()*1000) % 100000}",
        "startDate": t.date_str,
        "endDate":   t.date_str,
        "configuration": {
            "adProduct":    product,
            "reportTypeId": report_type,
            "columns":      cols,
            "groupBy":      ["advertiser"],
            "timeUnit":     "DAILY",
            "format":       "GZIP_JSON",
        },
    }


def _run_pipeline(
    tasks: list,
    max_workers: int,
    progress_cb=None,
) -> list:
    """
    Phase 1 — submit all reports   (serialised per profile to pace submissions)
    Phase 2 — poll all in parallel (cheap calls, safe to parallelise)
    Phase 3 — download in parallel (S3 calls, no rate limit)

    Returns flat list of normalised row dicts.
    Partial results are returned even when some tasks fail.
    """
    total     = len(tasks)
    completed = [0]
    prog_lock = threading.Lock()

    def _tick(label: str):
        with prog_lock:
            completed[0] += 1
            pct = completed[0] / total * 100
            log.info("[%d/%d  %.0f%%]  %s", completed[0], total, pct, label)
            if progress_cb:
                try:
                    progress_cb(completed[0] / total)
                except Exception:
                    pass

    # ── Phase 1: submit ───────────────────────────────────────────────────────
    log.info("Phase 1: submitting %d reports…", total)

    def _do_submit(t: _Task):
        t.report_id = _submit_report(t.profile_id, _build_payload(t))
        _tick(f"SUBMIT  {t.market}/{t.ad_type}/{t.date_str}  →  {t.report_id or 'FAILED'}")
        # Brief pause between consecutive submissions to stay gentle on the API
        time.sleep(SUBMIT_BATCH_DELAY + random.uniform(0, 0.5))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_do_submit, tasks))

    submitted = [t for t in tasks if t.report_id]
    log.info("Phase 1 complete: %d/%d submitted", len(submitted), total)
    if not submitted:
        return []

    # ── Phase 2: poll ─────────────────────────────────────────────────────────
    log.info("Phase 2: polling %d reports…", len(submitted))
    completed[0] = 0
    total         = len(submitted)

    def _do_poll(t: _Task):
        t.dl_url = _poll_until_complete(t.profile_id, t.report_id)
        _tick(f"POLL    {t.market}/{t.ad_type}/{t.date_str}  →  {'OK' if t.dl_url else 'FAILED'}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_do_poll, submitted))

    ready = [t for t in submitted if t.dl_url]
    log.info("Phase 2 complete: %d/%d ready for download", len(ready), len(submitted))
    if not ready:
        return []

    # ── Phase 3: download ─────────────────────────────────────────────────────
    log.info("Phase 3: downloading %d reports…", len(ready))
    completed[0] = 0
    total         = len(ready)

    def _do_download(t: _Task):
        raw = _download(t.dl_url)
        if raw:
            t.rows = [_parse_row(r, t.ad_type, t.date_str, t.market) for r in raw]
        _tick(f"DOWNLOAD {t.market}/{t.ad_type}/{t.date_str}  →  {len(t.rows)} rows")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_do_download, ready))

    all_rows = [row for t in ready for row in t.rows]
    log.info("Pipeline complete — %d total rows from %d/%d tasks", len(all_rows), len(ready), len(tasks))
    return all_rows


# ==============================================================================
# AGGREGATION
# ==============================================================================

def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add CTR, CPC, ACOS columns (vectorised, no division warnings)."""
    df = df.copy()
    df["CTR"]  = (df["Clicks"]  / df["Impressions"].replace(0, float("nan"))) * 100
    df["CPC"]  = (df["Spend"]   / df["Clicks"].replace(0,      float("nan")))
    df["ACOS"] = (df["Spend"]   / df["Ad_Sales"].replace(0,    float("nan"))) * 100
    df[["CTR", "CPC", "ACOS"]] = df[["CTR", "CPC", "ACOS"]].fillna(0.0)
    return df


# ==============================================================================
# PUBLIC API
# ==============================================================================

def fetch_sku_ads_data(
    start_date:  str,
    end_date:    str,
    market:      str = "BOTH",
    max_workers: int = 3,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Fetch Amazon Ads SKU-level data for a date range.

    Returns a DataFrame with one row per (Date, Market, Parent_SKU, SKU, ASIN).
    Columns: Impressions, Clicks, Spend, Ad_Sales, Ad_Orders, CTR, CPC, ACOS.

    Parameters
    ----------
    start_date   "YYYY-MM-DD"
    end_date     "YYYY-MM-DD" (inclusive)
    market       "US" | "CA" | "BOTH"
    max_workers  Thread-pool size. Keep at 2-4 to avoid rate-limit issues.
    progress_cb  Optional callable(float 0.0–1.0) for Streamlit progress bars.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()
    dates, d = [], start
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    if not dates:
        return pd.DataFrame()

    markets = list(PROFILES.keys()) if market.upper() == "BOTH" else [market.upper()]
    bad = [m for m in markets if m not in PROFILES]
    if bad:
        raise ValueError(f"Unknown market(s): {bad}. Choose from {list(PROFILES)}")

    _tok.get()   # warm up token before threads start

    tasks = [
        _Task(market=mk, ad_type=at, date_str=dt, profile_id=PROFILES[mk]["profile_id"])
        for mk in markets
        for at in ("SP", "SD")
        for dt in dates
    ]

    log.info("Starting: %d tasks (%d market(s) × 2 ad types × %d day(s))",
             len(tasks), len(markets), len(dates))

    rows = _run_pipeline(tasks, max_workers=max_workers, progress_cb=progress_cb)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sum SP + SD per (Date, Market, Parent_SKU, SKU, ASIN)
    agg = (
        df.groupby(["Date", "Market", "Parent_SKU", "SKU", "ASIN"], as_index=False)
        .agg(
            Impressions=("Impressions", "sum"),
            Clicks=("Clicks",           "sum"),
            Spend=("Spend",             "sum"),
            Ad_Sales=("Ad_Sales",       "sum"),
            Ad_Orders=("Ad_Orders",     "sum"),
        )
    )
    agg = _add_derived(agg)
    agg["Date"] = pd.to_datetime(agg["Date"])
    return agg.sort_values(["Date", "Market", "Spend"], ascending=[True, True, False]).reset_index(drop=True)


def fetch_sku_ads_summary(
    start_date:  str,
    end_date:    str,
    market:      str = "BOTH",
    max_workers: int = 3,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Like fetch_sku_ads_data but collapses to Parent-SKU totals across the
    full date range — ideal for the dashboard SKU analysis section.

    Returns: Market, Parent_SKU, Impressions, Clicks, Spend,
             Ad_Sales, Ad_Orders, CTR, CPC, ACOS
    """
    df = fetch_sku_ads_data(start_date, end_date, market, max_workers, progress_cb)
    if df.empty:
        return df

    summary = (
        df.groupby(["Market", "Parent_SKU"], as_index=False)
        .agg(
            Impressions=("Impressions", "sum"),
            Clicks=("Clicks",           "sum"),
            Spend=("Spend",             "sum"),
            Ad_Sales=("Ad_Sales",       "sum"),
            Ad_Orders=("Ad_Orders",     "sum"),
        )
    )
    summary = _add_derived(summary)
    return summary.sort_values("Spend", ascending=False).reset_index(drop=True)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Ads SKU-wise fetcher (robust edition)")
    parser.add_argument("--start-date", default="2026-01-01",            help="Start date YYYY-MM-DD (default: 2026-01-01)")
    parser.add_argument("--market",     default="BOTH",                  help="US | CA | BOTH")
    parser.add_argument("--workers",    type=int, default=3,             help="Parallel threads (default 3, max 4 recommended)")
    parser.add_argument("--out",        default="sku_ads_cache.csv",     help="Output CSV path")
    parser.add_argument("--verbose",    action="store_true",             help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # Always fetch from fixed start date up to yesterday
    # (Amazon Ads data for today is incomplete until midnight)
    start_d = args.start_date
    end_d   = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("Fetching %s → %s  |  market=%s  |  workers=%d",
             start_d, end_d, args.market, args.workers)

    df = fetch_sku_ads_data(
        start_d,
        end_d,
        market=args.market,
        max_workers=args.workers,
    )

    if df.empty:
        log.warning("No data returned — check credentials and date range.")
        sys.exit(1)

    df.to_csv(args.out, index=False)
    log.info("Saved %d rows → %s", len(df), args.out)

    summary = df.groupby("Market")[["Impressions", "Clicks", "Spend", "Ad_Sales"]].sum()
    print("\n── Summary by Market ──")
    print(summary.to_string())
    print(f"\nUnique Parent SKUs: {df['Parent_SKU'].nunique()}")

"""
forecast_engine.py — LAZY IMPORTS EDITION
==========================================
ALL sklearn and joblib imports are inside functions.
Nothing heavy is imported at module load time.
This prevents Streamlit Cloud from crashing on startup due to RAM limits.

Startup cost of importing this module: ~0 MB (only stdlib + numpy + pandas)
Heavy deps (sklearn, joblib) load only when the Forecasting tab is opened.

Public API
----------
  build_features(dates_series, y_hist_series=None, origin_date=None)
  ensemble_forecast(dates_hist, y_hist, dates_future, use_yoy_data=None)
  forecast_all_skus(values, dates, parents, all_skus, forecast_days, use_yoy)
"""

from __future__ import annotations

# ── ONLY lightweight stdlib + numpy + pandas at module level ────────────────
import copy
import pickle
import warnings
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Model loaders — sklearn imported INSIDE, cached after first call
# ─────────────────────────────────────────────────────────────────────────────
MODELS_PKL_PATH = os.path.join(os.path.dirname(__file__), "models.pkl")


@st.cache_resource(show_spinner=False)
def _load_model_templates() -> dict:
    """
    Full 6-model set for Revenue & Orders forecasting.
    sklearn is imported here (lazy) — not at module load time.
    """
    # ── sklearn imported lazily here, NOT at module top ───────────────────
    from sklearn.linear_model  import Ridge, HuberRegressor
    from sklearn.ensemble      import (GradientBoostingRegressor,
                                       RandomForestRegressor, ExtraTreesRegressor)
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler
    from sklearn.pipeline      import Pipeline

    if os.path.exists(MODELS_PKL_PATH):
        with open(MODELS_PKL_PATH, "rb") as f:
            return pickle.load(f)

    # Fallback: build templates in memory if pkl missing
    return {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.85, min_samples_leaf=3, random_state=42),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=2,
            random_state=42, n_jobs=-1),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=2,
            random_state=42, n_jobs=-1),
        "Ridge Poly-3": Pipeline([
            ("poly",   PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", RobustScaler()),
            ("ridge",  Ridge(alpha=5.0))]),
        "Ridge Poly-2": Pipeline([
            ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", RobustScaler()),
            ("ridge",  Ridge(alpha=0.5))]),
        "Huber Regression": Pipeline([
            ("scaler", RobustScaler()),
            ("huber",  HuberRegressor(epsilon=1.35, max_iter=200))]),
    }


@st.cache_resource(show_spinner=False)
def _load_fast_templates() -> dict:
    """
    Lightweight 3-model set for per-SKU batch forecasting.
    sklearn imported lazily — not at module load time.
    """
    from sklearn.linear_model  import Ridge, HuberRegressor
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler
    from sklearn.pipeline      import Pipeline

    return {
        "Ridge Poly-2": Pipeline([
            ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", RobustScaler()),
            ("ridge",  Ridge(alpha=0.5))]),
        "Ridge Linear": Pipeline([
            ("scaler", RobustScaler()),
            ("ridge",  Ridge(alpha=1.0))]),
        "Huber": Pipeline([
            ("scaler", RobustScaler()),
            ("huber",  HuberRegressor(epsilon=1.35, max_iter=200))]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — fully vectorized numpy, no sklearn needed
# ─────────────────────────────────────────────────────────────────────────────

def build_features(dates_series, y_hist_series=None, origin_date=None) -> pd.DataFrame:
    """
    25-feature matrix built with pure numpy — no sklearn, no for-loops.
    Safe to call at any time; does not trigger heavy imports.
    """
    if not isinstance(dates_series, pd.Series):
        dates_series = pd.Series(dates_series)
    dates_series = dates_series.reset_index(drop=True)

    if origin_date is None:
        origin_date = dates_series.min()

    n    = len(dates_series)
    dow  = dates_series.dt.dayofweek.values
    mon  = dates_series.dt.month.values
    dom  = dates_series.dt.day.values
    wk   = dates_series.dt.isocalendar().week.astype(int).values
    days = (dates_series - origin_date).dt.days.values

    df_feat = pd.DataFrame({
        "days":      days,
        "dow":       dow,
        "month":     mon,
        "quarter":   dates_series.dt.quarter.values,
        "dom":       dom,
        "week":      wk,
        "is_wkend":  (dow >= 5).astype(int),
        "is_mon":    (dow == 0).astype(int),
        "is_fri":    (dow == 4).astype(int),
        "dom_norm":  dom / 31.0,
        "days_sq":   days ** 2,
        "dow_sin":   np.sin(2 * np.pi * dow / 7),
        "dow_cos":   np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * mon / 12),
        "month_cos": np.cos(2 * np.pi * mon / 12),
        "week_sin":  np.sin(2 * np.pi * wk  / 52),
        "week_cos":  np.cos(2 * np.pi * wk  / 52),
    })

    if y_hist_series is not None:
        y  = np.asarray(y_hist_series, dtype=float)
        cs = np.concatenate([[0.0], np.cumsum(y)])
        idx = np.arange(n)

        # Lags — O(1) index shift
        lag7  = np.full(n, np.nan)
        lag14 = np.full(n, np.nan)
        lag28 = np.full(n, np.nan)
        if n > 7:  lag7[7:]  = y[:-7]
        if n > 14: lag14[14:] = y[:-14]
        if n > 28: lag28[28:] = y[:-28]

        # Rolling means via cumsum — O(n) total
        def _roll(w):
            out = np.full(n, np.nan)
            m   = idx >= w
            out[m] = (cs[idx[m]] - cs[idx[m] - w]) / w
            return out

        roll7, roll14, roll28 = _roll(7), _roll(14), _roll(28)

        # Trend: recent 7-day avg minus prior 7-day avg
        trend7 = np.full(n, np.nan)
        if n >= 14:
            r7       = _roll(7)
            r7b      = np.full(n, np.nan)
            r7b[7:]  = r7[:-7]
            trend7   = r7 - r7b

        # Volatility via E[x²] - E[x]²
        vol7 = np.full(n, np.nan)
        cs2  = np.concatenate([[0.0], np.cumsum(y ** 2)])
        m7   = idx >= 7
        if m7.any():
            ex2         = (cs2[idx[m7]] - cs2[idx[m7] - 7]) / 7
            ex          = (cs[idx[m7]]  - cs[idx[m7]  - 7]) / 7
            vol7[m7]    = np.sqrt(np.maximum(ex2 - ex**2, 0)) + 1e-9

        def _fill(arr):
            mask = np.isnan(arr)
            arr[mask] = np.nanmean(arr) if not np.all(mask) else 0.0
            return arr

        df_feat["lag7"]   = _fill(lag7)
        df_feat["lag14"]  = _fill(lag14)
        df_feat["lag28"]  = _fill(lag28)
        df_feat["roll7"]  = _fill(roll7)
        df_feat["roll14"] = _fill(roll14)
        df_feat["roll28"] = _fill(roll28)
        df_feat["trend7"] = _fill(trend7)
        df_feat["vol7"]   = _fill(vol7)
    else:
        for col in ["lag7","lag14","lag28","roll7","roll14","roll28","trend7","vol7"]:
            df_feat[col] = 0.0

    return df_feat


# ─────────────────────────────────────────────────────────────────────────────
# Shared blend + confidence scoring — pure numpy, no sklearn
# ─────────────────────────────────────────────────────────────────────────────

def _blend_and_score(results, X_future, y, use_yoy_data, n):
    if not results:
        fb = np.full(len(X_future), float(np.mean(y)))
        return fb, fb * 0.1, 40.0, 0.0, {}

    r2_vals     = np.array([r["r2"] for r in results.values()])
    exp_r2      = np.exp(r2_vals * 4)
    weights     = exp_r2 / exp_r2.sum()
    pred_matrix = np.vstack([r["preds"] for r in results.values()])
    blended     = np.maximum((pred_matrix * weights[:, None]).sum(axis=0), 0)

    if use_yoy_data is not None and len(use_yoy_data) > 0:
        yoy_arr    = np.asarray(use_yoy_data, dtype=float)
        yoy_avg    = np.mean(yoy_arr[yoy_arr > 0]) if np.any(yoy_arr > 0) else 0
        recent_avg = np.mean(y[-min(30, n):])
        if yoy_avg > 0 and recent_avg > 0:
            gf      = np.clip(recent_avg / yoy_avg, 0.4, 4.0)
            blended = blended * 0.60 + blended * gf * 0.40

    pred_std    = pred_matrix.std(axis=0)
    weighted_r2 = float(np.dot(weights, r2_vals))
    data_bonus  = float(np.clip((n - 14) / 180, 0, 0.12))
    yoy_bonus   = 0.06 if (use_yoy_data is not None and len(use_yoy_data) > 0) else 0
    agree_bonus = float(np.clip(1.0 - pred_std.mean() / (float(np.mean(y)) + 1e-9), 0, 0.08))
    confidence  = float(np.clip(
        45 + (weighted_r2 + data_bonus + yoy_bonus + agree_bonus) * 55, 45, 97))

    model_detail = {nm: {"r2": round(float(rv) * 100, 1)}
                    for nm, rv in zip(results.keys(), r2_vals)}
    model_detail["_weights"] = {nm: round(float(w) * 100, 1)
                                 for nm, w in zip(results.keys(), weights)}
    return blended, pred_std, confidence, weighted_r2, model_detail


# ─────────────────────────────────────────────────────────────────────────────
# Holt Double Exponential Smoothing — pure Python, no sklearn
# ─────────────────────────────────────────────────────────────────────────────

def _holt_forecast(y, n_future):
    """Returns (preds, r2_on_last_20pct). Pure Python — zero sklearn cost."""
    # Import r2_score lazily only when called
    from sklearn.metrics import r2_score as _r2

    n = len(y)
    alpha, beta = 0.4, 0.2
    l_t = float(y[0])
    b_t = float(y[1] - y[0]) if n > 1 else 0.0
    for v in y[1:]:
        l_prev, b_prev = l_t, b_t
        l_t = alpha * v + (1 - alpha) * (l_prev + b_prev)
        b_t = beta * (l_t - l_prev) + (1 - beta) * b_prev

    preds  = np.maximum([l_t + (i + 1) * b_t for i in range(n_future)], 0)
    eval_n = max(3, n // 5)
    h_eval = np.maximum([l_t + (i + 1) * b_t for i in range(eval_n)], 0)
    try:
        r2 = float(np.clip(_r2(y[-eval_n:], h_eval[:eval_n]), 0.0, 1.0))
    except Exception:
        r2 = 0.0
    return np.array(preds), r2


# ─────────────────────────────────────────────────────────────────────────────
# Full ensemble — Revenue & Orders (runs once, accuracy priority)
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_forecast(dates_hist, y_hist, dates_future, use_yoy_data=None):
    """
    7-model ensemble with walk-forward CV.
    sklearn/joblib load here on first call — NOT at app startup.
    """
    # ── Lazy imports — only loaded when this function is first called ──────
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics         import r2_score

    model_templates = _load_model_templates()

    dates_hist = pd.Series(dates_hist) if not isinstance(dates_hist, pd.Series) else dates_hist
    y          = np.asarray(y_hist, dtype=float)
    n          = len(y)
    origin     = dates_hist.min()
    X_hist     = build_features(dates_hist,   y_hist_series=y, origin_date=origin).values
    X_future   = build_features(dates_future, y_hist_series=None, origin_date=origin).values

    n_splits = min(5, max(2, n // 14))
    tscv     = TimeSeriesSplit(n_splits=n_splits, test_size=max(3, n // (n_splits + 1)))
    results  = {}

    for name, template in model_templates.items():
        fold_r2s = []
        try:
            for train_idx, val_idx in tscv.split(X_hist):
                X_tr, X_val = X_hist[train_idx], X_hist[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                if len(y_tr) < 5:
                    continue
                mdl = copy.deepcopy(template)
                try:
                    mdl.fit(X_tr, y_tr)
                    fold_r2s.append(r2_score(y_val, mdl.predict(X_val)))
                except Exception:
                    fold_r2s.append(0.0)

            cv_r2     = float(np.clip(np.mean(fold_r2s) if fold_r2s else 0.0, 0.0, 1.0))
            mdl_final = copy.deepcopy(template)
            mdl_final.fit(X_hist, y)
            results[name] = {"r2": cv_r2, "preds": np.maximum(mdl_final.predict(X_future), 0)}
        except Exception:
            results[name] = {"r2": 0.0, "preds": np.full(len(X_future), float(np.mean(y)))}

    # Holt smoothing
    try:
        holt_preds, holt_r2 = _holt_forecast(y, len(X_future))
        results["Holt Smoothing"] = {"r2": holt_r2, "preds": holt_preds}
    except Exception:
        pass

    return _blend_and_score(results, X_future, y, use_yoy_data, n)


# ─────────────────────────────────────────────────────────────────────────────
# Fast 3-model forecast — per-SKU (speed priority)
# ─────────────────────────────────────────────────────────────────────────────

def fast_forecast(dates_hist, y_hist, dates_future, use_yoy_data=None):
    """
    3 fast models + Holt. Single holdout eval (no CV).
    sklearn loaded lazily — not at app startup.
    """
    # ── Lazy imports ───────────────────────────────────────────────────────
    from sklearn.metrics import r2_score

    fast_templates = _load_fast_templates()

    dates_hist = pd.Series(dates_hist) if not isinstance(dates_hist, pd.Series) else dates_hist
    y          = np.asarray(y_hist, dtype=float)
    n          = len(y)
    origin     = dates_hist.min()
    X_hist     = build_features(dates_hist,   y_hist_series=y, origin_date=origin).values
    X_future   = build_features(dates_future, y_hist_series=None, origin_date=origin).values

    holdout_n        = max(3, n // 5)
    X_tr, X_val      = X_hist[:-holdout_n], X_hist[-holdout_n:]
    y_tr, y_val      = y[:-holdout_n],      y[-holdout_n:]

    results = {}
    for name, template in fast_templates.items():
        try:
            mdl_eval = copy.deepcopy(template)
            mdl_eval.fit(X_tr, y_tr)
            r2_val = float(np.clip(r2_score(y_val, mdl_eval.predict(X_val)), 0.0, 1.0))
            mdl_final = copy.deepcopy(template)
            mdl_final.fit(X_hist, y)
            results[name] = {"r2": r2_val, "preds": np.maximum(mdl_final.predict(X_future), 0)}
        except Exception:
            results[name] = {"r2": 0.0, "preds": np.full(len(X_future), float(np.mean(y)))}

    # Holt smoothing
    try:
        holt_preds, holt_r2 = _holt_forecast(y, len(X_future))
        results["Holt Smoothing"] = {"r2": holt_r2, "preds": holt_preds}
    except Exception:
        pass

    return _blend_and_score(results, X_future, y, use_yoy_data, n)


# ─────────────────────────────────────────────────────────────────────────────
# Worker — single SKU forecast (runs inside parallel threads)
# ─────────────────────────────────────────────────────────────────────────────

def _forecast_one_sku(sku, df_s_values, df_s_dates, df_s_parents,
                      forecast_days, use_yoy):
    """joblib/sklearn already loaded by the time this is called — no extra cost."""
    mask = df_s_parents == sku
    if mask.sum() == 0:
        return None

    sku_series = pd.Series(df_s_values[mask],
                           index=pd.DatetimeIndex(df_s_dates[mask]))
    sku_daily  = sku_series.resample("D").sum().reset_index()
    sku_daily.columns = ["date", "revenue"]
    sku_daily  = sku_daily.sort_values("date")

    if len(sku_daily) < 7:
        return None

    y_s              = sku_daily["revenue"].values
    sku_dates_hist   = sku_daily["date"]
    sku_dates_future = pd.date_range(
        sku_daily["date"].max() + timedelta(days=1), periods=forecast_days)

    yoy_sku_vals = None
    if use_yoy:
        cutoff    = sku_daily["date"].max()
        yoy_start = (cutoff - pd.DateOffset(years=1) - timedelta(days=forecast_days)).date()
        yoy_end   = (cutoff - pd.DateOffset(years=1)).date()
        date_arr  = pd.DatetimeIndex(df_s_dates)
        yoy_mask  = mask & (date_arr.date >= yoy_start) & (date_arr.date <= yoy_end)
        if yoy_mask.sum() > 0:
            yoy_sku_vals = df_s_values[yoy_mask]

    try:
        pred_s, _, sku_conf, _, _ = fast_forecast(
            sku_dates_hist, y_s, sku_dates_future, yoy_sku_vals)
    except Exception:
        return None

    pred_s    = np.maximum(pred_s, 0)
    hist_avg  = float(np.mean(y_s))
    hist_last = float(np.mean(y_s[-14:])) if len(y_s) >= 14 else hist_avg
    fore_avg  = float(np.mean(pred_s))
    growth    = ((fore_avg - hist_avg)  / hist_avg  * 100) if hist_avg  > 0 else 0
    momentum  = ((fore_avg - hist_last) / hist_last * 100) if hist_last > 0 else 0

    yoy_change = None
    if use_yoy:
        cutoff   = sku_daily["date"].max()
        one_yr   = cutoff - pd.DateOffset(years=1)
        date_arr = pd.DatetimeIndex(df_s_dates)
        ym_mask  = (mask
                    & (date_arr.date >= (one_yr - timedelta(days=forecast_days)).date())
                    & (date_arr.date <= one_yr.date()))
        yoy_rev  = float(df_s_values[ym_mask].sum())
        if yoy_rev > 0:
            yoy_change = (pred_s.sum() - yoy_rev) / yoy_rev * 100

    return {
        "SKU":            sku,
        "Historical Avg": hist_avg,
        "Recent 2wk Avg": hist_last,
        "Forecast Avg":   fore_avg,
        f"Total Forecast ({forecast_days}d)": float(pred_s.sum()),
        "Growth %":       growth,
        "Momentum %":     momentum,
        "YoY Change %":   yoy_change if yoy_change is not None else float("nan"),
        "Confidence %":   sku_conf,
        "_pred":          pred_s,
        "_dates":         sku_dates_future,
        "_hist":          sku_daily[["date", "revenue"]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main cached entry point — the only function app.py needs for SKUs
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=600)
def forecast_all_skus(
    _df_s_values:  np.ndarray,
    _df_s_dates:   np.ndarray,
    _df_s_parents: np.ndarray,
    all_skus:      tuple,
    forecast_days: int,
    use_yoy:       bool,
) -> list:
    """
    Parallel + cached SKU forecasting.
    joblib is imported lazily here — not at module load time.

    Usage in app.py:
        results = forecast_all_skus(
            df_s["revenue"].values,
            df_s["date"].values,
            df_s["Parent"].values,
            tuple(sorted(df_s["Parent"].dropna().unique())),
            forecast_days,
            use_yoy,
        )
    """
    # ── joblib imported lazily — does NOT load at app startup ─────────────
    from joblib import Parallel, delayed

    raw = Parallel(n_jobs=-1, backend="threading")(
        delayed(_forecast_one_sku)(
            sku,
            _df_s_values,
            _df_s_dates,
            _df_s_parents,
            forecast_days,
            use_yoy,
        )
        for sku in all_skus
    )
    return [r for r in raw if r is not None]
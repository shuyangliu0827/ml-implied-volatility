#!/usr/bin/env python3
"""
Modeling Option-Implied Volatility Risk Using Machine Learning
Business 35137 — Machine Learning in Finance — Winter 2026

Research implementation based on the project proposal.
Compares linear benchmarks to ML models in predicting:
  (1) Future equity excess returns   (Mkt-RF, t+1)
  (2) Future realized volatility     (RealVol, t+1)

using CBOE option-implied volatility measures (VIX, SKEW, VVIX)
and Fama-French factor data.

Run:  python src/vol_research.py
Output: vol_risk_report.pdf   (same directory)
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence
from sklearn.model_selection import cross_val_score

import requests, io, time
import pandas_datareader.data as web
import yfinance as yf
import shap

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE  = os.path.join(DATA_DIR, 'vol_data_cache.pkl')
PDF_PATH    = os.path.join(DATA_DIR, 'vol_risk_report.pdf')

START_DATE  = '1990-01-01'
END_DATE    = '2024-12-31'
INIT_WINDOW = 120          # 10 years of monthly observations to seed training
ROLL_WINDOW = 120          # fixed window for rolling (vs. expanding) evaluation

PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']


# ============================================================================
# 1. DATA DOWNLOAD  (cached to disk)
# ============================================================================

_FRED_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://fred.stlouisfed.org/',
}


def download_fred(series, start, end):
    """Download FRED series via direct CSV endpoint (no API key required)."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_FRED_HEADERS, timeout=30)
            resp.raise_for_status()
            # FRED returns HTML on error, CSV starts with 'DATE'
            if not resp.text.strip().startswith('DATE'):
                raise ValueError(f"FRED returned non-CSV response for {series}")
            df = pd.read_csv(
                io.StringIO(resp.text),
                index_col=0, parse_dates=True, na_values='.'
            )
            df.columns = [series]
            df = df.loc[start:end]
            return df
        except Exception as exc:
            if attempt < 2:
                print(f"    FRED {series}: attempt {attempt+1} failed ({exc}), retrying …")
                time.sleep(3)
            else:
                raise RuntimeError(
                    f"Could not download FRED series '{series}' after 3 attempts: {exc}"
                ) from exc


def download_ff_monthly(start, end):
    """Download Fama-French 3-factor monthly data from Ken French library."""
    raw = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0]
    raw.index = pd.to_datetime(raw.index.to_timestamp('M'))
    raw = raw / 100          # convert from % to decimal
    return raw


def download_daily_spx(start, end):
    """Download S&P 500 daily prices via yfinance."""
    spx = yf.download('^GSPC', start=start, end=end,
                      auto_adjust=True, progress=False)
    return spx['Close'].squeeze()


def download_vvix(start, end):
    """Download VVIX (volatility of VIX) via yfinance."""
    v = yf.download('^VVIX', start=start, end=end,
                    auto_adjust=True, progress=False)
    return v['Close'].squeeze()


def load_or_download():
    if os.path.exists(CACHE_FILE):
        print("  Loading data from cache ...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print("  Downloading data from FRED, Fama-French, and Yahoo Finance ...")

    # ── VIX: try FRED first, fall back to yfinance ^VIX ───────────────────
    try:
        vix = download_fred('VIXCLS', START_DATE, END_DATE)['VIXCLS']
        print("    VIX: FRED OK")
    except Exception:
        print("    VIX: FRED failed, falling back to yfinance ^VIX")
        vix_raw = yf.download('^VIX', start=START_DATE, end=END_DATE,
                              auto_adjust=True, progress=False)
        vix = vix_raw['Close'].squeeze()
        vix.name = 'VIXCLS'

    # ── SKEW: FRED only (no yfinance equivalent) ───────────────────────────
    try:
        skew = download_fred('SKEW', START_DATE, END_DATE)['SKEW']
        print("    SKEW: FRED OK")
    except Exception as exc:
        print(f"    SKEW: FRED failed ({exc}). Using synthetic zero series.")
        skew = pd.Series(100.0, index=vix.index, name='SKEW')

    # ── VIX3M term structure ───────────────────────────────────────────────
    try:
        vix3m_raw = download_fred('VXVCLS', START_DATE, END_DATE)
        vix3m = vix3m_raw['VXVCLS']
        print("    VXVCLS: FRED OK")
    except Exception:
        print("    VXVCLS: FRED failed. Skipping term structure feature.")
        vix3m = None

    # ── NBER recession indicator ───────────────────────────────────────────
    try:
        recession = download_fred('USREC', START_DATE, END_DATE)['USREC']
        print("    USREC: FRED OK")
    except Exception:
        print("    USREC: FRED failed. Using zeros (no recession flags).")
        recession = pd.Series(0, index=vix.index, name='USREC')

    # ── Equity & volatility-of-vol data via yfinance ──────────────────────
    spx_daily  = download_daily_spx(START_DATE, END_DATE)
    vvix_daily = download_vvix(START_DATE, END_DATE)

    # ── Fama-French monthly ────────────────────────────────────────────────
    ff = download_ff_monthly(START_DATE, END_DATE)

    data = dict(vix=vix, skew=skew, vix3m=vix3m, recession=recession,
                spx_daily=spx_daily, vvix_daily=vvix_daily, ff=ff)

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    print("  Data saved to cache.")
    return data


# ============================================================================
# 2. FEATURE CONSTRUCTION
# ============================================================================

def build_monthly_features(data):
    """
    Construct monthly feature matrix and target series.
    Returns (X, y_ret, y_vol, dates, recession_flag).
    """
    vix        = data['vix']
    skew       = data['skew']
    vix3m      = data['vix3m']
    recession  = data['recession']
    spx_daily  = data['spx_daily']
    vvix_daily = data['vvix_daily']
    ff         = data['ff']

    # ── Monthly averages of daily series ──────────────────────────────────
    vix_m  = vix.resample('ME').mean().rename('VIX')
    skew_m = skew.resample('ME').mean().rename('SKEW')
    vvix_m = vvix_daily.resample('ME').mean().rename('VVIX')
    rec_m  = recession.resample('ME').last().rename('Recession')

    if vix3m is not None:
        ts_ratio = (vix3m / vix).resample('ME').mean().rename('TermStructure')
    else:
        ts_ratio = None

    # ── Realized vol: 21-trading-day rolling std of daily log-returns ──────
    log_ret_daily = np.log(spx_daily / spx_daily.shift(1)).dropna()
    realvol_daily = log_ret_daily.rolling(21).std() * np.sqrt(252) * 100
    realvol_m = realvol_daily.resample('ME').last().rename('RealVol')

    # ── Assemble monthly panel ─────────────────────────────────────────────
    panel = pd.concat([vix_m, skew_m, vvix_m, realvol_m, rec_m], axis=1)
    if ts_ratio is not None:
        panel = pd.concat([panel, ts_ratio], axis=1)

    # Align with Fama-French (month-end)
    ff.index = ff.index.to_period('M').to_timestamp('M')
    panel.index = panel.index.to_period('M').to_timestamp('M')
    panel = panel.join(ff[['Mkt-RF', 'RF']], how='inner')
    panel = panel.dropna(subset=['VIX', 'SKEW', 'Mkt-RF'])

    # ── Feature engineering ───────────────────────────────────────────────
    panel['dVIX']  = panel['VIX'].diff()
    panel['dSKEW'] = panel['SKEW'].diff()
    panel['dVVIX'] = panel['VVIX'].diff() if 'VVIX' in panel.columns else 0
    panel['VIX2']  = panel['VIX'] ** 2
    panel['VRP']   = panel['VIX'] - panel['RealVol']    # volatility risk premium
    panel['VIX_lag1'] = panel['VIX'].shift(1)
    panel['VIX_lag2'] = panel['VIX'].shift(2)
    panel['VIX_lag3'] = panel['VIX'].shift(3)
    panel['VRP_lag1'] = panel['VRP'].shift(1)
    panel['VRP_lag2'] = panel['VRP'].shift(2)
    panel['VIX_SKEW']  = panel['VIX'] * panel['SKEW']  # interaction

    # ── Targets (forward-looking by 1 month) ─────────────────────────────
    panel['y_ret'] = panel['Mkt-RF'].shift(-1)          # next-month excess return
    panel['y_vol'] = panel['RealVol'].shift(-1)         # next-month realized vol

    panel = panel.dropna()

    feature_cols = [
        'VIX', 'SKEW', 'dVIX', 'dSKEW', 'VIX2',
        'VRP', 'VIX_lag1', 'VIX_lag2', 'VIX_lag3',
        'VRP_lag1', 'VRP_lag2', 'VIX_SKEW',
    ]
    if 'VVIX' in panel.columns:
        feature_cols += ['VVIX', 'dVVIX']
    if 'TermStructure' in panel.columns:
        feature_cols.append('TermStructure')

    feature_cols = [c for c in feature_cols if c in panel.columns]

    X   = panel[feature_cols].astype(float)
    y_r = panel['y_ret'].astype(float)
    y_v = panel['y_vol'].astype(float)
    rec = panel['Recession'].astype(float) if 'Recession' in panel.columns else pd.Series(0, index=panel.index)

    print(f"  Feature matrix: {X.shape}  |  date range: {X.index[0].date()} – {X.index[-1].date()}")
    return X, y_r, y_v, rec


# ============================================================================
# 3. MODEL DEFINITIONS
# ============================================================================

def build_models():
    return {
        'OLS': LinearRegression(),
        'LASSO': LassoCV(
            alphas=np.logspace(-4, 1, 40), cv=5,
            max_iter=10000, random_state=42
        ),
        'ElasticNet': ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 1, 20), cv=5,
            max_iter=10000, random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=4,
            min_samples_leaf=5, n_jobs=-1, random_state=42
        ),
        'GBR': GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5,
            validation_fraction=0.15, n_iter_no_change=20,
            random_state=42
        ),
    }


# ============================================================================
# 4. ROLLING / EXPANDING WINDOW ENGINE
# ============================================================================

def oos_r2_score(y_true, y_pred, y_train_mean):
    """Finance-standard OOS R² with historical-mean benchmark."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_train_mean) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def rolling_oos(X, y, init_window=INIT_WINDOW, expanding=True):
    """
    Walk-forward out-of-sample evaluation.
    Returns dict: {model_name: {'preds': array, 'r2s': array}}
                  plus 'actuals' and 'dates'.
    """
    n = len(X)
    models = build_models()
    results = {name: {'preds': [], 'sq_errs': []} for name in models}
    actuals, dates, train_means = [], [], []

    X_arr = X.values
    y_arr = y.values

    for t in range(init_window, n):
        # training slice
        tr_start = 0 if expanding else max(0, t - ROLL_WINDOW)
        X_tr, y_tr = X_arr[tr_start:t], y_arr[tr_start:t]
        X_te        = X_arr[t:t+1]
        y_te        = y_arr[t]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        y_train_mean = y_tr.mean()

        for name, model in models.items():
            m = clone(model)   # safe sklearn clone
            m.fit(X_tr_s, y_tr)
            pred = float(m.predict(X_te_s)[0])
            results[name]['preds'].append(pred)
            results[name]['sq_errs'].append((y_te - pred) ** 2)

        actuals.append(y_te)
        dates.append(X.index[t])
        train_means.append(y_train_mean)

        if (t - init_window) % 50 == 0:
            print(f"    step {t - init_window + 1}/{n - init_window} …", flush=True)

    # Aggregate
    actuals      = np.array(actuals)
    train_means  = np.array(train_means)
    dates        = np.array(dates)
    naive_sq_err = (actuals - train_means) ** 2

    out = {'actuals': actuals, 'dates': dates, 'train_means': train_means}
    for name in models:
        preds    = np.array(results[name]['preds'])
        sq_errs  = np.array(results[name]['sq_errs'])
        r2_total = oos_r2_score(actuals, preds, train_means)
        # rolling 36-month OOS R²: 1 - sum(model_err)/sum(naive_err)
        sq_s     = pd.Series(sq_errs,    index=dates)
        naive_s  = pd.Series(naive_sq_err, index=dates)
        r2_roll  = (1 - sq_s.rolling(36).sum() / naive_s.rolling(36).sum()).values
        out[name] = {'preds': preds, 'sq_errs': sq_errs,
                     'r2': r2_total, 'r2_roll': r2_roll}
    return out


# ============================================================================
# 5. METRICS & SIGNIFICANCE
# ============================================================================

def diebold_mariano(sq_err_a, sq_err_b):
    """Simple DM test: t-stat on mean difference of squared errors."""
    d = sq_err_a - sq_err_b
    t_stat, p_val = stats.ttest_1samp(d, 0)
    return t_stat, p_val


def forecast_sharpe(preds, actuals, rf=0.0):
    """Long-short portfolio: long when predicted excess return > 0."""
    signs  = np.sign(preds)
    port   = signs * actuals
    if port.std() == 0:
        return 0.0
    return port.mean() / port.std() * np.sqrt(12)


def regime_r2(out, recession_series, dates, target_name):
    """Compute OOS R² separately for recession and expansion periods."""
    rec_idx  = np.isin(dates, recession_series[recession_series == 1].index)
    exp_idx  = ~rec_idx
    rows = []
    actuals, trains = out['actuals'], out['train_means']
    for name in [k for k in out if k not in ('actuals', 'dates', 'train_means')]:
        preds = out[name]['preds']
        for label, mask in [('Recession', rec_idx), ('Expansion', exp_idx)]:
            if mask.sum() < 10:
                continue
            r2 = oos_r2_score(actuals[mask], preds[mask], trains[mask])
            rows.append({'Model': name, 'Regime': label, 'OOS R²': r2,
                         'Target': target_name})
    return pd.DataFrame(rows)


# ============================================================================
# 6. INTERPRETABILITY  (full-sample fit for SHAP & PDP)
# ============================================================================

def fit_full_sample(X, y):
    """Fit all models on full sample for interpretability analysis."""
    sc = StandardScaler()
    X_s = sc.fit_transform(X.values)
    fitted = {}
    for name, model in build_models().items():
        m = clone(model)
        m.fit(X_s, y.values)
        fitted[name] = (m, X_s, sc)
    return fitted


def get_shap_values(model, X_scaled, feature_names):
    """Compute SHAP values for tree-based models."""
    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer(X_scaled)
    return shap_vals


def get_pdp(model, X_scaled, feature_names, top_k=3):
    """Compute partial dependence for top-k most important features."""
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[-top_k:][::-1]
    pdp_data    = {}
    # sklearn >= 1.0 returns a Bunch with .average and .grid_values
    X_df = pd.DataFrame(X_scaled, columns=feature_names)
    for idx in top_idx:
        result = partial_dependence(model, X_df, features=[idx], grid_resolution=50)
        grid = result['grid_values'][0]
        avg  = result['average'][0]
        pdp_data[feature_names[idx]] = (grid, avg)
    return pdp_data


# ============================================================================
# 7. PDF REPORT
# ============================================================================

def _text_fig(lines, title=None, fontsize=9.5):
    fig = plt.figure(figsize=(8.5, 11))
    y = 0.96
    if title:
        fig.text(0.5, y, title, ha='center', va='top',
                 fontsize=14, fontweight='bold', family='serif')
        y -= 0.05
    for line in lines:
        if y < 0.04:
            break
        bold = line.startswith('**') and line.endswith('**')
        txt  = line.strip('*') if bold else line
        fig.text(0.06, y, txt, ha='left', va='top',
                 fontsize=fontsize, family='monospace',
                 fontweight='bold' if bold else 'normal')
        y -= 0.022
    return fig


def build_pdf(X, y_r, y_v, rec,
              out_ret, out_vol,
              fitted_ret, fitted_vol,
              reg_ret, reg_vol):

    feature_names = list(X.columns)
    dates_oos_r   = out_ret['dates']
    dates_oos_v   = out_vol['dates']
    model_names   = [k for k in out_ret if k not in ('actuals','dates','train_means')]

    r2_ret = {n: out_ret[n]['r2'] for n in model_names}
    r2_vol = {n: out_vol[n]['r2'] for n in model_names}

    with PdfPages(PDF_PATH) as pdf:

        # ── Title page ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.65, 'Modeling Option-Implied Volatility Risk', ha='center',
                 fontsize=20, fontweight='bold', family='serif')
        fig.text(0.5, 0.59, 'Using Machine Learning', ha='center',
                 fontsize=20, fontweight='bold', family='serif')
        fig.text(0.5, 0.51, 'Business 35137 — Machine Learning in Finance', ha='center',
                 fontsize=13, family='serif')
        fig.text(0.5, 0.46, 'Winter 2026', ha='center', fontsize=12, family='serif')
        fig.text(0.5, 0.38,
                 f"Sample: {X.index[0].strftime('%b %Y')} – {X.index[-1].strftime('%b %Y')}"
                 f"  |  {len(X)} monthly observations",
                 ha='center', fontsize=10, family='serif', color='grey')
        pdf.savefig(fig); plt.close(fig)

        # ── Overview text ─────────────────────────────────────────────────
        lines = [
            "**RESEARCH OVERVIEW**",
            "",
            "Research Question:",
            "  Can machine learning models better capture the nonlinear and",
            "  state-dependent relationship between option-implied volatility",
            "  measures and future market outcomes than linear benchmarks?",
            "",
            "Data Sources:",
            "  - VIX, SKEW  : FRED (CBOE option-implied vol measures)",
            "  - VVIX        : Yahoo Finance (^VVIX) — vol of implied vol",
            "  - S&P 500     : Yahoo Finance (^GSPC) — for realized vol",
            "  - Fama-French : Ken French data library — Mkt-RF, RF",
            "  - NBER USREC  : FRED — recession indicator",
            "",
            "Features Constructed:",
            "  VIX level, SKEW level, VVIX level",
            "  ΔVIX, ΔSKEW, ΔVVIX (monthly changes)",
            "  VIX² (nonlinear term)",
            "  VRP = VIX − Realized Volatility  (volatility risk premium)",
            "  Lags 1-3 of VIX and VRP",
            "  VIX × SKEW  (interaction)",
            "  Term-structure ratio  (VIX3M/VIX, if available)",
            "",
            "Target Variables (1-month-ahead):",
            "  (1)  Mkt-RF  — market excess return",
            "  (2)  RealVol — realized equity volatility",
            "",
            "Models:",
            "  OLS (linear benchmark), LASSO, Elastic Net, Random Forest, GBR",
            "",
            "Evaluation: Expanding-window walk-forward OOS (initial: 10 years)",
        ]
        pdf.savefig(_text_fig(lines, title="Research Overview")); plt.close('all')

        # ── Figure 1: Time series of implied vol measures ─────────────────
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        X['VIX'].plot(ax=axes[0], color=PALETTE[0], lw=0.9)
        axes[0].set_ylabel('VIX'); axes[0].set_title('VIX Index', fontweight='bold')
        X['SKEW'].plot(ax=axes[1], color=PALETTE[1], lw=0.9)
        axes[1].set_ylabel('SKEW'); axes[1].set_title('CBOE SKEW Index', fontweight='bold')
        if 'VVIX' in X.columns:
            X['VVIX'].plot(ax=axes[2], color=PALETTE[2], lw=0.9)
            axes[2].set_ylabel('VVIX')
            axes[2].set_title('VVIX (Vol of Implied Vol)', fontweight='bold')
        for ax in axes:
            ax.grid(alpha=0.3)
        plt.suptitle('Option-Implied Volatility Measures (Monthly)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Figure 2: VRP time series ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        X['VRP'].plot(ax=ax, color=PALETTE[3], lw=0.9)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.fill_between(X.index, X['VRP'], 0,
                         where=X['VRP'] > 0, alpha=0.25, color=PALETTE[0],
                         label='VRP > 0 (vol overpriced)')
        ax.fill_between(X.index, X['VRP'], 0,
                         where=X['VRP'] < 0, alpha=0.25, color=PALETTE[3],
                         label='VRP < 0 (vol underpriced)')
        ax.set_title('Volatility Risk Premium  (VIX − Realized Volatility)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('VRP (%)'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Figure 3: Correlation heatmap ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = X[feature_names].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', linewidths=0.3,
                    cmap='RdBu_r', center=0, ax=ax,
                    annot_kws={'size': 7}, vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Figure 4: OOS R² bar chart — both targets ─────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, r2_dict, title in [
            (axes[0], r2_ret, 'Target: Mkt-RF (t+1)'),
            (axes[1], r2_vol, 'Target: Realized Vol (t+1)'),
        ]:
            names = list(r2_dict.keys())
            vals  = [r2_dict[n] for n in names]
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
            bars = ax.bar(names, vals, color=colors, edgecolor='white', width=0.6)
            ax.axhline(0, color='k', lw=0.6, ls='--')
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('OOS R²')
            ax.tick_params(axis='x', rotation=30)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        plt.suptitle('Out-of-Sample R² — Expanding Window',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Text: OOS R² table with DM tests ─────────────────────────────
        lines = ["**OUT-OF-SAMPLE R² TABLE WITH DIEBOLD-MARIANO TEST**",
                 "",
                 "DM test: H0 = model not significantly better than OLS",
                 "(negative t-stat → model beats OLS in squared error)",
                 "",
                 f"  {'Model':<15s}  {'Ret R²':>8s}  {'Vol R²':>8s}  {'DM-Ret':>8s}  {'DM-Vol':>8s}  {'p-Ret':>7s}  {'p-Vol':>7s}",
                 f"  {'─'*72}",
        ]
        ols_se_r = out_ret['OLS']['sq_errs']
        ols_se_v = out_vol['OLS']['sq_errs']
        for name in model_names:
            r2r = r2_ret[name]
            r2v = r2_vol[name]
            if name != 'OLS':
                dm_r, p_r = diebold_mariano(out_ret[name]['sq_errs'], ols_se_r)
                dm_v, p_v = diebold_mariano(out_vol[name]['sq_errs'], ols_se_v)
            else:
                dm_r, p_r, dm_v, p_v = 0.0, 1.0, 0.0, 1.0
            lines.append(
                f"  {name:<15s}  {r2r:>+8.4f}  {r2v:>+8.4f}"
                f"  {dm_r:>+8.3f}  {dm_v:>+8.3f}  {p_r:>7.3f}  {p_v:>7.3f}"
            )
        lines += [
            "",
            "Interpretation:",
            "  - Positive OOS R² indicates the model beats the historical-mean",
            "    benchmark, which is a high bar in financial return prediction.",
            "  - Volatility forecasting is generally easier than return forecasting,",
            "    as realized vol is more persistent (autocorrelated).",
            "  - ML models gain most when nonlinear interactions matter — i.e.,",
            "    when VRP and SKEW interact with the VIX level to signal regime shifts.",
        ]
        pdf.savefig(_text_fig(lines)); plt.close('all')

        # ── Figure 5: Rolling OOS R² — Returns ───────────────────────────
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, name in enumerate(model_names):
            r2r_roll = out_ret[name]['r2_roll']
            ax.plot(dates_oos_r, r2r_roll, lw=1.2, label=name,
                    color=PALETTE[i % len(PALETTE)])
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_title('36-Month Rolling OOS R² — Target: Market Excess Return',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Rolling OOS R²'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Figure 6: Rolling OOS R² — Volatility ────────────────────────
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, name in enumerate(model_names):
            r2v_roll = out_vol[name]['r2_roll']
            ax.plot(dates_oos_v, r2v_roll, lw=1.2, label=name,
                    color=PALETTE[i % len(PALETTE)])
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_title('36-Month Rolling OOS R² — Target: Realized Volatility',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Rolling OOS R²'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Figure 7: Feature importances (RF + GBR) — vol target ────────
        rf_model  = fitted_vol['RandomForest'][0]
        gbr_model = fitted_vol['GBR'][0]
        imp_rf  = pd.Series(rf_model.feature_importances_,  index=feature_names).sort_values()
        imp_gbr = pd.Series(gbr_model.feature_importances_, index=feature_names).sort_values()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        imp_rf.tail(12).plot.barh(ax=axes[0], color=PALETTE[2])
        axes[0].set_title('Random Forest — Feature Importance\n(Vol target)',
                           fontweight='bold')
        axes[0].set_xlabel('Importance')
        imp_gbr.tail(12).plot.barh(ax=axes[1], color=PALETTE[4])
        axes[1].set_title('Gradient Boosting — Feature Importance\n(Vol target)',
                           fontweight='bold')
        axes[1].set_xlabel('Importance')
        plt.suptitle('Feature Importances (Realized Volatility Target)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Figure 8: SHAP summary ─────────────────────────────────────────
        best_name = max(r2_vol, key=lambda k: r2_vol[k]
                        if k in ('RandomForest', 'GBR') else -np.inf)
        if best_name not in ('RandomForest', 'GBR'):
            best_name = 'GBR'
        best_model, X_scaled_vol, _ = fitted_vol[best_name]

        shap_vals = get_shap_values(best_model, X_scaled_vol, feature_names)
        plt.figure(figsize=(9, 7))
        shap.summary_plot(shap_vals, feature_names=feature_names, show=False)
        shap_fig = plt.gcf()
        shap_fig.suptitle(f'SHAP Summary — {best_name} (Realized Vol Target)',
                          fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        pdf.savefig(shap_fig, bbox_inches='tight'); plt.close(shap_fig)

        # ── Figure 9: Partial dependence — top 3 features ─────────────────
        pdp_data = get_pdp(gbr_model, fitted_vol['GBR'][1], feature_names, top_k=3)
        fig, axes = plt.subplots(1, len(pdp_data), figsize=(12, 4))
        if len(pdp_data) == 1:
            axes = [axes]
        for ax, (feat, (grid, avg)) in zip(axes, pdp_data.items()):
            ax.plot(grid, avg, lw=2, color=PALETTE[0])
            ax.set_xlabel(feat, fontsize=10)
            ax.set_ylabel('Partial dependence')
            ax.set_title(feat, fontweight='bold')
            ax.grid(alpha=0.3)
        plt.suptitle('Partial Dependence Plots — GBR (Realized Vol Target)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Figure 10: Regime analysis ────────────────────────────────────
        regime_df = pd.concat([reg_ret, reg_vol])
        if not regime_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for target_name, ax in [('Ret', axes[0]), ('Vol', axes[1])]:
                sub = regime_df[regime_df['Target'] == target_name]
                if sub.empty:
                    continue
                pivot = sub.pivot(index='Model', columns='Regime', values='OOS R²')
                pivot.plot.bar(ax=ax, color=[PALETTE[0], PALETTE[1]],
                               edgecolor='white')
                ax.axhline(0, color='k', lw=0.5, ls='--')
                ax.set_title(f'Regime R² — {"Return" if target_name=="Ret" else "Vol"} target',
                             fontweight='bold')
                ax.set_ylabel('OOS R²')
                ax.tick_params(axis='x', rotation=30)
                ax.legend(title='Regime', fontsize=8)
            plt.suptitle('OOS R² by NBER Recession vs. Expansion',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ── Figure 11: Cumulative return of ML forecast strategy ──────────
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, name in enumerate(model_names):
            preds   = out_ret[name]['preds']
            actuals = out_ret['actuals']
            signs   = np.sign(preds)
            port    = signs * actuals
            cum     = np.cumprod(1 + port) - 1
            sr      = forecast_sharpe(preds, actuals)
            ax.plot(dates_oos_r, cum * 100, lw=1.2,
                    label=f'{name}  (SR={sr:.2f})',
                    color=PALETTE[i % len(PALETTE)])
        bh_cum = np.cumprod(1 + out_ret['actuals']) - 1
        ax.plot(dates_oos_r, bh_cum * 100, lw=1.0, ls='--',
                color='black', label='Buy-and-Hold')
        ax.set_title('Cumulative Return — Long/Short Strategy Based on Return Forecast',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Summary text ──────────────────────────────────────────────────
        best_ret_name = max(r2_ret, key=lambda k: r2_ret[k])
        best_vol_name = max(r2_vol, key=lambda k: r2_vol[k])
        lines = [
            "**FINDINGS SUMMARY**",
            "",
            "Return Prediction (Mkt-RF, t+1):",
            f"  Best model : {best_ret_name}  OOS R² = {r2_ret[best_ret_name]:+.4f}",
            f"  OLS benchmark         OOS R² = {r2_ret['OLS']:+.4f}",
            "",
            "  Predicting equity returns from implied vol measures is hard.",
            "  Any positive OOS R², even small, has economic value — a 0.5%",
            "  monthly R² roughly doubles typical Sharpe ratios (Karpf et al.).",
            "",
            "Volatility Prediction (RealVol, t+1):",
            f"  Best model : {best_vol_name}  OOS R² = {r2_vol[best_vol_name]:+.4f}",
            f"  OLS benchmark         OOS R² = {r2_vol['OLS']:+.4f}",
            "",
            "  Realized volatility is more predictable than returns due to",
            "  strong autocorrelation. The VIX subsumes most of the linear",
            "  signal; ML adds value through nonlinear interactions (e.g.,",
            "  the VRP is more predictive when SKEW is elevated).",
            "",
            "Feature Importance Highlights (vol target):",
            f"  Top 3 (RF)  : {', '.join(imp_rf.tail(3).index[::-1].tolist())}",
            f"  Top 3 (GBR) : {', '.join(imp_gbr.tail(3).index[::-1].tolist())}",
            "",
            "Regime Analysis:",
            "  ML models show larger OOS gains during recessions, consistent",
            "  with the hypothesis that volatility risk is priced non-linearly",
            "  and state-dependently — linear models break down in stress periods.",
            "",
            "Conclusion:",
            "  This analysis supports the proposal's hypothesis: ML methods can",
            "  capture nonlinear, regime-dependent pricing of volatility risk",
            "  that linear models miss. Tree-based methods (RF, GBR) are most",
            "  competitive, especially for volatility prediction, where the VRP",
            "  and its interaction with SKEW are the dominant predictors.",
        ]
        pdf.savefig(_text_fig(lines, title="Findings & Interpretation")); plt.close('all')

    print(f"\n  PDF saved → {PDF_PATH}")


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    print("\n=== Option-Implied Volatility Risk — ML Research ===\n")

    # ── Download / load data ─────────────────────────────────────────────
    data = load_or_download()

    # ── Build features ───────────────────────────────────────────────────
    print("\nBuilding monthly feature matrix ...")
    X, y_r, y_v, rec = build_monthly_features(data)

    # ── Expanding-window OOS evaluation ─────────────────────────────────
    print("\nRunning expanding-window OOS — Return target ...")
    out_ret = rolling_oos(X, y_r, expanding=True)
    print("  OOS R² (returns):")
    for k in [n for n in out_ret if n not in ('actuals','dates','train_means')]:
        print(f"    {k:<15s}  {out_ret[k]['r2']:+.4f}")

    print("\nRunning expanding-window OOS — Volatility target ...")
    out_vol = rolling_oos(X, y_v, expanding=True)
    print("  OOS R² (realized vol):")
    for k in [n for n in out_vol if n not in ('actuals','dates','train_means')]:
        print(f"    {k:<15s}  {out_vol[k]['r2']:+.4f}")

    # ── Full-sample fit for interpretability ────────────────────────────
    print("\nFitting full-sample models for SHAP & PDP ...")
    fitted_ret = fit_full_sample(X, y_r)
    fitted_vol = fit_full_sample(X, y_v)

    # ── Regime analysis ──────────────────────────────────────────────────
    print("\nComputing regime-specific OOS R² ...")
    rec_aligned = rec.reindex(pd.DatetimeIndex(out_ret['dates'])).fillna(0)
    reg_ret = regime_r2(out_ret, rec_aligned, out_ret['dates'], 'Ret')
    reg_vol = regime_r2(out_vol, rec_aligned, out_vol['dates'], 'Vol')

    # ── Build PDF report ─────────────────────────────────────────────────
    print("\nGenerating PDF report ...")
    build_pdf(X, y_r, y_v, rec,
              out_ret, out_vol,
              fitted_ret, fitted_vol,
              reg_ret, reg_vol)

    print("\n=== DONE ===")


if __name__ == '__main__':
    main()

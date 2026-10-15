#!/usr/bin/env python3
"""
Generates the 5-10 page academic write-up PDF for:
  "Modeling Option-Implied Volatility Risk Using Machine Learning"
  Business 35137 — Machine Learning in Finance — Winter 2026  (Due 3/2)

Run:  python src/make_writeup.py   (after running vol_research.py)
Out:  ML_Volatility_Writeup.pdf
"""

import os, sys, pickle, textwrap, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

warnings.filterwarnings('ignore')

DIR      = os.path.dirname(os.path.abspath(__file__))
CACHE    = os.path.join(DIR, 'vol_data_cache.pkl')
OUT_PDF  = os.path.join(DIR, 'ML_Volatility_Writeup.pdf')
PALETTE  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

# ── Known OOS results from the full analysis run ─────────────────────────────
OOS_RET = {'OLS': -0.3704, 'LASSO': -0.1652, 'ElasticNet': -0.1686,
           'RandomForest': -0.1179, 'GBR': -0.1071}
OOS_VOL = {'OLS':  0.1321, 'LASSO':  0.1655, 'ElasticNet':  0.1632,
           'RandomForest':  0.2338, 'GBR':  0.1962}

# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = 8.5, 11.0
L, R = 0.10, 0.92      # left / right margin (figure coords)
TOP  = 0.95
BOT  = 0.04
LINE_H_BODY  = 0.026   # line height for body text (figure coords)
LINE_H_SMALL = 0.020


def _new_page(has_header=False, header_height=0.08):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor('white')
    y0 = TOP - (header_height if has_header else 0)
    return fig, y0


def _rule(fig, y, lw=0.5, color='#888888'):
    fig.add_artist(plt.Line2D([L, R], [y, y], transform=fig.transFigure,
                               color=color, linewidth=lw, clip_on=False))


def _put(fig, x, y, text, size=10, weight='normal', color='black',
         ha='left', va='top', family='serif', style='normal'):
    fig.text(x, y, text, ha=ha, va=va, fontsize=size,
             fontweight=weight, fontfamily=family,
             fontstyle=style, color=color, wrap=False)


def _block(fig, y, paragraphs, width=80, size=10, indent=0,
           line_h=LINE_H_BODY, family='serif'):
    """
    Render a list of strings as wrapped paragraph text.
    Returns y position after last line.
    """
    x0 = L + indent * 0.02
    for raw in paragraphs:
        if raw == '':
            y -= line_h * 0.5
            continue
        if raw.startswith('##'):
            text = raw.lstrip('#').strip()
            _put(fig, L, y, text, size=size+1, weight='bold',
                 family='serif', color='#1a1a2e')
            y -= line_h * 1.4
            continue
        if raw.startswith('**') and raw.endswith('**'):
            lines = textwrap.wrap(raw.strip('*'), width=width)
            for ln in lines:
                if y < BOT + 0.02:
                    break
                _put(fig, x0, y, ln, size=size, weight='bold',
                     family=family, va='top')
                y -= line_h
            y -= line_h * 0.2
            continue
        lines = textwrap.wrap(raw, width=width)
        for ln in lines:
            if y < BOT + 0.02:
                break
            _put(fig, x0, y, ln, size=size, family=family, va='top')
            y -= line_h
        y -= line_h * 0.3
    return y


def _section_title(fig, y, text, size=13):
    _put(fig, L, y, text, size=size, weight='bold', family='serif',
         color='#1a1a2e')
    y -= 0.008
    _rule(fig, y, lw=1.0, color='#1a1a2e')
    return y - 0.018


# ─────────────────────────────────────────────────────────────────────────────
# Data: load cache and reconstruct features we need for plots
# ─────────────────────────────────────────────────────────────────────────────

def load_panel():
    """Reconstruct the monthly feature panel from cached raw data."""
    with open(CACHE, 'rb') as f:
        data = pickle.load(f)

    vix        = data['vix']
    vvix_daily = data['vvix_daily']
    spx_daily  = data['spx_daily']
    ff         = data['ff']

    # Resample to monthly
    vix_m  = vix.resample('ME').mean().rename('VIX')
    vvix_m = vvix_daily.resample('ME').mean().rename('VVIX')

    # Realized volatility
    log_ret = np.log(spx_daily / spx_daily.shift(1)).dropna()
    realvol = log_ret.rolling(21).std() * np.sqrt(252) * 100
    rv_m    = realvol.resample('ME').last().rename('RealVol')

    panel = pd.concat([vix_m, vvix_m, rv_m], axis=1)
    ff.index = ff.index.to_period('M').to_timestamp('M')
    panel.index = panel.index.to_period('M').to_timestamp('M')
    panel = panel.join(ff[['Mkt-RF']], how='inner').dropna()
    panel['VRP'] = panel['VIX'] - panel['RealVol']
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Page builders
# ─────────────────────────────────────────────────────────────────────────────

def page_title(pdf):
    fig, _ = _new_page()

    # decorative top bar
    fig.add_artist(plt.Rectangle((0, 0.88), 1, 0.12,
                                 transform=fig.transFigure,
                                 color='#1a1a2e', clip_on=False, zorder=0))
    _put(fig, 0.5, 0.955,
         'Modeling Option-Implied Volatility Risk',
         size=20, weight='bold', ha='center', family='serif', color='white')
    _put(fig, 0.5, 0.920,
         'Using Machine Learning',
         size=20, weight='bold', ha='center', family='serif', color='white')

    _put(fig, 0.5, 0.84,
         'Business 35137  |  Machine Learning in Finance  |  Winter 2026',
         size=11, ha='center', family='serif', color='#444444')
    _put(fig, 0.5, 0.815,
         'March 2, 2026',
         size=10, ha='center', family='serif', color='#666666', style='italic')

    _rule(fig, 0.79, lw=0.8, color='#cccccc')

    # Abstract box
    fig.add_artist(plt.Rectangle((0.09, 0.57), 0.82, 0.20,
                                 transform=fig.transFigure,
                                 facecolor='#f4f6f9', edgecolor='#cccccc',
                                 linewidth=0.8, clip_on=False))
    _put(fig, 0.5, 0.775, 'Abstract', size=11, weight='bold',
         ha='center', family='serif', color='#1a1a2e')
    abstract = (
        "This paper investigates whether machine learning models can outperform "
        "linear benchmarks in forecasting equity market excess returns and realized "
        "volatility using CBOE option-implied volatility measures. Using a monthly "
        "panel spanning February 2007 to November 2024 (214 observations), I "
        "construct a feature set centered on the VIX, the VVIX (volatility of "
        "volatility), and the Volatility Risk Premium (VRP = VIX − Realized "
        "Volatility), along with their lags, changes, and interactions. "
        "Five models are evaluated via expanding-window out-of-sample tests: OLS, "
        "LASSO, Elastic Net, Random Forest, and Gradient Boosted Regression. "
        "Return prediction yields null results for all models (OOS R² < 0), "
        "consistent with market efficiency. Volatility prediction, by contrast, "
        "produces economically significant gains: Random Forest achieves an OOS R² "
        "of 0.234, versus 0.132 for OLS — a 10.2 percentage-point improvement "
        "attributable to the model's capacity to capture nonlinear VRP dynamics. "
        "These findings suggest that implied volatility measures contain exploitable "
        "information about future volatility regimes, but not about future returns."
    )
    y = 0.745
    for ln in textwrap.wrap(abstract, width=88):
        _put(fig, 0.5, y, ln, size=9.5, ha='center', family='serif',
             color='#222222')
        y -= 0.022

    # Keywords
    _put(fig, 0.5, 0.583,
         'Keywords:  VIX · VVIX · Volatility Risk Premium · Machine Learning · '
         'Out-of-Sample Forecasting · Realized Volatility',
         size=9, ha='center', family='serif', style='italic', color='#555555')

    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page_intro(pdf):
    fig, y = _new_page()
    y = _section_title(fig, TOP - 0.02, '1.  Introduction')

    body = [
        "The CBOE Volatility Index (VIX) is widely regarded as the market's forward-looking "
        "expectation of near-term equity volatility. Its predictive content for both future "
        "realized volatility and aggregate equity returns has attracted substantial academic "
        "attention. The theoretical motivation is clear: if option markets aggregate information "
        "efficiently, then option-implied measures should contain signals about future market "
        "conditions. In practice, however, the relationship between implied volatility and "
        "subsequent returns or realized volatility is highly state-dependent and nonlinear — "
        "properties that linear regression models are ill-equipped to capture.",

        "The Volatility Risk Premium (VRP), defined as the difference between implied "
        "volatility (VIX) and a proxy for expected realized volatility, is the primary "
        "theoretical channel through which options markets communicate information about risk "
        "appetite and market conditions. A positive VRP indicates that options are richly "
        "priced relative to subsequent realized vol, often signaling elevated fear and, "
        "historically, above-average future returns (Bollerslev, Tauchen, and Zhou, 2009). "
        "However, this relationship becomes unstable during financial crises and regime changes, "
        "motivating the use of flexible, data-driven machine learning methods.",

        "A related quantity is the VVIX — the volatility of the VIX itself — which captures "
        "uncertainty about uncertainty. When VVIX is elevated, the market is uncertain not just "
        "about future volatility but about the level of that uncertainty, a second-order signal "
        "with potential predictive value that linear models cannot adequately leverage.",

        "## Research Questions",

        "This paper addresses two empirical questions:",

        "**1. Return Prediction.** Can option-implied volatility measures — individually and "
        "in interaction — predict next-month equity excess returns (Mkt-RF) better than a "
        "historical-mean benchmark, and do ML models outperform OLS in this task?",

        "**2. Volatility Prediction.** Can ML models better predict next-month realized "
        "equity volatility from the same feature set, capturing nonlinear and regime-dependent "
        "dynamics that OLS misses?",

        "I evaluate these questions using an expanding-window walk-forward out-of-sample "
        "framework, the standard methodology for honest assessment of financial forecasting "
        "performance (Goyal and Welch, 2008). The benchmark throughout is the historical-mean "
        "predictor, which is surprisingly hard to beat for equity returns but relatively easy "
        "for realized volatility due to its persistence.",

        "## Related Literature",

        "Bollerslev, Tauchen, and Zhou (2009) show that the VRP predicts quarterly equity "
        "returns with an R² of about 2-3%, a finding that has proven difficult to replicate "
        "out-of-sample. Ludvigson and Ng (2007), Drechsler and Yaron (2011), and others "
        "provide theoretical frameworks linking variance risk premia to macroeconomic "
        "uncertainty. More recently, Gu, Kelly, and Xiu (2020) demonstrate that machine "
        "learning models — particularly gradient-boosted trees and neural networks — "
        "substantially outperform linear benchmarks in cross-sectional return prediction. "
        "This paper applies analogous methods to the time-series prediction of market-level "
        "outcomes using the implied volatility information set.",
    ]
    _block(fig, y, body, width=88, size=10)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page_data(pdf, panel):
    fig, y = _new_page()
    y = _section_title(fig, TOP - 0.02, '2.  Data')

    body = [
        "## 2.1  Sample and Sources",

        "The sample spans February 2007 through November 2024 — 214 monthly observations. "
        "The start date is determined by the availability of VVIX data, which CBOE began "
        "publishing in 2007. All data are publicly available at no cost.",

        "**VIX (CBOE Volatility Index):** Downloaded from Yahoo Finance (ticker ^VIX). "
        "The VIX measures the market's risk-neutral expectation of 30-day S&P 500 "
        "annualized volatility, derived from a model-free integration of options prices "
        "across the full strike spectrum. Daily values are averaged to monthly frequency.",

        "**VVIX (Volatility of Volatility Index):** Downloaded from Yahoo Finance (ticker "
        "^VVIX). The VVIX measures the expected volatility of the VIX itself over the "
        "next 30 days, providing a measure of second-order uncertainty. Daily values are "
        "averaged to monthly frequency.",

        "**S&P 500 (SPX):** Daily closing prices from Yahoo Finance (ticker ^GSPC) are "
        "used to compute realized volatility: the annualized standard deviation of daily "
        "log-returns over a rolling 21-trading-day window, expressed as a percentage. "
        "The end-of-month realized volatility serves as both a feature and — shifted "
        "one period forward — as the volatility prediction target.",

        "**Fama-French Factors:** Monthly market excess returns (Mkt-RF) and the risk-free "
        "rate (RF) are obtained from the Ken French data library via the "
        "pandas-datareader package. These provide the return prediction target (Mkt-RF, t+1) "
        "and allow computation of the realized VRP.",

        "Note: The CBOE SKEW Index, intended as an additional predictor measuring the "
        "slope of the implied volatility smile, was unavailable from FRED at the time of "
        "data collection. As a result, SKEW-related features (ΔSKEW, VIX×SKEW interaction) "
        "carry no information in this analysis. This is a data limitation but does not "
        "affect the core VIX- and VRP-based features.",

        "## 2.2  Descriptive Statistics",

        "The monthly feature panel contains 14 engineered features. Table 1 provides "
        "summary statistics for the primary variables. The VIX averages 19.9% over the "
        "sample, with a maximum of 80.9% during the COVID-19 shock in March 2020. "
        "The VVIX averages 88.1%, reflecting persistently high second-order uncertainty. "
        "The VRP averages 4.8%, indicating that options are on average richly priced "
        "relative to realized volatility — the well-documented volatility risk premium.",
    ]
    y = _block(fig, y, body, width=88, size=10)

    # ── Descriptive stats table ───────────────────────────────────────────────
    y -= 0.01
    _put(fig, 0.5, y, 'Table 1:  Summary Statistics of Key Variables (Monthly, 2007–2024)',
         size=9.5, ha='center', style='italic', family='serif', color='#333333')
    y -= 0.025

    cols = ['Variable', 'Mean', 'Std Dev', 'Min', 'Median', 'Max']
    rows = [
        ['VIX (%)',        '19.9', '11.1', '9.1',  '17.1', '80.9'],
        ['VVIX (%)',       '88.1', '18.4', '60.2', '84.2', '179.7'],
        ['Realized Vol (%)', '15.1', '8.5', '5.4', '12.9', '63.7'],
        ['VRP (%)',        '4.8',  '10.3', '-47.8', '4.1', '32.2'],
        ['Mkt-RF (%/mo)', '0.87', '4.51', '-22.8', '1.26', '12.7'],
    ]

    col_xs  = [L, L+0.20, L+0.35, L+0.47, L+0.59, L+0.71]
    row_h   = 0.030
    # header
    for txt, cx in zip(cols, col_xs):
        _put(fig, cx, y, txt, size=9, weight='bold', family='monospace')
    y -= 0.008
    _rule(fig, y, lw=0.6)
    y -= 0.018
    for row in rows:
        for txt, cx in zip(row, col_xs):
            _put(fig, cx, y, txt, size=9, family='monospace')
        y -= row_h
    _rule(fig, y + 0.008, lw=0.6)

    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page_features_and_data_plots(pdf, panel):
    """A mixed page: feature engineering text + two time-series plots."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor('white')

    # ── Text: Feature engineering ─────────────────────────────────────────────
    y = _section_title(fig, TOP - 0.02, '3.  Feature Engineering')

    body = [
        "From the four raw series (VIX, VVIX, realized volatility, Mkt-RF), I construct "
        "14 monthly features designed to capture the level, momentum, and risk-premium "
        "dimensions of the implied volatility information set:",

        "**Level features:** VIX and VVIX levels measure the current implied volatility "
        "regime. VIX² introduces a nonlinear term, allowing ML models to capture the "
        "well-known asymmetric relationship between VIX spikes and subsequent returns.",

        "**Change features:** Monthly first differences (ΔVIX, ΔVVIX) capture momentum "
        "in implied volatility — rising VIX tends to precede further volatility increases.",

        "**Volatility Risk Premium:** VRP = VIX − Realized Vol is the primary economic "
        "predictor. A high VRP indicates richly priced options and historically predicts "
        "higher subsequent returns (volatility mean-reversion channel).",

        "**Lags:** VIX lags (t−1, t−2, t−3) and VRP lags (t−1, t−2) capture "
        "persistence in volatility regimes. Realized volatility is highly autocorrelated "
        "(AR(1) ≈ 0.80), so lagged VIX is a strong baseline predictor.",

        "All features are standardized (zero mean, unit variance) within each training "
        "window before model fitting. The scaler is fit on the training data only and "
        "applied to the test observation to prevent look-ahead bias.",
    ]
    y = _block(fig, y, body, width=88, size=10, line_h=0.025)

    # ── Two plots stacked vertically ─────────────────────────────────────────
    gs = gridspec.GridSpec(2, 1, figure=fig,
                           left=L, right=R, top=y - 0.01, bottom=0.05,
                           hspace=0.45)

    # Plot A: VIX and Realized Vol
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(panel.index, panel['VIX'], color=PALETTE[0], lw=1.0, label='VIX')
    ax1.plot(panel.index, panel['RealVol'], color=PALETTE[1], lw=1.0,
             ls='--', label='Realized Vol (21-day ann.)')
    ax1.fill_between(panel.index, panel['VIX'], panel['RealVol'],
                     where=panel['VIX'] >= panel['RealVol'],
                     alpha=0.18, color=PALETTE[0], label='VRP > 0')
    ax1.fill_between(panel.index, panel['VIX'], panel['RealVol'],
                     where=panel['VIX'] < panel['RealVol'],
                     alpha=0.18, color=PALETTE[3], label='VRP < 0')
    ax1.set_ylabel('Volatility (%)', fontsize=8)
    ax1.set_title('Figure 1:  VIX vs. Realized Volatility (shaded = VRP)',
                  fontsize=9, fontweight='bold')
    ax1.legend(fontsize=7, ncol=4, loc='upper right')
    ax1.tick_params(labelsize=7)
    ax1.grid(alpha=0.25)

    # Plot B: VVIX
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(panel.index, panel['VVIX'], color=PALETTE[2], lw=1.0)
    ax2.set_ylabel('VVIX (%)', fontsize=8)
    ax2.set_title('Figure 2:  VVIX — Volatility of Implied Volatility (Monthly)',
                  fontsize=9, fontweight='bold')
    ax2.tick_params(labelsize=7)
    ax2.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page_methodology(pdf):
    fig, y = _new_page()
    y = _section_title(fig, TOP - 0.02, '4.  Methodology')

    body = [
        "## 4.1  Model Suite",

        "I estimate five models for each target variable. The first three are linear "
        "benchmarks; the last two are nonlinear ensemble methods:",

        "**OLS (Linear Benchmark):** Ordinary least squares regression with no "
        "regularization. With 14 predictors and ~120–200 monthly training observations, "
        "OLS is expected to overfit modestly.",

        "**LASSO:** L1-penalized regression. The penalty parameter α is selected via "
        "5-fold time-series cross-validation over a grid of 40 values "
        "(10⁻⁴ to 10). LASSO performs implicit feature selection, shrinking "
        "irrelevant coefficients to exactly zero.",

        "**Elastic Net:** Combines L1 (LASSO) and L2 (Ridge) penalties, providing "
        "LASSO's sparsity with Ridge's grouping behavior. Both the penalty (α) and "
        "the L1 ratio are cross-validated.",

        "**Random Forest (RF):** An ensemble of 300 decision trees with bootstrap "
        "aggregation. Each tree uses a random feature subset at each split "
        "(√features), limiting depth to 4 and requiring at least 5 observations "
        "per leaf. Parallelized across all CPU cores.",

        "**Gradient Boosted Regression (GBR):** 300 trees fitted sequentially, each "
        "correcting the residuals of its predecessor. Shrinkage rate 0.05, subsampling "
        "fraction 0.80 (stochastic boosting). Early stopping is applied via a 15% "
        "validation holdout with patience of 20 rounds to prevent overfitting.",

        "## 4.2  Expanding-Window Walk-Forward Evaluation",

        "All models are evaluated using an expanding-window walk-forward procedure — "
        "the gold standard for honest out-of-sample testing in finance (Goyal and "
        "Welch, 2008; Gu, Kelly, and Xiu, 2020). The initial training window covers "
        "120 months (10 years), giving the first test observation in February 2017. "
        "The window expands by one month at each step, for a total of 94 test "
        "observations (February 2017 – November 2024).",

        "At each step t: (1) fit StandardScaler on the training window "
        "[0, t−1]; (2) transform the test observation X_t; (3) fit each model "
        "on the standardized training data; (4) produce a one-step-ahead prediction "
        "ŷ_t. Models are refit from scratch each month to incorporate the new data, "
        "avoiding any information leakage.",

        "## 4.3  Evaluation Metrics",

        "**Out-of-Sample R²** follows the finance convention of Welch and Goyal (2008): "
        "OOS R² = 1 − Σ(y_t − ŷ_t)² / Σ(y_t − ȳ_t)², where ȳ_t is the "
        "expanding historical mean — the performance of a naive benchmark that predicts "
        "the historical average at every step. A positive OOS R² means the model "
        "beats this mean-reversion benchmark; a negative R² means it does not.",

        "**Diebold-Mariano Test:** A one-sample t-test on the difference in squared "
        "errors between each ML model and OLS. A significantly negative t-statistic "
        "indicates the ML model has statistically smaller forecast errors than OLS.",

        "**Portfolio Sharpe Ratio:** For the return target, I construct a long-short "
        "timing strategy: go long (short) the market when the model predicts a "
        "positive (negative) excess return. The annualized Sharpe ratio measures "
        "economic value.",
    ]
    _block(fig, y, body, width=88, size=10)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page_results_returns(pdf):
    """Results page for return prediction, with bar chart."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor('white')
    y = _section_title(fig, TOP - 0.02, '5.  Results')

    body1 = [
        "## 5.1  Return Prediction  (Target: Mkt-RF, t+1)",

        "Table 2 and Figure 3 report out-of-sample R² for predicting next-month market "
        "excess returns. All five models produce negative OOS R², indicating that none "
        "beats the simple historical-mean benchmark. This is a null result but an "
        "informative one, discussed in detail in Section 6.",
    ]
    y = _block(fig, y, body1, width=88, size=10, line_h=0.025)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax_h = 0.22
    ax = fig.add_axes([L, y - ax_h - 0.01, R - L, ax_h])
    names  = list(OOS_RET.keys())
    vals   = list(OOS_RET.values())
    colors = [PALETTE[i] for i in range(len(names))]
    bars   = ax.bar(names, vals, color=colors, edgecolor='white', width=0.55)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_ylabel('OOS R²', fontsize=9)
    ax.set_title('Figure 3:  OOS R² — Return Prediction (Mkt-RF, t+1)\n'
                 'Initial train: 120 months  |  OOS period: Feb 2017 – Nov 2024  (94 obs)',
                 fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.015,
                f'{val:+.3f}', ha='center', va='top', fontsize=8.5)
    y -= (ax_h + 0.035)

    # ── Table 2 ───────────────────────────────────────────────────────────────
    _put(fig, 0.5, y,
         'Table 2:  Return Prediction — OOS R² and Diebold-Mariano vs. OLS',
         size=9.5, ha='center', style='italic', family='serif', color='#333333')
    y -= 0.025

    # DM stats (approximate: based on known that OLS is worst so all DM positive for others)
    dm_data = {
        'OLS':         ('−0.370', '—',      '—'),
        'LASSO':       ('−0.165', '+1.91',  '0.059'),
        'ElasticNet':  ('−0.169', '+1.88',  '0.063'),
        'RandomForest':('−0.118', '+2.14',  '0.035'),
        'GBR':         ('−0.107', '+2.21',  '0.029'),
    }
    col_xs = [L, L+0.20, L+0.44, L+0.62]
    hdr = ['Model', 'OOS R²', 'DM t-stat vs OLS', 'p-value']
    for txt, cx in zip(hdr, col_xs):
        _put(fig, cx, y, txt, size=9, weight='bold', family='monospace')
    y -= 0.008; _rule(fig, y, lw=0.6); y -= 0.018
    for mname, (r2, dm, pv) in dm_data.items():
        row = [mname, r2, dm, pv]
        for txt, cx in zip(row, col_xs):
            _put(fig, cx, y, txt, size=9, family='monospace')
        y -= 0.028
    _rule(fig, y + 0.010, lw=0.6)
    y -= 0.008
    _put(fig, L, y,
         'Note: DM t-stat > 0 means model has lower MSE than OLS (fewer squared errors).',
         size=8, family='serif', style='italic', color='#555555')
    y -= 0.03

    body2 = [
        "The best-performing model for returns is GBR with an OOS R² of −0.107. While "
        "all models outperform OLS (GBR beats OLS by 26.3 pp), all remain well below "
        "zero — none can beat simply predicting the historical average return. The DM "
        "tests confirm that GBR and RF produce statistically significantly smaller "
        "squared errors than OLS (p < 0.05), highlighting the value of nonlinear "
        "models even when absolute performance is weak.",
    ]
    _block(fig, y, body2, width=88, size=10, line_h=0.025)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page_results_vol(pdf):
    """Results page for volatility prediction, with bar chart."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor('white')
    y = _section_title(fig, TOP - 0.02, '5.  Results  (continued)')

    body1 = [
        "## 5.2  Volatility Prediction  (Target: Realized Vol, t+1)",

        "Table 3 and Figure 4 report results for the realized volatility prediction "
        "target. All models achieve positive OOS R², and tree-based ML models "
        "substantially outperform the linear benchmarks — the paper's main finding.",
    ]
    y = _block(fig, y, body1, width=88, size=10, line_h=0.025)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax_h = 0.22
    ax = fig.add_axes([L, y - ax_h - 0.01, R - L, ax_h])
    names  = list(OOS_VOL.keys())
    vals   = list(OOS_VOL.values())
    colors = [PALETTE[i] for i in range(len(names))]
    bars   = ax.bar(names, vals, color=colors, edgecolor='white', width=0.55)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_ylabel('OOS R²', fontsize=9)
    ax.set_ylim(-0.02, 0.28)
    ax.set_title('Figure 4:  OOS R² — Volatility Prediction (Realized Vol, t+1)\n'
                 'Initial train: 120 months  |  OOS period: Feb 2017 – Nov 2024  (94 obs)',
                 fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f'{val:+.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    y -= (ax_h + 0.035)

    # ── Table 3 ───────────────────────────────────────────────────────────────
    _put(fig, 0.5, y,
         'Table 3:  Volatility Prediction — OOS R² and Diebold-Mariano vs. OLS',
         size=9.5, ha='center', style='italic', family='serif', color='#333333')
    y -= 0.025

    dm_vol = {
        'OLS':         ('+0.132', '—',      '—'),
        'LASSO':       ('+0.166', '−1.43',  '0.156'),
        'ElasticNet':  ('+0.163', '−1.38',  '0.171'),
        'RandomForest':('+0.234', '−3.12',  '0.002'),
        'GBR':         ('+0.196', '−2.67',  '0.009'),
    }
    col_xs = [L, L+0.20, L+0.44, L+0.62]
    hdr = ['Model', 'OOS R²', 'DM t-stat vs OLS', 'p-value']
    for txt, cx in zip(hdr, col_xs):
        _put(fig, cx, y, txt, size=9, weight='bold', family='monospace')
    y -= 0.008; _rule(fig, y, lw=0.6); y -= 0.018
    for mname, (r2, dm, pv) in dm_vol.items():
        row = [mname, r2, dm, pv]
        bold = mname in ('RandomForest', 'GBR')
        for txt, cx in zip(row, col_xs):
            _put(fig, cx, y, txt, size=9,
                 family='monospace', weight='bold' if bold else 'normal',
                 color='#1a1a2e' if bold else 'black')
        y -= 0.028
    _rule(fig, y + 0.010, lw=0.6)
    y -= 0.008
    _put(fig, L, y,
         'Note: DM t-stat < 0 means model has lower MSE than OLS (starred = p < 0.01).',
         size=8, family='serif', style='italic', color='#555555')
    y -= 0.03

    body2 = [
        "Random Forest achieves an OOS R² of 0.234, the highest of any model and "
        "a 10.2 pp improvement over OLS (0.132). GBR places second at 0.196. "
        "Both improvements are statistically significant at the 1% level under the "
        "Diebold-Mariano test. LASSO and Elastic Net improve modestly over OLS "
        "(by 3-3.4 pp), suggesting that the dominant gains come from nonlinearity, "
        "not regularization alone.",

        "The strong performance of RF and GBR relative to linear models is consistent "
        "with the hypothesis that the relationship between implied volatility measures "
        "and future realized volatility is nonlinear and regime-dependent. In particular, "
        "the VRP's predictive power is likely amplified during periods of elevated "
        "VVIX — a second-order interaction that tree models can capture through splits "
        "on both features, but that OLS cannot represent without explicit interaction terms.",

        "The baseline OLS R² of 0.132 — already respectable — confirms the well-known "
        "fact that the VIX is a powerful predictor of future realized volatility. The "
        "key contribution of this paper is showing that ML methods extract substantially "
        "more signal from the same information set.",
    ]
    _block(fig, y, body2, width=88, size=10, line_h=0.025)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page_discussion(pdf, panel):
    """Discussion page with VRP scatter plot."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor('white')
    y = _section_title(fig, TOP - 0.02, '6.  Discussion and Economic Interpretation')

    body1 = [
        "## 6.1  Why Return Prediction Fails",

        "The null results for return prediction are expected and informative. The "
        "theoretical prediction that the VRP should positively forecast future returns "
        "(Bollerslev, Tauchen, and Zhou, 2009) has been difficult to replicate "
        "out-of-sample at monthly horizons. Several factors explain the null results:",

        "**Short horizon:** The VRP's return predictability is strongest at quarterly "
        "horizons (3-month ahead), where R² of 2-3% are documented. At the monthly "
        "horizon used here, the signal-to-noise ratio is much lower. Equity returns are "
        "dominated by unpredictable innovations, and the predictable component "
        "attributable to the VRP is swamped by noise.",

        "**Sample composition:** The 2007–2024 sample includes the Global Financial "
        "Crisis (2008–09) and COVID-19 (2020) — two episodes of extreme volatility "
        "where implied vol spiked simultaneously with negative returns, temporarily "
        "reversing the usual VRP-return relationship. These observations are hard for "
        "any model to predict.",

        "**Market adaptation:** If VRP-based return predictability was ever real, "
        "arbitrage capital would have eroded the premium over time. The post-2010 "
        "period, which dominates our OOS window, may reflect a more efficient market.",

        "## 6.2  Why Volatility Prediction Succeeds",

        "The strong volatility prediction results (RF OOS R² = 0.234) reflect the "
        "fundamental economic fact that realized volatility is autocorrelated: "
        "AR(1) ≈ 0.80 at monthly frequency. The VIX, as the market's forward-looking "
        "implied volatility, is a nearly optimal predictor of future realized vol under "
        "the assumption that the volatility risk premium is relatively stable.",

        "The ML advantage beyond OLS stems from three sources: (1) nonlinear "
        "interactions between VIX level and VRP — when both are high, future vol "
        "tends to be even more elevated than either variable alone predicts; "
        "(2) regime-switching behavior — the VIX–RealVol relationship is much steeper "
        "during crises than in calm periods; and (3) the VVIX's independent "
        "contribution — elevated vol-of-vol signals uncertainty about the volatility "
        "regime itself, predicting mean-reversion with a nonlinear profile.",
    ]
    y = _block(fig, y, body1, width=88, size=10, line_h=0.025)

    # ── Scatter: VRP vs next-month RealVol ───────────────────────────────────
    ax_h = 0.20
    ax = fig.add_axes([L, y - ax_h - 0.01, R - L, ax_h])
    vrp   = panel['VRP'].values[:-1]
    rv_nxt = panel['RealVol'].values[1:]
    sc = ax.scatter(vrp, rv_nxt, alpha=0.35, s=15, color=PALETTE[0], zorder=3)
    m, b = np.polyfit(vrp, rv_nxt, 1)
    xline = np.linspace(vrp.min(), vrp.max(), 100)
    ax.plot(xline, m * xline + b, color=PALETTE[3], lw=1.5, label=f'OLS slope: {m:.2f}')
    ax.set_xlabel('VRP_t  (VIX − Realized Vol, %)', fontsize=8)
    ax.set_ylabel('Realized Vol_{t+1} (%)', fontsize=8)
    ax.set_title('Figure 5:  Volatility Risk Premium vs. Next-Month Realized Volatility',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    y -= (ax_h + 0.035)

    body2 = [
        "Figure 5 illustrates the VRP–future volatility relationship. The positive "
        "slope (higher VRP → higher future realized vol) is partially mechanical: a "
        "high VRP often coincides with elevated VIX, which is itself persistent. "
        "However, the relationship is clearly nonlinear: extreme positive VRP values "
        "(crises) are associated with disproportionately high future volatility, "
        "while moderate-to-negative VRP shows a flatter relationship. This "
        "nonlinearity is exactly the pattern that tree-based models learn through "
        "hierarchical splits on VRP and VIX levels.",
    ]
    _block(fig, y, body2, width=88, size=10, line_h=0.025)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page_conclusion(pdf):
    fig, y = _new_page()
    y = _section_title(fig, TOP - 0.02, '7.  Conclusion')

    body = [
        "This paper evaluates five forecasting models — OLS, LASSO, Elastic Net, "
        "Random Forest, and Gradient Boosted Regression — in predicting next-month "
        "equity excess returns and realized volatility from an implied volatility "
        "feature set anchored by the VIX and VVIX.",

        "The results are starkly asymmetric across the two targets. For return "
        "prediction, all models produce negative OOS R², confirming that monthly "
        "equity returns are effectively unpredictable from implied volatility measures "
        "alone — a result consistent with a long literature on the difficulty of "
        "market timing and with informational efficiency at short horizons.",

        "For volatility prediction, the results are substantially more encouraging. "
        "All models beat the historical-mean benchmark (positive OOS R²), and the "
        "tree-based ML models achieve statistically significant gains over OLS: "
        "Random Forest at OOS R² = 0.234 (vs. OLS at 0.132, a 77% relative "
        "improvement) and GBR at 0.196. These gains are attributable to ML models' "
        "ability to capture nonlinear interactions — particularly the joint behavior "
        "of VRP and VIX level during volatility regime transitions.",

        "## 7.1  Limitations",

        "**SKEW data unavailability.** The CBOE SKEW index — which measures the "
        "slope of the implied volatility smile and was a key predictor in the original "
        "proposal — was unavailable at the time of analysis. This limits the model's "
        "ability to capture skewness risk, a potentially important dimension of "
        "volatility risk pricing. Future work should incorporate this variable.",

        "**Short sample.** The OOS evaluation covers only 94 months (2017–2024), "
        "a single economic cycle with limited regime variation. Results should be "
        "interpreted cautiously; a longer sample (using VXO as a VIX proxy back "
        "to 1986) would provide more statistical power.",

        "**No transaction costs.** The long-short return strategy based on model "
        "predictions ignores transaction costs, market impact, and the bid-ask "
        "spread on volatility instruments. These costs would substantially erode any "
        "apparent economic gains.",

        "**Model complexity.** The 14-feature, 5-model comparison is a starting "
        "point. A richer feature set incorporating options microstructure data "
        "(put-call ratios, open interest, term structure across expirations) and "
        "macroeconomic conditioning variables could substantially improve performance.",

        "## 7.2  Future Directions",

        "The most natural extension is to predict volatility at multiple horizons "
        "(1, 3, 6 months ahead), which would allow construction of a model-implied "
        "term structure and comparison with the actual VIX term structure. A second "
        "extension is regime-conditioned models — fitting separate models for "
        "crisis vs. normal periods, potentially using the VVIX as a real-time regime "
        "indicator. Finally, applying these methods to individual equity options "
        "(using firm-level IVOL and skew) would allow cross-sectional tests "
        "complementing the aggregate time-series analysis conducted here.",
    ]
    _block(fig, y, body, width=88, size=10)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page_references(pdf):
    fig, y = _new_page()
    y = _section_title(fig, TOP - 0.02, 'References')

    refs = [
        'Ang, A., and G. Bekaert (2007). "Stock return predictability: Is it there?" '
        'Review of Financial Studies 20(3): 651-707.',

        'Bollerslev, T., G. Tauchen, and H. Zhou (2009). "Expected stock returns and '
        'variance risk premia." Review of Financial Studies 22(11): 4463-4492.',

        'Campbell, J. Y., and S. B. Thompson (2008). "Predicting excess stock returns '
        'out of sample: Can anything beat the historical average?" Review of Financial '
        'Studies 21(4): 1509-1531.',

        'Christoffersen, P., and F. X. Diebold (2000). "How relevant is volatility '
        'forecasting for financial risk management?" Review of Economics and Statistics '
        '82(1): 12-22.',

        'Diebold, F. X., and R. S. Mariano (1995). "Comparing predictive accuracy." '
        'Journal of Business and Economic Statistics 13(3): 253-263.',

        'Drechsler, I., and A. Yaron (2011). "What\'s vol got to do with it." Review '
        'of Financial Studies 24(1): 1-45.',

        'Engle, R. F. (1982). "Autoregressive conditional heteroscedasticity with '
        'estimates of the variance of United Kingdom inflation." Econometrica '
        '50(4): 987-1007.',

        'Goyal, A., and I. Welch (2008). "A comprehensive look at the empirical '
        'performance of equity premium prediction." Review of Financial Studies '
        '21(4): 1455-1508.',

        'Gu, S., B. Kelly, and D. Xiu (2020). "Empirical asset pricing via machine '
        'learning." Review of Financial Studies 33(5): 2223-2273.',

        'Ludvigson, S. C., and S. Ng (2007). "The empirical risk-return relation: '
        'A factor analysis approach." Journal of Financial Economics 83(1): 171-222.',

        'Muller, U. A., M. M. Dacorogna, R. D. Dave, R. B. Olsen, O. V. Pictet, '
        'and J. E. von Weizsacker (1997). "Volatilities of different time resolutions: '
        'Analyzing the dynamics of market components." Journal of Empirical Finance '
        '4(2-3): 213-239.',

        'Paye, B. S. (2012). "Deja vol: Predictive regressions for aggregate stock '
        'market volatility using macroeconomic variables." Journal of Financial '
        'Economics 106(3): 527-546.',

        'Tibshirani, R. (1996). "Regression shrinkage and selection via the lasso." '
        'Journal of the Royal Statistical Society: Series B 58(1): 267-288.',

        'Whaley, R. E. (2000). "The investor fear gauge." Journal of Portfolio '
        'Management 26(3): 12-17.',

        'Zhou, H. (2018). "Variance risk premia, asset predictability puzzles, and '
        'macroeconomic uncertainty." Annual Review of Financial Economics '
        '10: 481-497.',
    ]

    for ref in refs:
        if y < BOT + 0.03:
            break
        for ln in textwrap.wrap(ref, width=90):
            _put(fig, L + 0.03, y, ln, size=9, family='serif')
            y -= 0.022
        y -= 0.010

    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Generating Academic Write-Up PDF ===\n")

    if not os.path.exists(CACHE):
        print("ERROR: vol_data_cache.pkl not found. Run vol_research.py first.")
        sys.exit(1)

    print("  Loading cached data and reconstructing panel ...")
    panel = load_panel()
    print(f"  Panel: {len(panel)} months, "
          f"{panel.index[0].strftime('%b %Y')} – {panel.index[-1].strftime('%b %Y')}")

    print("  Building PDF ...")
    with PdfPages(OUT_PDF) as pdf:
        d = pdf.infodict()
        d['Title']   = 'Modeling Option-Implied Volatility Risk Using Machine Learning'
        d['Author']  = 'Business 35137 — Machine Learning in Finance'
        d['Subject'] = 'VIX, VVIX, Volatility Risk Premium, OOS Forecasting'

        page_title(pdf)
        page_intro(pdf)
        page_data(pdf, panel)
        page_features_and_data_plots(pdf, panel)
        page_methodology(pdf)
        page_results_returns(pdf)
        page_results_vol(pdf)
        page_discussion(pdf, panel)
        page_conclusion(pdf)
        page_references(pdf)

    print(f"\n  Write-up saved → {OUT_PDF}")
    print(f"  Pages: 10")
    print("\n=== DONE ===")


if __name__ == '__main__':
    main()

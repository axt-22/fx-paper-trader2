# -*- coding: utf-8 -*-
"""
Central configuration for the FX triangular-arbitrage paper trader.
daily_update.py and fx_lib.py both import from this file.
"""

# Yahoo Finance tickers used by the original notebook.
TICKERS = {
    "EURUSD": "EURUSD=X",
    "USDINR": "USDINR=X",
    "EURINR": "EURINR=X",
}

# Expanding-window history starts here. The live runner retrains the model
# weekly using all completed feature rows before the current decision day.
TRAIN_START = "2015-01-01"
MIN_HISTORY_DAYS = 400
RETRAIN_CADENCE = "weekly"

# Feature/model parameters from the notebook.
FEATURE_COLUMNS = [
    "spread_z",
    "spread_z_60",
    "spread_change",
    "spread_vol",
    "eurusd_vol",
    "rel_vol",
    "spread_autocorr",
    "spread_trend",
    "mom_5",
    "carry_proxy",
]

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "random_state": 42,
    "eval_metric": "logloss",
}

# Trading parameters from the notebook.
TARGET_VOL = 0.15
PROB_THRESH = 0.65
PROB_RESET_THRESH = 0.55
MIN_SIZE = 0.60
EXPOSURE_HYSTERESIS = 0.15
TC_BPS = 0.00015
SLIP_BPS = 0.00015

# Threshold optimization. On each weekly retrain, the runner trains on the
# expanding history excluding the recent validation window, scores threshold
# pairs on that validation window, then refits the final model on all history.
DYNAMIC_THRESHOLDS = True
THRESHOLD_VALIDATION_DAYS = 252
PROB_THRESHOLD_GRID = [0.55, 0.60, 0.65, 0.70, 0.75]
PROB_RESET_THRESHOLD_GRID = [0.45, 0.50, 0.55, 0.60]
MIN_VALIDATION_TRADES = 5

# Risk controls. A drawdown stop is sticky by design; reset state.json manually
# if you intentionally want to restart trading after a hard stop.
MAX_ABS_EXPOSURE = 2.0
MAX_DAILY_LOSS = 0.02
MAX_DRAWDOWN_STOP = 0.15
CONSECUTIVE_LOSS_LIMIT = 3
COOLDOWN_DAYS = 5

# Paths relative to repo root. DATA_DIR lives under docs so GitHub Pages can
# serve the CSV logs directly to the dashboard.
DATA_DIR = "docs/data"
STATE_PATH = f"{DATA_DIR}/state.json"
MODEL_PATH = f"{DATA_DIR}/model_bundle.joblib"
TRADE_LOG_PATH = f"{DATA_DIR}/trade_log.csv"

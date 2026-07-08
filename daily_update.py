# -*- coding: utf-8 -*-
"""
Run once per trading day after the latest Yahoo Finance daily FX close is
available. Safe to re-run: it no-ops when there is no new feature day and
replays missed feature days in order.

The XGBoost probability model retrains weekly on an expanding window using
only completed rows before the current decision day. Today's decision is the
exposure that earns the next feature day's spread return.
"""

import json
import os

import pandas as pd

from config import (
    CONSECUTIVE_LOSS_LIMIT,
    COOLDOWN_DAYS,
    EXPOSURE_HYSTERESIS,
    MAX_ABS_EXPOSURE,
    MAX_DAILY_LOSS,
    MAX_DRAWDOWN_STOP,
    MIN_HISTORY_DAYS,
    MIN_SIZE,
    MODEL_PATH,
    PROB_RESET_THRESH,
    PROB_THRESH,
    SLIP_BPS,
    STATE_PATH,
    TC_BPS,
    TRADE_LOG_PATH,
)
from fx_lib import (
    calculate_decision,
    load_model,
    predict_probability,
    prepare_feature_frame,
    save_model,
    train_model,
)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "prev_exposure": None,
        "prev_prob_signal": 0,
        "pending_cost": 0.0,
        "equity": 1.0,
        "peak_equity": 1.0,
        "consecutive_losses": 0,
        "cooldown_days_remaining": 0,
        "risk_stop_active": False,
        "risk_reason": None,
        "last_decision_date": None,
        "last_train_week": None,
        "last_train_date": None,
    }


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def iso_week_key(ts):
    iso = pd.Timestamp(ts).isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def should_retrain(state, decision_date, model_bundle):
    if model_bundle is None:
        return True
    return state.get("last_train_week") != iso_week_key(decision_date)


def main():
    state = load_state()

    print("Downloading latest FX prices...")
    features = prepare_feature_frame()
    valid_dates = features.index

    last_decision_date = state.get("last_decision_date")
    if last_decision_date is None:
        dates_to_process = valid_dates[-1:]
    else:
        last_ts = pd.Timestamp(last_decision_date)
        dates_to_process = valid_dates[valid_dates > last_ts]

    if len(dates_to_process) == 0:
        print(f"No new feature day since {last_decision_date}. Nothing to do.")
        return

    model_bundle = load_model(MODEL_PATH)
    prev_exposure = state.get("prev_exposure")
    prev_prob_signal = int(state.get("prev_prob_signal") or 0)
    pending_cost = float(state.get("pending_cost") or 0.0)
    equity = float(state.get("equity") or 1.0)
    peak_equity = float(state.get("peak_equity") or equity)
    consecutive_losses = int(state.get("consecutive_losses") or 0)
    cooldown_days_remaining = int(state.get("cooldown_days_remaining") or 0)
    risk_stop_active = bool(state.get("risk_stop_active") or False)
    risk_reason = state.get("risk_reason")

    for decision_date in dates_to_process:
        history = features.loc[features.index < decision_date]
        if len(history) < MIN_HISTORY_DAYS:
            print(
                f"{decision_date.date()} only {len(history)} training rows available "
                f"(< MIN_HISTORY_DAYS={MIN_HISTORY_DAYS}); skipping decision."
            )
            continue

        retrained = False
        if should_retrain(state, decision_date, model_bundle):
            model_bundle = train_model(history)
            model_bundle["trained_on"] = decision_date.date().isoformat()
            model_bundle["trained_through"] = history.index[-1].date().isoformat()
            model_bundle["trained_week"] = iso_week_key(decision_date)
            model_bundle["n_train_rows"] = int(len(history))
            save_model(model_bundle, MODEL_PATH)
            state["last_train_week"] = model_bundle["trained_week"]
            state["last_train_date"] = model_bundle["trained_on"]
            retrained = True
            print(
                f"{decision_date.date()} retrained model on {len(history)} rows "
                f"through {history.index[-1].date()}."
            )

        row = features.loc[decision_date]
        realized_return = None
        drawdown = (equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0
        if prev_exposure is not None:
            realized_return = float(prev_exposure * row["spread_ret"] - pending_cost)
            equity *= 1.0 + realized_return
            peak_equity = max(peak_equity, equity)
            drawdown = (equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0
            consecutive_losses = consecutive_losses + 1 if realized_return < 0 else 0
            print(
                f"{decision_date.date()} realized return={realized_return:.4%} "
                f"equity={equity:.4f}"
            )
        else:
            print(f"{decision_date.date()} bootstrap day, no prior exposure yet.")

        risk_action = "none"
        if realized_return is not None and realized_return <= -MAX_DAILY_LOSS:
            cooldown_days_remaining = max(cooldown_days_remaining, COOLDOWN_DAYS)
            risk_reason = "max_daily_loss"
            risk_action = "cooldown"

        if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            cooldown_days_remaining = max(cooldown_days_remaining, COOLDOWN_DAYS)
            risk_reason = "consecutive_losses"
            risk_action = "cooldown"

        if drawdown <= -MAX_DRAWDOWN_STOP:
            risk_stop_active = True
            risk_reason = "max_drawdown_stop"
            risk_action = "hard_stop"

        prob = predict_probability(row, model_bundle)
        prob_thresh = float(model_bundle.get("prob_thresh", PROB_THRESH))
        prob_reset_thresh = float(model_bundle.get("prob_reset_thresh", PROB_RESET_THRESH))
        risk_active = risk_stop_active or cooldown_days_remaining > 0

        if risk_active:
            prior = 0.0 if prev_exposure is None else float(prev_exposure)
            decision = {
                "prob_signal": 0,
                "signal": 0,
                "target_exposure": 0.0,
                "exposure": 0.0,
                "turnover": abs(prior),
            }
            if risk_action == "none":
                risk_action = "hard_stop" if risk_stop_active else "cooldown"
        else:
            decision = calculate_decision(
                row=row,
                prob=prob,
                prev_exposure=prev_exposure,
                prev_prob_signal=prev_prob_signal,
                prob_thresh=prob_thresh,
                prob_reset_thresh=prob_reset_thresh,
                min_size=MIN_SIZE,
                hysteresis=EXPOSURE_HYSTERESIS,
                exposure_cap=MAX_ABS_EXPOSURE,
            )
        cost = decision["turnover"] * (TC_BPS + SLIP_BPS)
        cooldown_after_decision = cooldown_days_remaining
        if cooldown_days_remaining > 0 and not risk_stop_active:
            cooldown_days_remaining = max(0, cooldown_days_remaining - 1)

        append_csv(
            TRADE_LOG_PATH,
            {
                "date": decision_date.date().isoformat(),
                "realized_return": realized_return,
                "equity": equity,
                "prob": prob,
                "prob_signal": decision["prob_signal"],
                "signal": decision["signal"],
                "target_exposure": decision["target_exposure"],
                "exposure": decision["exposure"],
                "turnover": decision["turnover"],
                "cost_next_day": cost,
                "prob_thresh": prob_thresh,
                "prob_reset_thresh": prob_reset_thresh,
                "spread": row["SPREAD"],
                "spread_z": row["spread_z"],
                "z_entry_dynamic": row["z_entry_dynamic"],
                "capital_weight": row["capital_weight"],
                "spread_ret": row["spread_ret"],
                "eurusd": row["EURUSD"],
                "usdinr": row["USDINR"],
                "eurinr": row["EURINR"],
                "implied_eurusd": row["IMPLIED_EURUSD"],
                "retrained": retrained,
                "trained_week": state.get("last_train_week"),
                "trained_through": model_bundle.get("trained_through"),
                "n_train_rows": model_bundle.get("n_train_rows"),
                "threshold_validation_sharpe": model_bundle.get("threshold_validation_sharpe"),
                "threshold_validation_return": model_bundle.get("threshold_validation_return"),
                "threshold_validation_trades": model_bundle.get("threshold_validation_trades"),
                "drawdown": drawdown,
                "peak_equity": peak_equity,
                "consecutive_losses": consecutive_losses,
                "cooldown_days_remaining": cooldown_after_decision,
                "risk_stop_active": risk_stop_active,
                "risk_action": risk_action,
                "risk_reason": risk_reason,
            },
        )

        prev_exposure = decision["exposure"]
        prev_prob_signal = decision["prob_signal"]
        pending_cost = cost

    state.update(
        {
            "prev_exposure": prev_exposure,
            "prev_prob_signal": prev_prob_signal,
            "pending_cost": pending_cost,
            "equity": equity,
            "peak_equity": peak_equity,
            "consecutive_losses": consecutive_losses,
            "cooldown_days_remaining": cooldown_days_remaining,
            "risk_stop_active": risk_stop_active,
            "risk_reason": risk_reason,
            "last_decision_date": dates_to_process[-1].date().isoformat(),
        }
    )
    save_state(state)
    print("State saved. Last decision date:", state["last_decision_date"])


if __name__ == "__main__":
    main()
